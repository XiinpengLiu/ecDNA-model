import importlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config as cfg
from core.simulation import SimulationResult
from fit.data import CanonicalFitDataset, ConditionSpec, CountRecord, EcTAGRecord, FlowRecord, QPCDRRecord, load_count_csv
from fit.full_calibration import FullCalibrationRunner, FullCalibrationSettings, write_full_calibration_reports
from fit.v4_lite import (
    FullToLiteProjection,
    V4_LITE_STAGE_SEQUENCE,
    V4LiteFakeDataRecoveryReport,
    V4LiteFitRunner,
    V4LiteLeaveOneWeekOutReport,
    V4LiteObjective,
    V4LiteOptimizationSettings,
    V4LiteParameters,
    V4LitePosteriorSamples,
    V4LiteStageFitResult,
    _copy_log_signals,
    _expected_qpcdr_value,
    build_prior_diagnostics_report,
    build_parameter_status_table,
    build_v4_lite_reports,
    build_v4_lite_tensor,
    predict_v4_lite,
    project_full_to_lite,
    run_leave_one_week_out,
    write_v4_lite_reports,
)


def make_dataset(*, qpcdr_scale: str = "copy_number", same_cell_ectag: bool = False) -> CanonicalFitDataset:
    conditions = {"ctrl": ConditionSpec("ctrl")}
    flow = []
    for week, fractions in ((1, (0.40, 0.30, 0.20, 0.10)), (2, (0.35, 0.35, 0.20, 0.10)), (3, (0.30, 0.38, 0.22, 0.10))):
        for state_name, fraction in zip(cfg.STATE_NAMES, fractions):
            flow.append(
                FlowRecord(
                    condition="ctrl",
                    week=week,
                    state=state_name,
                    count=int(round(fraction * 1000)),
                    fraction=fraction,
                    total_events=1000,
                    replicate_id="r1",
                )
            )
    counts = (CountRecord("ctrl", 2, 1100.0, "r1"), CountRecord("ctrl", 3, 1200.0, "r1"))
    qpcdr = []
    for week in (2, 3):
        for state_name in cfg.STATE_NAMES:
            for species_index, species_name in enumerate(cfg.SPECIES):
                qpcdr.append(
                    QPCDRRecord(
                        "ctrl",
                        week,
                        state_name,
                        species_name,
                        2.0 + 0.25 * species_index + 0.1 * week,
                        "r1",
                        value_scale=qpcdr_scale,
                    )
                )
    ectag = []
    for week in (2, 3):
        for state_name in cfg.STATE_NAMES:
            for species_name in cfg.SPECIES:
                for cell_index in range(6):
                    cell_id = f"cell{cell_index}" if same_cell_ectag else f"{species_name}-cell{cell_index}"
                    ectag.append(EcTAGRecord("ctrl", week, state_name, species_name, cell_id, cell_index % 5, "r1"))
    week1 = {"ctrl": {}}
    for state_index, state_name in enumerate(cfg.STATE_NAMES):
        week1["ctrl"][state_name] = np.array(
            [[1 + state_index, 2, 3], [2 + state_index, 3, 4], [3 + state_index, 4, 5], [4 + state_index, 5, 6]],
            dtype=int,
        )
    return CanonicalFitDataset(
        conditions=conditions,
        flow=tuple(flow),
        counts=counts,
        qpcdr=tuple(qpcdr),
        ectag=tuple(ectag),
        week1_copy_distributions=week1,
    )


def make_dataset_with_counts(counts) -> CanonicalFitDataset:
    dataset = make_dataset()
    return CanonicalFitDataset(
        conditions=dataset.conditions,
        flow=dataset.flow,
        counts=tuple(counts),
        qpcdr=dataset.qpcdr,
        ectag=dataset.ectag,
        week1_copy_distributions=dataset.week1_copy_distributions,
        purity_matrix=dataset.purity_matrix,
        purity_sensitivity=dataset.purity_sensitivity,
        qpcdr_calibration=dataset.qpcdr_calibration,
    )


def make_simulation_result_with_events() -> SimulationResult:
    first_state = np.eye(cfg.N_STATES)[0]
    second_state = np.eye(cfg.N_STATES)[1]
    return SimulationResult(
        times=[0.0, 1.0],
        population_sizes=[2, 2],
        soft_state_fractions=[np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.25, 0.75, 0.0, 0.0])],
        cycle_fractions=[],
        bulk_copy_means=[],
        truth_snapshots=[
            {"population_size": 2, "soft_state_fractions": [0.5, 0.5, 0.0, 0.0]},
            {"population_size": 2, "soft_state_fractions": [0.25, 0.75, 0.0, 0.0]},
        ],
        observations=[],
        cell_snapshots=[
            [
                {"soft_state": first_state, "copy_numbers": np.array([1, 2, 3])},
                {"soft_state": second_state, "copy_numbers": np.array([2, 3, 4])},
            ],
            [
                {"soft_state": first_state, "copy_numbers": np.array([1, 2, 3])},
                {"soft_state": second_state, "copy_numbers": np.array([3, 4, 5])},
            ],
        ],
        events=[
            (
                0.5,
                "transition",
                1,
                {
                    "state_pre": {"soft_state": first_state, "copy_numbers": np.array([1, 2, 3])},
                    "state_post": {"soft_state": second_state, "copy_numbers": np.array([2, 3, 4])},
                },
            )
        ],
    )


class V4LiteTests(unittest.TestCase):
    def test_compile_and_import(self):
        fit_module = importlib.import_module("fit")
        v4_module = importlib.import_module("fit.v4_lite")
        self.assertTrue(hasattr(fit_module, "V4LiteFitRunner"))
        self.assertTrue(hasattr(fit_module, "FullCalibrationRunner"))
        self.assertEqual(v4_module.V4LiteDynamicsMode, {"joint", "ecDNA_only", "state_only"})

    def test_count_record_total_backward_compatibility(self):
        csv_text = "condition,week,count,replicate_id\nctrl,2,1100,r1\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=csv_text)):
            rows = load_count_csv(Path.cwd() / "counts.csv")
        self.assertIsNone(rows[0].gate)
        tensor = build_v4_lite_tensor(make_dataset_with_counts(rows))
        self.assertIn("count_total", tensor.observed_summary.blocks)
        self.assertNotIn("count_gate", tensor.observed_summary.blocks)

    def test_count_record_gate_recovery(self):
        dataset = make_dataset()
        gated = dataset.counts + (CountRecord("ctrl", 2, 350.0, "r1", gate=cfg.STATE_NAMES[0]),)
        tensor = build_v4_lite_tensor(make_dataset_with_counts(gated))
        params = V4LiteParameters.default(tensor.structure)
        prediction = predict_v4_lite(tensor, params)
        self.assertIn("count_gate", tensor.observed_summary.blocks)
        self.assertIn("count_gate", prediction.summary.blocks)
        objective = V4LiteObjective(tensor=tensor, active_groups=("observation", "state_abundance"), model_version="M1")
        block_names = {block.name for block in objective.evaluate_vector(objective.adapter.default_vector()).block_results}
        self.assertIn("count_gate", block_names)

    def test_week1_initialization(self):
        tensor = build_v4_lite_tensor(make_dataset())
        abundance = tensor.initial_state_abundance["ctrl"]
        distributions = tensor.initial_copy_distributions["ctrl"]
        self.assertEqual(abundance.shape, (cfg.N_STATES,))
        self.assertEqual(distributions.shape, (cfg.N_STATES, cfg.N_SPECIES, tensor.structure.binning.n_bins))
        self.assertAlmostEqual(float(np.sum(abundance)), 1000.0)
        self.assertTrue(np.allclose(np.sum(distributions, axis=2), 1.0))

    def test_log_copy_summary_formula(self):
        tensor = build_v4_lite_tensor(make_dataset())
        distributions = tensor.initial_copy_distributions["ctrl"]
        signals = _copy_log_signals(distributions, tensor.structure.binning)
        manual = float(np.dot(distributions[0, 0, :], np.log1p(tensor.structure.binning.centers)))
        self.assertAlmostEqual(float(signals[0, 0]), manual)

    def test_ecdna_only_no_state_mixing(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        params.mobility_log[:] = math.log(0.8)
        joint = predict_v4_lite(tensor, params, dynamics_mode="joint")
        ecdna_only = predict_v4_lite(tensor, params, dynamics_mode="ecDNA_only")
        self.assertFalse(np.allclose(joint.copy_distributions["ctrl"][1], ecdna_only.copy_distributions["ctrl"][1]))
        self.assertTrue(np.allclose(ecdna_only.transition_matrices["ctrl"][0], np.eye(cfg.N_STATES)))

    def test_state_only_uses_frozen_copy_distributions(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        frozen = predict_v4_lite(tensor, params, dynamics_mode="ecDNA_only").copy_distributions
        params.kernel_up_species[:] = 5.0
        prediction = predict_v4_lite(tensor, params, dynamics_mode="state_only", frozen_copy_distributions=frozen)
        self.assertTrue(np.allclose(prediction.copy_distributions["ctrl"], frozen["ctrl"]))

    def test_observation_calibration_without_replicates(self):
        runner = V4LiteFitRunner(make_dataset())
        result = runner.run_stage(V4_LITE_STAGE_SEQUENCE[0])
        self.assertTrue(result.accepted)
        calibration = result.diagnostics["calibration"]
        self.assertTrue(calibration["qpcdr"]["insufficient_replicates"])
        self.assertTrue(calibration["flow"]["insufficient_replicates"])

    def test_qpcdr_copy_and_ct_modes(self):
        copy_tensor = build_v4_lite_tensor(make_dataset(qpcdr_scale="copy_number"))
        copy_params = V4LiteParameters.default(copy_tensor.structure)
        copy_pred = predict_v4_lite(copy_tensor, copy_params)
        copy_expected = _expected_qpcdr_value(copy_tensor, copy_params, copy_pred, copy_tensor.qpcdr_observations[0])
        self.assertGreater(copy_expected, copy_params.qpcdr_intercept[0])

        ct_tensor = build_v4_lite_tensor(make_dataset(qpcdr_scale="ct"))
        ct_params = V4LiteParameters.default(ct_tensor.structure)
        ct_params.qpcdr_intercept[:] = 10.0
        ct_pred = predict_v4_lite(ct_tensor, ct_params)
        ct_expected = _expected_qpcdr_value(ct_tensor, ct_params, ct_pred, ct_tensor.qpcdr_observations[0])
        self.assertLess(ct_expected, 10.0)

    def test_m3_skips_without_same_cell_ectag(self):
        runner = V4LiteFitRunner(make_dataset())
        m3_stage = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M3-co-segregation")
        result = runner.run_stage(m3_stage)
        self.assertFalse(result.accepted)
        self.assertIn("No same-cell", result.skipped_reason)

    def test_leave_one_week_out_excludes_heldout(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("exposure", "ecDNA_kernel", "state_abundance"),
            model_version="M1",
        )
        vector = objective.adapter.default_vector()

        def optimizer(obj, initial, n_restarts):
            return np.asarray(initial, dtype=float), obj.evaluate_vector(initial).total_objective

        heldout_objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("exposure", "ecDNA_kernel", "state_abundance"),
            model_version="M1",
            heldout_weeks=(2,),
        )
        full_dimension = sum(block.dimension for block in objective.evaluate_vector(vector).block_results)
        train_dimension = sum(block.dimension for block in heldout_objective.evaluate_vector(vector).block_results)
        self.assertLess(train_dimension, full_dimension)
        report = run_leave_one_week_out(objective, vector, optimizer, n_restarts=1)
        self.assertIn(2, report.heldout_scores)

    def test_parameter_status_boundary_margin(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        params.omega_O_given_C = 1e-9
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("state_abundance",),
            model_version="M1",
            base_params=params,
        )
        vector = objective.adapter.pack(params)
        samples = np.tile(vector, (4, 1))
        posterior = V4LitePosteriorSamples(objective.adapter.parameter_names(), samples, 1.0)
        fake = V4LiteFakeDataRecoveryReport(0.0, 0.0, True, {})
        table = build_parameter_status_table(objective, vector, (), fake, posterior)
        omega_row = next(row for row in table if row["field"] == "omega_O_given_C")
        self.assertLess(float(omega_row["boundary_margin"]), 0.02)
        self.assertIn("boundary warning", omega_row["rationale"])

    def test_prior_diagnostics_report_flags_approximations(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("ecDNA_kernel", "state_abundance"),
            model_version="M1",
        )
        report = build_prior_diagnostics_report(objective, objective.adapter.default_vector())
        kinds = {field["prior_kind"] for field in report["active_fields"]}
        self.assertIn("gaussian_shrinkage_approximation", kinds)
        self.assertEqual(report["strict_horseshoe_prior"], "not_implemented_strictly")

    def test_full_projection_with_snapshots_and_events(self):
        projection = project_full_to_lite(make_simulation_result_with_events())
        self.assertEqual(projection.state_abundance.shape[0], 2)
        self.assertIsNotNone(projection.transition_matrices)
        self.assertIsNotNone(projection.growth_rates)
        self.assertIsNotNone(projection.copy_kernels)

    def test_full_calibration_f0_runs(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        result = FullCalibrationRunner(make_dataset(), target).run_f0_from_simulation_result(simulation_result)
        self.assertTrue(result.stage_results[0].accepted)
        self.assertEqual(result.stage_results[0].stage_name, "F0-skeleton")
        self.assertEqual(result.coarse_residual_report["calibration_mode"], "v4_lite_summary_target")

    def test_full_calibration_f1_optimizes_parameters(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        runner = FullCalibrationRunner(make_dataset(), target, settings=FullCalibrationSettings(maxiter=8, run_formal_raw_refinement=False))

        def fake_project(bundle, condition_name):
            alpha0 = float(bundle.model.landscape.alpha[0])
            shifted = target.state_abundance.copy()
            shifted[:, 0] += alpha0
            return FullToLiteProjection(
                target.weeks,
                shifted,
                target.copy_distributions,
                target.transition_matrices,
                target.growth_rates,
                target.copy_kernels,
                {"mock": True},
            )

        with mock.patch.object(runner, "_project_bundle", side_effect=fake_project):
            stage, bundle, _projection = runner._optimize_calibration_stage(
                runner.bundle,
                "ctrl",
                "F1-state-landscape",
                {"state_abundance": 1.0, "transition_matrix": 0.0, "copy_distribution": 0.0, "growth_rate": 0.0, "copy_kernel": 0.0},
            )
        self.assertTrue(stage.accepted)
        self.assertLessEqual(stage.objective_after, stage.objective_before)
        self.assertNotAlmostEqual(float(bundle.model.landscape.alpha[0]), float(runner.bundle.model.landscape.alpha[0]))

    def test_formal_raw_refinement_mode_uses_raw_report(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        runner = FullCalibrationRunner(make_dataset(), target, settings=FullCalibrationSettings(formal_maxiter=2))

        with mock.patch.object(runner, "_project_bundle", return_value=target), mock.patch.object(
            runner,
            "_formal_raw_observation_report",
            return_value={"mode": "direct_raw_observation_map", "weighted_relative_rmse": 0.0, "rows": []},
        ):
            stage, _bundle, _projection, report = runner._run_formal_raw_refinement(runner.bundle, "ctrl")
        self.assertEqual(stage.stage_name, "F-formal-raw-MAP")
        self.assertTrue(stage.accepted)
        self.assertEqual(report["mode"], "direct_raw_observation_map")

    def test_full_calibration_skips_f4_f5_by_default(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        result = FullCalibrationRunner(make_dataset(), target).run_f0_from_simulation_result(simulation_result)
        self.assertIn("F4-RV", result.skipped_stages)
        self.assertIn("F5-co-segregation-daughter-memory", result.skipped_stages)

    def test_no_double_counting_mode_labels(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        result = FullCalibrationRunner(make_dataset(), target).run_f0_from_simulation_result(simulation_result)
        self.assertEqual(result.mode_label, "calibration_not_formal_full_bayesian_posterior")
        self.assertEqual(result.coarse_residual_report["formal_inference_mode"], "not_run")
        self.assertIn("double_counting_policy", result.coarse_residual_report)

    def test_report_files_written(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        prediction = predict_v4_lite(tensor, params)
        fake_recovery = V4LiteFakeDataRecoveryReport(0.0, 0.0, True, {})
        loo = V4LiteLeaveOneWeekOutReport({2: 0.1})
        stage_results = [V4LiteStageFitResult("observation", ("observation",), (), None, None, (), accepted=True)]
        reports = build_v4_lite_reports(tensor, prediction, stage_results, (), fake_recovery, loo, None, {"M1": 0.0}, {}, ())
        written = set()

        def fake_write_text(path, text, *args, **kwargs):
            written.add(Path(path).name)
            return len(text)

        def fake_open(path, *args, **kwargs):
            written.add(Path(path).name)
            return mock.mock_open()(path, *args, **kwargs)

        with mock.patch.object(Path, "mkdir", return_value=None), mock.patch.object(Path, "write_text", fake_write_text), mock.patch("builtins.open", side_effect=fake_open):
            output_dir = Path("unused-report-dir")
            write_v4_lite_reports(output_dir, reports, (), {"M1": 0.0}, write_optional_plots=False)
            expected = {
                "v4_lite_reports.json",
                "parameter_status.csv",
                "parameter_status.json",
                "cleaned_tensor_summary.json",
                "observation_calibration_report.json",
                "ecDNA_only_report.json",
                "identifiability_report.json",
                "posterior_predictive_report.json",
                "count_observation_report.json",
                "prior_diagnostics_report.json",
                "implementation_status_report.json",
                "posterior_predictive_residuals.csv",
            }
            self.assertTrue(expected.issubset(written))

    def test_full_calibration_report_files_written(self):
        simulation_result = make_simulation_result_with_events()
        target = project_full_to_lite(simulation_result)
        result = FullCalibrationRunner(make_dataset(), target).run_f0_from_simulation_result(simulation_result)
        written = set()

        def fake_write_text(path, text, *args, **kwargs):
            written.add(Path(path).name)
            return len(text)

        def fake_open(path, *args, **kwargs):
            written.add(Path(path).name)
            return mock.mock_open()(path, *args, **kwargs)

        with mock.patch.object(Path, "mkdir", return_value=None), mock.patch.object(Path, "write_text", fake_write_text), mock.patch("builtins.open", side_effect=fake_open):
            write_full_calibration_reports(Path("unused-full-report-dir"), result)
        self.assertIn("full_calibration_report.json", written)
        self.assertIn("full_coarse_residuals.csv", written)

    def test_no_new_dependency_requirement(self):
        tensor = build_v4_lite_tensor(make_dataset())
        prediction = predict_v4_lite(tensor, V4LiteParameters.default(tensor.structure))
        fake_recovery = V4LiteFakeDataRecoveryReport(0.0, 0.0, True, {})
        reports = build_v4_lite_reports(tensor, prediction, [], (), fake_recovery, V4LiteLeaveOneWeekOutReport({}), None, {}, {}, ())
        saved_matplotlib = sys.modules.get("matplotlib")
        sys.modules["matplotlib"] = None
        try:
            with mock.patch.object(Path, "mkdir", return_value=None), mock.patch.object(Path, "write_text", lambda path, text, *args, **kwargs: len(text)), mock.patch("builtins.open", mock.mock_open()):
                write_v4_lite_reports(Path("unused-report-dir"), reports, (), {}, write_optional_plots=True)
        finally:
            if saved_matplotlib is None:
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = saved_matplotlib


if __name__ == "__main__":
    unittest.main()
