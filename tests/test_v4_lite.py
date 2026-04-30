import importlib
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config as cfg
from core.simulation import SimulationResult
from fit.data import CanonicalFitDataset, ConditionSpec, CountRecord, DDPCRRecord, EcTAGRecord, FlowRecord, QPCDRRecord, load_count_csv
from fit.full_calibration import FullCalibrationRunner, FullCalibrationSettings, _projection_rows, full_model_capability_report, write_full_calibration_reports
from fit.v4_lite import (
    FullToLiteProjection,
    QPCDR_COPY_EPSILON,
    V4_LITE_STAGE_SEQUENCE,
    V4LiteFakeDataRecoveryReport,
    V4LiteFitResult,
    V4LiteFitRunner,
    V4LiteLeaveOneWeekOutReport,
    V4LiteObjective,
    V4LiteOptimizationSettings,
    V4LiteParameters,
    V4LitePosteriorSamples,
    V4LiteStageFitResult,
    V4LiteStructure,
    _copy_log_signals,
    _expected_qpcdr_value,
    _posterior_predictive_interval_rows,
    _projection_targets_from_prediction,
    _stage_criteria,
    build_lite_release_table_rows,
    build_obs_params_for_full,
    build_prior_diagnostics_report,
    build_parameter_status_table,
    build_v4_lite_reports,
    build_v4_lite_tensor,
    predict_v4_lite,
    project_full_to_lite,
    run_leave_one_week_out,
    run_v4_lite_fake_data_recovery,
    run_v4_lite_prior_predictive,
    sample_prior_parameters,
    write_fit_method_artifacts,
    write_v4_lite_reports,
)


def make_dataset(*, qpcdr_scale: str = "copy_number", same_cell_ectag: bool = False, ddpcr=(), schedules=None) -> CanonicalFitDataset:
    conditions = {"ctrl": ConditionSpec("ctrl", schedules or {})}
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
        ddpcr=tuple(ddpcr),
        week1_copy_distributions=week1,
    )


def make_two_condition_dataset() -> CanonicalFitDataset:
    base = make_dataset()
    conditions = {
        "ctrl": base.conditions["ctrl"],
        "drug": ConditionSpec("drug", {"u_C": ((0.0, 1.0),)}),
    }
    flow = base.flow + tuple(FlowRecord("drug", row.week, row.state, row.count, row.fraction, row.total_events, row.replicate_id) for row in base.flow)
    counts = base.counts + tuple(CountRecord("drug", row.week, row.value, row.replicate_id, row.gate) for row in base.counts)
    qpcdr = base.qpcdr + tuple(QPCDRRecord("drug", row.week, row.state, row.species, row.value, row.replicate_id, row.batch, row.value_scale) for row in base.qpcdr)
    ectag = base.ectag + tuple(EcTAGRecord("drug", row.week, row.state, row.species, f"drug-{row.cell_id}", row.value, row.replicate_id) for row in base.ectag)
    week1 = {
        "ctrl": {state: values.copy() for state, values in base.week1_copy_distributions["ctrl"].items()},
        "drug": {state: values.copy() for state, values in base.week1_copy_distributions["ctrl"].items()},
    }
    return CanonicalFitDataset(
        conditions=conditions,
        flow=flow,
        counts=counts,
        qpcdr=qpcdr,
        ectag=ectag,
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

    def test_manifest_ectag_hist_max_requires_metadata_flag(self):
        manifest = json.dumps({"conditions": {"ctrl": {}}, "ectag_hist_max": 30})
        with mock.patch.object(Path, "read_text", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "experimental metadata"):
                CanonicalFitDataset.from_manifest(Path.cwd() / "manifest.json")

    def test_manifest_accepts_metadata_ectag_hist_max(self):
        dataset = make_dataset()
        week1 = {
            condition: {state: values.tolist() for state, values in by_state.items()}
            for condition, by_state in dataset.week1_copy_distributions.items()
        }
        manifest = {
            "conditions": {"ctrl": {}},
            "files": {
                "flow": "flow.csv",
                "counts": "counts.csv",
                "qpcdr": "qpcdr.csv",
                "ectag": "ectag.csv",
                "week1_copy_distributions": "week1.json",
            },
            "ectag_hist_max": 30,
            "ectag_hist_max_from_metadata": True,
        }

        def fake_read_text(path, *args, **kwargs):
            if Path(path).name == "week1.json":
                return json.dumps(week1)
            return json.dumps(manifest)

        with mock.patch.object(Path, "read_text", fake_read_text), mock.patch("fit.data.load_flow_csv", return_value=dataset.flow), mock.patch("fit.data.load_count_csv", return_value=dataset.counts), mock.patch("fit.data.load_qpcdr_csv", return_value=dataset.qpcdr), mock.patch("fit.data.load_ectag_csv", return_value=dataset.ectag):
            loaded = CanonicalFitDataset.from_manifest(Path.cwd() / "manifest.json")
        self.assertEqual(loaded.ectag_hist_max, 30)

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

    def test_week1_ddpcr_anchor_is_in_likelihood(self):
        dataset = make_dataset(ddpcr=(DDPCRRecord("ctrl", 1, cfg.SPECIES[0], 3.0, replicate_id="r1"),))
        tensor = build_v4_lite_tensor(dataset)
        self.assertEqual([obs.week for obs in tensor.ddpcr_observations], [1])
        self.assertIn(f"ctrl|week1|species={cfg.SPECIES[0]}|rep=r1", tensor.observed_summary.blocks["ddpcr_pooled_mean"].keys)

        objective = V4LiteObjective(tensor=tensor, active_groups=("observation",), model_version="M0")
        block = next(result for result in objective.evaluate_vector(objective.adapter.default_vector()).block_results if result.name == "ddpcr_pooled_mean")
        self.assertEqual(block.dimension, 1)

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

    def test_adapter_releases_state_kernel_without_invalid_lite_params(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(tensor=tensor, active_groups=("ecDNA_kernel", "state_abundance"), model_version="M1")
        names = set(objective.adapter.parameter_names())
        self.assertIn("kernel_up_state[0]", names)
        self.assertIn("kernel_down_state[0]", names)
        self.assertNotIn("omega_O_given_C", names)
        self.assertNotIn("exposure_C_scale", names)
        self.assertNotIn("burden_loss_effect", names)
        self.assertNotIn("drug_gain_effect_C", names)

    def test_drug_effects_are_conditional_and_use_exposure(self):
        tensor = build_v4_lite_tensor(make_dataset(schedules={"u_C": ((0.0, 1.0),)}))
        objective = V4LiteObjective(tensor=tensor, active_groups=("drug_effects",), model_version="M1")
        self.assertIn("drug_gain_effect_C", objective.adapter.parameter_names())

        baseline = predict_v4_lite(tensor, V4LiteParameters.default(tensor.structure), coupling_mode="none")
        params = V4LiteParameters.default(tensor.structure)
        params.drug_gain_effect_C = 1.0
        params.drug_growth_effect_C = 0.5
        params.drug_transition_effect_C = 1.0
        exposed = predict_v4_lite(tensor, params, coupling_mode="none")

        self.assertFalse(np.allclose(exposed.copy_kernels["ctrl"], baseline.copy_kernels["ctrl"]))
        self.assertFalse(np.allclose(exposed.growth_rates["ctrl"], baseline.growth_rates["ctrl"]))
        self.assertFalse(np.allclose(exposed.transition_matrices["ctrl"], baseline.transition_matrices["ctrl"]))

    def test_observation_calibration_without_replicates(self):
        runner = V4LiteFitRunner(make_dataset())
        result = runner.run_stage(V4_LITE_STAGE_SEQUENCE[0])
        self.assertTrue(result.accepted)
        calibration = result.diagnostics["calibration"]
        self.assertTrue(calibration["qpcdr"]["insufficient_replicates"])
        self.assertTrue(calibration["flow"]["insufficient_replicates"])

    def test_run_stage_stores_stage_map_outputs(self):
        settings = V4LiteOptimizationSettings(maxiter=1, n_restarts=1, stage_ppc_draws=4)
        runner = V4LiteFitRunner(make_dataset(), optimization_settings=settings)
        stage = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M0-observation-only")
        result = runner.run_stage(stage)
        self.assertIsNotNone(result.best_params)
        self.assertIsNotNone(result.prediction)
        self.assertIsNotNone(result.posterior_or_map_vector)

    def test_qpcdr_copy_and_ct_modes(self):
        copy_tensor = build_v4_lite_tensor(make_dataset(qpcdr_scale="copy_number"))
        copy_params = V4LiteParameters.default(copy_tensor.structure)
        copy_params.qpcdr_intercept[:] = 0.3
        copy_params.qpcdr_slope[:] = 1.7
        copy_pred = predict_v4_lite(copy_tensor, copy_params)
        copy_obs = copy_tensor.qpcdr_observations[0]
        copy_expected = _expected_qpcdr_value(copy_tensor, copy_params, copy_pred, copy_obs)
        probs = copy_pred.copy_distributions[copy_obs.condition][copy_tensor.week_to_index[copy_obs.week], copy_obs.gate_index, copy_obs.species_index]
        mean_copy = max(copy_tensor.structure.binning.mean(probs), QPCDR_COPY_EPSILON)
        self.assertAlmostEqual(copy_expected, float(np.exp(copy_params.qpcdr_intercept[0] + copy_params.qpcdr_slope[0] * np.log(mean_copy))))
        self.assertNotAlmostEqual(copy_expected, float(copy_params.qpcdr_intercept[0] + copy_params.qpcdr_slope[0] * mean_copy))

        ct_tensor = build_v4_lite_tensor(make_dataset(qpcdr_scale="ct"))
        ct_params = V4LiteParameters.default(ct_tensor.structure)
        ct_params.qpcdr_intercept[:] = 10.0
        ct_pred = predict_v4_lite(ct_tensor, ct_params)
        ct_expected = _expected_qpcdr_value(ct_tensor, ct_params, ct_pred, ct_tensor.qpcdr_observations[0])
        self.assertLess(ct_expected, 10.0)

    def test_m3_skips_without_same_cell_ectag(self):
        initial = V4LiteParameters.default()
        initial.co_segregation_rho = 0.5
        runner = V4LiteFitRunner(make_dataset(), initial_params=initial)
        m3_stage = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M3-co-segregation")
        result = runner.run_stage(m3_stage)
        self.assertFalse(result.accepted)
        self.assertIn("No same-cell", result.skipped_reason)
        self.assertEqual(runner.current_params.co_segregation_rho, 0.0)

    def test_co_segregation_stage_only_scores_joint_correlation(self):
        runner = V4LiteFitRunner(make_dataset(same_cell_ectag=True))
        stage = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M3-co-segregation")
        objective = runner._objective_for_stage(stage)
        self.assertEqual(objective.adapter.parameter_names(), ("co_segregation_rho",))
        block_names = {block.name for block in objective.evaluate_vector(objective.adapter.default_vector()).block_results}
        self.assertEqual(block_names, {"ectag_corr"})

    def test_co_segregation_requires_strict_release_criteria(self):
        tensor = build_v4_lite_tensor(make_dataset(same_cell_ectag=True))
        stage = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M3-co-segregation")
        weak = {
            "correlation_ppc_improvement": 0.1,
            "posterior_sign_probability": 0.75,
            "posterior_contraction": 0.5,
            "synthetic_sign_recovery": 1.0,
            "marginal_hist_nll_delta": 0.0,
        }
        _criteria, failed = _stage_criteria(stage, tensor, 1.0, 0.9, (), weak)
        self.assertIn("posterior_sign_probability failed", failed)

        strong = dict(weak, posterior_sign_probability=0.95, posterior_contraction=0.31, synthetic_sign_recovery=0.8)
        _criteria, failed = _stage_criteria(stage, tensor, 1.0, 0.9, (), strong)
        self.assertEqual(failed, ())

    def test_mandatory_stage_failure_hard_stops_run_all(self):
        runner = V4LiteFitRunner(make_dataset())
        accepted = V4LiteStageFitResult("observation", ("observation",), None, None, None, (), True)
        init = V4LiteStageFitResult("week1-init-check", (), None, None, None, (), True)
        failed = V4LiteStageFitResult("M0-observation-only", ("observation",), None, 1.0, 2.0, ("stage_ppc failed",), False)
        with mock.patch.object(runner, "run_stage", side_effect=(accepted, init, failed)):
            with self.assertRaisesRegex(RuntimeError, "Mandatory stage M0-observation-only failed"):
                runner.run_all()

    def test_optional_coupling_improvement_uses_accepted_baseline(self):
        runner = V4LiteFitRunner(make_dataset())
        runner.stage_objective_after["M2-abundance-null"] = 100.0
        m3 = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M3-growth-coupling")
        self.assertEqual(runner._baseline_improvement_for_stage(m3, 90.0), 10.0)
        runner.accepted_growth_coupling = True
        runner.stage_objective_after["M3-growth-coupling"] = 80.0
        m4 = next(stage for stage in V4_LITE_STAGE_SEQUENCE if stage.name == "M4-transition-coupling")
        self.assertEqual(runner._baseline_improvement_for_stage(m4, 70.0), 10.0)

    def test_leave_one_week_out_excludes_heldout(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("ecDNA_kernel", "state_abundance"),
            model_version="M1",
        )
        vector = objective.adapter.default_vector()

        def optimizer(obj, initial, n_restarts):
            return np.asarray(initial, dtype=float), obj.evaluate_vector(initial).total_objective

        heldout_objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("ecDNA_kernel", "state_abundance"),
            model_version="M1",
            heldout_weeks=(2,),
        )
        full_dimension = sum(block.dimension for block in objective.evaluate_vector(vector).block_results)
        train_dimension = sum(block.dimension for block in heldout_objective.evaluate_vector(vector).block_results)
        self.assertLess(train_dimension, full_dimension)
        report = run_leave_one_week_out(objective, vector, optimizer, n_restarts=1)
        self.assertIn(2, report.heldout_scores)

    def test_posterior_predictive_intervals_use_replicated_observation_noise(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(tensor=tensor, active_groups=("observation",), model_version="M0")
        vector = objective.adapter.default_vector()
        posterior = V4LitePosteriorSamples(objective.adapter.parameter_names(), np.tile(vector, (4, 1)), 1.0)
        rows = _posterior_predictive_interval_rows(objective, posterior)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["interval_source"], "replicated_observation")

    def test_fake_data_recovery_uses_requested_synthetic_count(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(tensor=tensor, active_groups=("observation",), model_version="M0")
        vector = objective.adapter.default_vector()

        def optimizer(obj, initial, n_restarts):
            return np.asarray(initial, dtype=float), obj.evaluate_vector(initial).total_objective

        report = run_v4_lite_fake_data_recovery(objective, vector, optimizer, n_restarts=1, n_synthetic=3, ppc_draws=2)
        self.assertEqual(report.n_synthetic, 3)
        self.assertIsNotNone(report.coverage_rate)

    def test_parameter_status_boundary_margin(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        params.count_dispersion = 1e-9
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("observation",),
            model_version="M1",
            base_params=params,
        )
        vector = objective.adapter.pack(params)
        samples = np.tile(vector, (4, 1))
        posterior = V4LitePosteriorSamples(objective.adapter.parameter_names(), samples, 1.0)
        fake = V4LiteFakeDataRecoveryReport(0.0, 0.0, True, {})
        table = build_parameter_status_table(objective, vector, (), fake, posterior)
        count_row = next(row for row in table if row["field"] == "count_dispersion")
        self.assertLess(float(count_row["boundary_margin"]), 0.02)
        self.assertIn("boundary warning", count_row["rationale"])

    def test_prior_diagnostics_report_flags_biological_priors(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(
            tensor=tensor,
            active_groups=("ecDNA_kernel", "state_abundance"),
            model_version="M1",
        )
        report = build_prior_diagnostics_report(objective, objective.adapter.default_vector())
        kinds = {field["prior_kind"] for field in report["active_fields"]}
        self.assertIn("biological_shrinkage_normal_prior", kinds)
        self.assertIn("sample_prior_parameters", report["prior_policy"])
        self.assertEqual(report["strict_horseshoe_prior"], "not_implemented_strictly")

    def test_sample_prior_parameters_shapes_and_constraints(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = sample_prior_parameters(tensor.structure, np.random.default_rng(123))
        self.assertEqual(params.kernel_up_species.shape, (cfg.N_SPECIES,))
        self.assertEqual(params.mobility_log.shape, (tensor.structure.n_mobility_edges,))
        self.assertTrue(np.all(params.qpcdr_sigma > 0.0))
        self.assertTrue(np.all(params.ectag_concentration > 0.0))
        self.assertTrue(np.allclose(np.sum(params.sort_purity_matrix, axis=0), 1.0))

    def test_prior_predictive_uses_biological_sampler(self):
        tensor = build_v4_lite_tensor(make_dataset())
        objective = V4LiteObjective(tensor=tensor, active_groups=("ecDNA_kernel", "state_abundance"), model_version="M1")
        with mock.patch("fit.v4_lite.sample_prior_parameters", wraps=sample_prior_parameters) as sampler:
            report = run_v4_lite_prior_predictive(objective, n_draws=3, seed=11)
        self.assertEqual(sampler.call_count, 3)
        self.assertEqual(report.n_draws, 3)

    def test_full_projection_with_snapshots_and_events(self):
        projection = project_full_to_lite(make_simulation_result_with_events())
        self.assertEqual(projection.state_abundance.shape[0], 2)
        self.assertIsNotNone(projection.transition_matrices)
        self.assertIsNotNone(projection.growth_rates)
        self.assertIsNotNone(projection.copy_kernels)

    def test_full_projection_rows_use_copy_bin_centers(self):
        binning = V4LiteStructure.default().binning
        copies = np.zeros((1, cfg.N_STATES, cfg.N_SPECIES, binning.n_bins), dtype=float)
        copies[:, :, :, 3] = 1.0
        projection = FullToLiteProjection(
            (1,),
            np.ones((1, cfg.N_STATES), dtype=float),
            copies,
            None,
            None,
            None,
            {},
            tuple(float(value) for value in binning.centers),
        )
        rows = _projection_rows(projection)
        copy_row = next(row for row in rows if row["block"] == "copy_mean" and row["state"] == cfg.STATE_NAMES[0] and row["species"] == cfg.SPECIES[0])
        self.assertEqual(copy_row["value"], float(binning.centers[3]))
        self.assertNotEqual(copy_row["value"], 3.0)

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

    def test_release_table_includes_full_blocks(self):
        stage_results = (
            V4LiteStageFitResult("M0-observation-only", ("observation",), None, 2.0, 1.0, (), True),
            V4LiteStageFitResult("M1-ecDNA-kernel", ("ecDNA_kernel",), None, 2.0, 1.0, (), True),
        )
        rows = build_lite_release_table_rows(stage_results)
        blocks = {row["full_block"] for row in rows}
        self.assertTrue({"drug", "stress_survival", "exposure", "observation_calibration"}.issubset(blocks))
        observation = next(row for row in rows if row["full_block"] == "observation_calibration")
        self.assertTrue(observation["release"])

    def test_full_calibration_applies_lite_observation_params(self):
        dataset = make_dataset()
        tensor = build_v4_lite_tensor(dataset)
        prediction = predict_v4_lite(tensor, V4LiteParameters.default(tensor.structure))
        target = FullToLiteProjection(
            tensor.weeks,
            prediction.state_abundance["ctrl"],
            prediction.copy_distributions["ctrl"],
            prediction.transition_matrices["ctrl"],
            prediction.growth_rates["ctrl"],
            prediction.copy_kernels["ctrl"],
            {},
        )
        payload = {
            "qpcdr": {
                species: {"intercept": 1.0 + idx, "slope": 0.5 + idx, "sigma": 0.1 + 0.01 * idx}
                for idx, species in enumerate(cfg.SPECIES)
            },
            "flow": {"concentration": 100.0, "sort_purity_matrix": np.eye(cfg.N_STATES).tolist()},
            "counts": {"total_count_dispersion": 50.0, "gate_count_dispersion": 60.0},
            "ectag": {
                "concentration_by_species": {species: 100.0 + idx for idx, species in enumerate(cfg.SPECIES)},
                "same_cell_correlation_sigma": 0.3,
            },
        }
        runner = FullCalibrationRunner(
            dataset,
            target,
            structure=tensor.structure,
            release_table=({"full_block": "observation_calibration", "release": True},),
            obs_params_for_full=payload,
            settings=FullCalibrationSettings(run_formal_raw_refinement=False),
        )
        self.assertTrue(np.allclose(runner.bundle.observation.qpcdr_intercept, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(runner.bundle.observation.qpcdr_slope, [0.5, 1.5, 2.5]))
        self.assertTrue(np.allclose(runner.bundle.observation.qpcdr_sigma, [0.1, 0.11, 0.12]))

        lite_params = runner._lite_observation_params(tensor)
        self.assertTrue(np.allclose(lite_params.qpcdr_intercept, [1.0, 2.0, 3.0]))
        report = runner._formal_raw_observation_report(target, "ctrl", bundle=runner.bundle)
        self.assertEqual(report["observation_calibration_source"], "lite_obs_params_for_full")

    def test_projection_targets_keep_all_conditions(self):
        dataset = make_two_condition_dataset()
        tensor = build_v4_lite_tensor(dataset)
        prediction = predict_v4_lite(tensor, V4LiteParameters.default(tensor.structure))
        targets = _projection_targets_from_prediction(prediction)
        self.assertEqual(set(targets), {"ctrl", "drug"})
        self.assertEqual(targets["drug"].diagnostics["condition"], "drug")
        self.assertFalse(np.shares_memory(targets["ctrl"].state_abundance, targets["drug"].state_abundance))

    def test_full_calibration_runs_each_condition_target(self):
        dataset = make_two_condition_dataset()
        tensor = build_v4_lite_tensor(dataset)
        prediction = predict_v4_lite(tensor, V4LiteParameters.default(tensor.structure))
        targets = _projection_targets_from_prediction(prediction)
        runner = FullCalibrationRunner(
            dataset,
            targets,
            structure=tensor.structure,
            release_table=(
                {"full_block": "state_landscape_transition", "release": False},
                {"full_block": "ecDNA_tail_turnover", "release": False},
                {"full_block": "growth_hazard", "release": False},
            ),
            settings=FullCalibrationSettings(run_formal_raw_refinement=False),
        )

        with mock.patch.object(runner, "_project_bundle", side_effect=lambda _bundle, condition_name: targets[condition_name]):
            results = runner.run_all_conditions()

        self.assertEqual(set(results), {"ctrl", "drug"})
        self.assertEqual(results["drug"].condition_name, "drug")
        self.assertEqual(results["drug"].coarse_residual_report["condition_scope"], "multi_condition_available")

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
        capability = full_model_capability_report(result)
        self.assertFalse(capability["is_final_full_model_fit"])
        self.assertEqual(capability["not_yet_formal"]["full_sbc"], "not_implemented")

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

    def test_fit_method_artifacts_use_stage_specific_prediction(self):
        tensor = build_v4_lite_tensor(make_dataset())
        params = V4LiteParameters.default(tensor.structure)
        final_prediction = predict_v4_lite(tensor, params)
        stage_prediction = predict_v4_lite(tensor, params)
        stage_prediction.state_abundance["ctrl"] = stage_prediction.state_abundance["ctrl"].copy()
        stage_prediction.state_abundance["ctrl"][0, 0] += 123.0
        m1_prediction = predict_v4_lite(tensor, params)
        m1_prediction.copy_distributions["ctrl"] = m1_prediction.copy_distributions["ctrl"].copy()
        m1_prediction.copy_distributions["ctrl"][0, 0, 0, 0] += 456.0
        m0_stage = V4LiteStageFitResult(
            "M0-observation-only",
            ("observation",),
            None,
            2.0,
            1.0,
            (),
            True,
            best_params=params.copy(),
            prediction=stage_prediction,
            posterior_or_map_vector=np.array([1.0, 2.0]),
        )
        m1_stage = V4LiteStageFitResult(
            "M1-ecDNA-kernel",
            ("ecDNA_kernel",),
            None,
            2.0,
            1.0,
            (),
            True,
            best_params=params.copy(),
            prediction=m1_prediction,
            posterior_or_map_vector=np.array([3.0]),
        )
        fake_recovery = V4LiteFakeDataRecoveryReport(0.0, 0.0, True, {})
        reports = build_v4_lite_reports(tensor, final_prediction, [m0_stage, m1_stage], (), fake_recovery, V4LiteLeaveOneWeekOutReport({}), None, {}, {}, ())
        result = V4LiteFitResult(params, tensor, (m0_stage, m1_stage), reports)
        self.assertIn("qpcdr", build_obs_params_for_full(result))
        arrays_by_stem = {}

        def fake_write_arrays(_output_dir, stem, arrays, _label):
            arrays_by_stem[stem] = {name: np.asarray(value).copy() for name, value in arrays.items()}

        with mock.patch.object(Path, "mkdir", return_value=None), mock.patch.object(Path, "write_text", lambda path, text, *args, **kwargs: len(text)), mock.patch("fit.v4_lite.write_standardized_dataset"), mock.patch("fit.v4_lite._write_fit_npz_and_nc_marker", side_effect=fake_write_arrays), mock.patch("fit.v4_lite.write_table_bundle", return_value={}), mock.patch("fit.v4_lite.write_json"), mock.patch("fit.v4_lite.write_text_pdf"):
            write_fit_method_artifacts(Path.cwd() / "unused-fit-method-dir", result, (), {})

        self.assertTrue(np.allclose(arrays_by_stem["M0_observation_only_fit"]["state_abundance"], stage_prediction.state_abundance["ctrl"]))
        self.assertFalse(np.allclose(arrays_by_stem["M0_observation_only_fit"]["state_abundance"], final_prediction.state_abundance["ctrl"]))
        self.assertTrue(np.allclose(arrays_by_stem["M0_observation_only_fit"]["posterior_or_map_vector"], np.array([1.0, 2.0])))
        self.assertTrue(np.allclose(arrays_by_stem["M1_ecDNA_kernel_fit"]["copy_distributions"], m1_prediction.copy_distributions["ctrl"]))
        self.assertFalse(np.allclose(arrays_by_stem["M1_ecDNA_kernel_fit"]["copy_distributions"], final_prediction.copy_distributions["ctrl"]))

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
        self.assertIn("FULL_model_capability_report.json", written)

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
