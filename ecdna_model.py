"""
Event-driven ecDNA simulator with exact OU propagation and bounded hazards.
This refactor introduces a strict Model/Bounder/Simulator layering that
implements thinning-based continuous-time dynamics.
"""

from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Core data structures
# =============================================================================


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class CellState:
    """Single cell state at time t."""

    e: int = 0
    m: int = 0
    k: np.ndarray = field(default_factory=lambda: np.array([0], dtype=int))
    y: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    a: float = 0.0

    def copy(self) -> "CellState":
        return CellState(e=self.e, m=self.m, k=self.k.copy(), y=self.y.copy(), a=self.a)

    @property
    def discrete_type(self) -> Tuple[int, int, Tuple[int, ...]]:
        return (self.e, self.m, tuple(int(v) for v in self.k))


def _serialize_state(state: Optional[CellState]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {
        "e": int(state.e),
        "m": int(state.m),
        "k": state.k.astype(int).copy(),
        "y": state.y.astype(float).copy(),
        "a": float(state.a),
    }


@dataclass
class SimulationConfig:
    """Configuration for simulation."""

    t_max: float = 100.0
    seed: Optional[int] = None
    max_cells: Optional[int] = None  # if set, exceeding this raises an error
    record_interval: float = 1.0
    check_bounds: bool = False
    bound_tolerance: float = 1e-9


# =============================================================================
# Model layer: bounded hazards + exact OU propagation
# =============================================================================


@dataclass
class OUParameters:
    """Diagonal OU parameters allowing exact sampling."""

    mean: np.ndarray
    rate: np.ndarray  # B_i > 0 elementwise
    diffusion: np.ndarray  # diagonal std dev

    def propagate(self, y0: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
        """Exact sampling for diagonal OU: dy = -B (y-mean) dt + Sigma dW."""
        decay = np.exp(-self.rate * delta)
        mean = self.mean + decay * (y0 - self.mean)
        var = (self.diffusion ** 2) * (1 - np.exp(-2 * self.rate * delta)) / (2 * self.rate)
        noise = rng.normal(0.0, np.sqrt(var))
        return mean + noise


@dataclass
class ModelParameters:
    """
    Bounded model specification. Hazards are parameterised as r_max * logistic(score),
    ensuring global upper bounds. ecDNA copy numbers are truncated to K_max.
    """

    n_env: int = 1
    n_reg: int = 1
    n_ecdna: int = 1
    ou_params: OUParameters = field(default_factory=lambda: OUParameters(mean=np.zeros(1), rate=np.ones(1), diffusion=np.zeros(1)))
    k_max: np.ndarray = field(default_factory=lambda: np.array([50], dtype=int))
    div_rate_max: float = 0.1
    death_rate_max: float = 0.05
    gain_rate_max: float = 0.01
    loss_rate_max: float = 0.01
    reg_switch_max: float = 0.02
    env_switch_max: float = 0.0

    def _score(self, state: CellState) -> float:
        return float(np.sum(state.k)) - float(np.sum(self.k_max) * 0.25)

    def lambda_div(self, state: CellState) -> float:
        age_effect = 1.0 - math.exp(-0.2 * state.a)
        return self.div_rate_max * _logistic(self._score(state)) * age_effect

    def lambda_death(self, state: CellState) -> float:
        return self.death_rate_max * _logistic(-self._score(state))

    def q_m_rates(self, state: CellState) -> np.ndarray:
        rates = np.zeros(self.n_reg)
        for m_new in range(self.n_reg):
            if m_new == state.m:
                continue
            rates[m_new] = self.reg_switch_max * _logistic(0.5 * (m_new - state.m))
        return rates

    def omega_e_rates(self, state: CellState) -> np.ndarray:
        rates = np.zeros(self.n_env)
        for e_new in range(self.n_env):
            if e_new == state.e:
                continue
            rates[e_new] = self.env_switch_max * _logistic(0.0)
        return rates

    def mu_gain(self, state: CellState, j: int) -> float:
        return self.gain_rate_max * _logistic(self._score(state))

    def mu_loss(self, state: CellState, j: int) -> float:
        return self.loss_rate_max * _logistic(-self._score(state))

    # Division kernel pieces -------------------------------------------------
    def sample_amplification(self, state: CellState, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(0.1 * state.k)

    def post_segregation_loss_prob(self, state: CellState, j: int, k_daughter_j: int) -> float:
        return min(0.5, 0.01 * (k_daughter_j + 1))

    def sample_daughter_m(self, parent: CellState, k_daughter: np.ndarray, rng: np.random.Generator) -> int:
        return rng.integers(0, self.n_reg)

    def sample_daughter_y(self, parent: CellState, m_daughter: int, k_daughter: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0.0, 0.05, size=parent.y.shape)
        return parent.y + noise

    # Exact phenotype propagation -------------------------------------------
    def propagate_y_exact(self, state: CellState, delta: float, rng: np.random.Generator) -> np.ndarray:
        return self.ou_params.propagate(state.y, delta, rng)


# =============================================================================
# Bounder
# =============================================================================


class Bounder:
    """Compute provable upper bounds for hazards."""

    def __init__(self, params: ModelParameters):
        self.params = params

    def bounds(self, state: CellState) -> Tuple[float, Dict[str, Any]]:
        # channel upper bounds using maxima and k_max
        b_div = self.params.div_rate_max
        b_death = self.params.death_rate_max
        b_m = np.full(self.params.n_reg, self.params.reg_switch_max)
        if b_m.size > 0:
            b_m[state.m] = 0.0
        b_e = np.full(self.params.n_env, self.params.env_switch_max)
        if b_e.size > 0:
            b_e[state.e] = 0.0
        b_gain = np.full(self.params.n_ecdna, self.params.gain_rate_max)
        b_loss = self.params.loss_rate_max * self.params.k_max

        total = b_div + b_death + b_gain.sum() + b_loss.sum()
        total += b_m.sum() + b_e.sum()
        return total, {
            "div": b_div,
            "death": b_death,
            "m_switch": b_m,
            "e_switch": b_e,
            "k_gain": b_gain,
            "k_loss": b_loss,
        }


# =============================================================================
# Simulator
# =============================================================================


class EcDNASimulator:
    """Event-driven simulator using thinning."""

    def __init__(self, params: ModelParameters, config: SimulationConfig):
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.bounder = Bounder(params)
        self.event_log: List[Dict[str, Any]] = []

    # --- helpers ------------------------------------------------------------
    def _truncate_k(self, k: np.ndarray) -> np.ndarray:
        return np.minimum(k, self.params.k_max)

    def _sanitize_state(self, state: CellState) -> CellState:
        clean = state.copy()
        clean.k = self._truncate_k(clean.k)
        return clean

    def _compute_rates(self, state: CellState) -> Dict[str, Any]:
        rates = {
            "div": self.params.lambda_div(state),
            "death": self.params.lambda_death(state),
            "m_switch": self.params.q_m_rates(state),
            "e_switch": self.params.omega_e_rates(state),
            "k_gain": np.array([self.params.mu_gain(state, j) for j in range(self.params.n_ecdna)]),
            "k_loss": np.array([self.params.mu_loss(state, j) * state.k[j] for j in range(self.params.n_ecdna)]),
        }
        rates["total"] = (
            rates["div"]
            + rates["death"]
            + rates["m_switch"].sum()
            + rates["e_switch"].sum()
            + rates["k_gain"].sum()
            + rates["k_loss"].sum()
        )
        return rates

    def _choose_event(self, rates: Dict[str, Any]) -> str:
        u = self.rng.random() * rates["total"]
        cumsum = rates["div"]
        if u < cumsum:
            return "div"
        cumsum += rates["death"]
        if u < cumsum:
            return "death"
        for m_new, rate in enumerate(rates["m_switch"]):
            cumsum += rate
            if u < cumsum:
                return f"m_switch_{m_new}"
        for e_new, rate in enumerate(rates["e_switch"]):
            cumsum += rate
            if u < cumsum:
                return f"e_switch_{e_new}"
        for j, rate in enumerate(rates["k_gain"]):
            cumsum += rate
            if u < cumsum:
                return f"k_gain_{j}"
        for j, rate in enumerate(rates["k_loss"]):
            cumsum += rate
            if u < cumsum:
                return f"k_loss_{j}"
        return "none"

    def _divide_cell(self, parent: CellState) -> Tuple[CellState, CellState]:
        amp = self.params.sample_amplification(parent, self.rng)
        k_tilde = 2 * parent.k + amp

        k1 = np.array([self.rng.binomial(int(k_tilde[j]), 0.5) for j in range(len(k_tilde))])
        k2 = k_tilde - k1

        daughters_k = []
        for k_r in [k1, k2]:
            k_star = np.zeros_like(k_r)
            for j in range(len(k_r)):
                loss_prob = self.params.post_segregation_loss_prob(parent, j, int(k_r[j]))
                k_star[j] = self.rng.binomial(int(k_r[j]), 1 - loss_prob)
            daughters_k.append(self._truncate_k(k_star))

        daughters: List[CellState] = []
        for k_d in daughters_k:
            m_d = self.params.sample_daughter_m(parent, k_d, self.rng)
            y_d = self.params.sample_daughter_y(parent, m_d, k_d, self.rng)
            daughters.append(CellState(e=parent.e, m=m_d, k=k_d, y=y_d, a=0.0))
        return daughters[0], daughters[1]

    # --- thinning loop for one cell ----------------------------------------
    def _next_true_event(
        self, state: CellState, t_current: float
    ) -> Tuple[float, Optional[str], Optional[List[CellState]], Dict[str, Any]]:
        state_work = state.copy()
        t_work = t_current
        n_proposed = 0
        n_reject = 0
        max_ratio = 0.0
        last_delta = 0.0
        last_ratio = 0.0
        last_rates: Optional[Dict[str, Any]] = None
        last_bounds: Optional[Dict[str, Any]] = None
        while True:
            bar_r, bounds = self.bounder.bounds(state_work)
            if bar_r <= 0:
                return math.inf, None, None, {
                    "n_proposed": n_proposed,
                    "n_reject": n_reject,
                    "max_ratio": max_ratio,
                    "bar_r": bar_r,
                    "total_rate": 0.0,
                    "m_prev": state_work.m,
                    "e_prev": state_work.e,
                    "accept_prob": last_ratio,
                    "delta_last": last_delta,
                    "rates": last_rates,
                    "bounds": bounds,
                }
            delta = self.rng.exponential(1.0 / bar_r)
            last_delta = delta
            t_candidate = t_work + delta
            state_work.y = self.params.propagate_y_exact(state_work, delta, self.rng)
            state_work.a += delta
            rates = self._compute_rates(state_work)
            last_rates = rates
            last_bounds = bounds
            if rates["total"] <= 0:
                return math.inf, None, None, {
                    "n_proposed": n_proposed,
                    "n_reject": n_reject,
                    "max_ratio": max_ratio,
                    "bar_r": bar_r,
                    "total_rate": rates["total"],
                    "m_prev": state_work.m,
                    "e_prev": state_work.e,
                    "accept_prob": last_ratio,
                    "delta_last": last_delta,
                    "rates": last_rates,
                    "bounds": last_bounds,
                }
            ratio = rates["total"] / bar_r
            last_ratio = ratio
            max_ratio = max(max_ratio, ratio)
            n_proposed += 1
            if self.config.check_bounds and ratio > 1.0 + self.config.bound_tolerance:
                raise RuntimeError(f"Invalid bound: r_total/bar_r={ratio:.6f} at t={t_candidate:.3f}")
            if self.rng.random() < ratio:
                event = self._choose_event(rates)
                daughters: Optional[List[CellState]] = None
                state_pre = state_work.copy()
                state_post: Optional[Any] = None
                if event == "div":
                    d1, d2 = self._divide_cell(state_pre)
                    daughters = [d1, d2]
                    state_post = {"parent": state_pre.copy(), "daughters": [d1.copy(), d2.copy()]}
                elif event == "death":
                    state_post = None
                elif event.startswith("m_switch_"):
                    state_work.m = int(event.split("_")[-1])
                    state_post = state_work.copy()
                elif event.startswith("e_switch_"):
                    state_work.e = int(event.split("_")[-1])
                    state_post = state_work.copy()
                elif event.startswith("k_gain_"):
                    j = int(event.split("_")[-1])
                    state_work.k[j] = min(state_work.k[j] + 1, self.params.k_max[j])
                    state_post = state_work.copy()
                elif event.startswith("k_loss_"):
                    j = int(event.split("_")[-1])
                    state_work.k[j] = max(0, state_work.k[j] - 1)
                    state_post = state_work.copy()
                return t_candidate, event, daughters, {
                    "n_proposed": n_proposed,
                    "n_reject": n_reject,
                    "max_ratio": max_ratio,
                    "bar_r": bar_r,
                    "total_rate": rates["total"],
                    "accept_prob": ratio,
                    "delta_last": delta,
                    "rates": rates,
                    "bounds": bounds,
                    "state_pre": state_pre,
                    "state_post": state_post,
                }
            n_reject += 1
            t_work = t_candidate

    # --- public APIs -------------------------------------------------------
    def simulate_single_cell_lineage(self, initial: CellState, daughter_rule: str = "random") -> List[Dict[str, Any]]:
        t = 0.0
        state = self._sanitize_state(initial)
        traj = [{"t": t, "state": state.copy()}]
        while t < self.config.t_max:
            t_event, event, daughters, diag = self._next_true_event(state, t)
            if t_event == math.inf or t_event > self.config.t_max:
                break
            t = t_event
            if event == "death":
                traj.append({"t": t, "state": state.copy(), "event": "death"})
                break
            if event == "div" and daughters is not None:
                chosen = 0 if daughter_rule == "first" else int(self.rng.integers(0, 2))
                state = daughters[chosen]
                traj.append({"t": t, "state": state.copy(), "event": "division"})
                continue
            state_post = diag.get("state_post")
            if isinstance(state_post, CellState):
                state = state_post
            traj.append({"t": t, "state": state.copy(), "event": event})
        return traj

    def simulate_population(self, initial_cells: List[CellState]) -> Dict[str, Any]:
        cells: Dict[int, CellState] = {i: self._sanitize_state(c) for i, c in enumerate(initial_cells)}
        heap: List[Tuple[float, int]] = []
        scheduled: Dict[int, Dict[str, Any]] = {}
        current_time = 0.0
        next_record = 0.0
        cell_counter = len(cells)
        history = {
            "times": [],
            "population_size": [],
            "mean_k": [],
            "std_k": [],
            "mean_k_species": [],
            "std_k_species": [],
            "mean_age": [],
            "mean_y": [],
            "k_distribution": [],
            "k_matrix": [],
            "m_distribution": [],
            "e_distribution": [],
            "m_values": [],
            "e_values": [],
            "y_values": [],
        }

        def schedule_event(cid: int, state: CellState, t_now: float) -> None:
            t_event, event, daughters, diag = self._next_true_event(state, t_now)
            if t_event == math.inf or event is None:
                return
            scheduled[cid] = {
                "t": t_event,
                "event": event,
                "daughters": daughters,
                "diag": diag,
            }
            heapq.heappush(heap, (t_event, cid))

        for cid, state in cells.items():
            schedule_event(cid, state, current_time)

        def record_snapshot(time_now: float):
            history["times"].append(time_now)
            if not cells:
                history["population_size"].append(0)
                history["mean_k"].append(0.0)
                history["std_k"].append(0.0)
                history["mean_k_species"].append(np.zeros(self.params.n_ecdna))
                history["std_k_species"].append(np.zeros(self.params.n_ecdna))
                history["mean_age"].append(0.0)
                history["mean_y"].append(np.zeros_like(self.params.ou_params.mean))
                history["k_distribution"].append(np.array([], dtype=int))
                history["k_matrix"].append(np.zeros((0, self.params.n_ecdna), dtype=int))
                history["m_distribution"].append(np.zeros(self.params.n_reg))
                history["e_distribution"].append(np.zeros(self.params.n_env))
                history["m_values"].append(np.array([], dtype=int))
                history["e_values"].append(np.array([], dtype=int))
                history["y_values"].append(np.zeros((0, self.params.ou_params.mean.size)))
                return
            history["population_size"].append(len(cells))
            k_matrix = np.array([c.k for c in cells.values()])
            k_vals = np.sum(k_matrix, axis=1)
            history["mean_k"].append(float(np.mean(k_vals)))
            history["std_k"].append(float(np.std(k_vals)))
            history["mean_k_species"].append(np.mean(k_matrix, axis=0))
            history["std_k_species"].append(np.std(k_matrix, axis=0))
            history["mean_age"].append(float(np.mean([c.a for c in cells.values()])))
            y_vals = np.array([c.y for c in cells.values()])
            history["mean_y"].append(np.mean(y_vals, axis=0))
            history["k_distribution"].append(k_vals.copy())
            history["k_matrix"].append(k_matrix.copy())
            m_vals = np.array([c.m for c in cells.values()])
            e_vals = np.array([c.e for c in cells.values()])
            m_counts = np.bincount(m_vals, minlength=self.params.n_reg)
            e_counts = np.bincount(e_vals, minlength=self.params.n_env)
            history["m_distribution"].append(m_counts / len(cells))
            history["e_distribution"].append(e_counts / len(cells))
            history["m_values"].append(m_vals.copy())
            history["e_values"].append(e_vals.copy())
            history["y_values"].append(y_vals.copy())

        record_snapshot(current_time)
        next_record += self.config.record_interval

        while heap:
            t_event, cid = heapq.heappop(heap)
            if t_event > self.config.t_max:
                break
            if cid not in cells:
                continue
            scheduled_event = scheduled.get(cid)
            if scheduled_event is None or scheduled_event["t"] != t_event:
                continue
            current_time = t_event
            event = scheduled_event["event"]
            daughters = scheduled_event["daughters"]
            diag = scheduled_event["diag"]
            scheduled.pop(cid, None)
            current_time = t_event
            if event == "death":
                del cells[cid]
            elif event == "div" and daughters is not None:
                del cells[cid]
                for d in daughters:
                    clean = self._sanitize_state(d)
                    cells[cell_counter] = clean
                    schedule_event(cell_counter, clean, current_time)
                    cell_counter += 1
            else:
                state_post = diag.get("state_post")
                if isinstance(state_post, CellState):
                    cells[cid] = state_post
                    schedule_event(cid, state_post, current_time)
                else:
                    schedule_event(cid, cells[cid], current_time)

            state_pre = _serialize_state(diag.get("state_pre"))
            state_post_raw = diag.get("state_post")
            if isinstance(state_post_raw, dict):
                state_post = {
                    "parent": _serialize_state(state_post_raw.get("parent")),
                    "daughters": [
                        _serialize_state(d) for d in state_post_raw.get("daughters", [])
                    ],
                }
            else:
                state_post = _serialize_state(state_post_raw)
            rates = diag.get("rates") or {}
            bounds = diag.get("bounds") or {}
            self.event_log.append(
                {
                    "t": current_time,
                    "cell_id": cid,
                    "event": event,
                    "state_pre": state_pre,
                    "state_post": state_post,
                    "rates_at_event": {
                        k: (v.copy() if hasattr(v, "copy") else v) for k, v in rates.items()
                    },
                    "bounds_at_event": {
                        k: (v.copy() if hasattr(v, "copy") else v) for k, v in bounds.items()
                    },
                    "bound_total": diag.get("bar_r", 0.0),
                    "n_proposals": diag.get("n_proposed", 0),
                    "n_reject": diag.get("n_reject", 0),
                    "accept_prob": diag.get("accept_prob", 0.0),
                    "delta_last": diag.get("delta_last", 0.0),
                    "bound_ratio": diag.get("max_ratio", 0.0),
                    "total_rate": diag.get("total_rate", 0.0),
                }
            )

            if self.config.max_cells is not None and len(cells) > self.config.max_cells:
                raise RuntimeError(f"Population exceeded max_cells={self.config.max_cells}")

            while next_record <= current_time:
                record_snapshot(next_record)
                next_record += self.config.record_interval

        if history["times"] and history["times"][-1] < self.config.t_max:
            record_snapshot(self.config.t_max)
        return history


# =============================================================================
# Parameter sweep utilities (updated for event-driven simulator)
# =============================================================================


class ParameterSweep:
    @staticmethod
    def sweep_1d(
        param_name: str,
        param_values: np.ndarray,
        base_params: Dict[str, Any],
        param_class: type,
        initial_cells: List[CellState],
        config: SimulationConfig,
        n_replicates: int = 1,
        metric_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if metric_fn is None:
            metric_fn = lambda h: h["population_size"][-1] if h["population_size"] else 0
        results = {"param": param_name, "values": param_values, "metrics": [], "metrics_mean": [], "metrics_std": [], "histories": []}
        for val in param_values:
            params_dict = base_params.copy()
            params_dict[param_name] = val
            rep_metrics = []
            rep_histories = []
            for rep in range(n_replicates):
                params = param_class(**params_dict)
                rep_config = SimulationConfig(
                    t_max=config.t_max,
                    record_interval=config.record_interval,
                    seed=(config.seed + rep) if config.seed is not None else None,
                    max_cells=config.max_cells,
                )
                sim = EcDNASimulator(params, rep_config)
                history = sim.simulate_population([c.copy() for c in initial_cells])
                rep_metrics.append(metric_fn(history))
                rep_histories.append(history)
            results["metrics"].append(rep_metrics)
            results["metrics_mean"].append(float(np.mean(rep_metrics)))
            results["metrics_std"].append(float(np.std(rep_metrics)))
            results["histories"].append(rep_histories)
        return results

    @staticmethod
    def sweep_2d(
        param1_name: str,
        param1_values: np.ndarray,
        param2_name: str,
        param2_values: np.ndarray,
        base_params: Dict[str, Any],
        param_class: type,
        initial_cells: List[CellState],
        config: SimulationConfig,
        n_replicates: int = 1,
        metric_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if metric_fn is None:
            metric_fn = lambda h: h["population_size"][-1] if h["population_size"] else 0
        metrics = np.zeros((len(param1_values), len(param2_values)))
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                params_dict = base_params.copy()
                params_dict[param1_name] = v1
                params_dict[param2_name] = v2
                rep_metrics = []
                for rep in range(n_replicates):
                    params = param_class(**params_dict)
                    rep_config = SimulationConfig(
                        t_max=config.t_max,
                        record_interval=config.record_interval,
                        seed=(config.seed + rep) if config.seed is not None else None,
                        max_cells=config.max_cells,
                    )
                    sim = EcDNASimulator(params, rep_config)
                    history = sim.simulate_population([c.copy() for c in initial_cells])
                    rep_metrics.append(metric_fn(history))
                metrics[i, j] = float(np.mean(rep_metrics))
        return {
            "param1_name": param1_name,
            "param1_values": param1_values,
            "param2_name": param2_name,
            "param2_values": param2_values,
            "metrics": metrics,
        }


# =============================================================================
# Analysis utilities
# =============================================================================


def compute_growth_rate(history: Dict[str, Any], start_frac: float = 0.2) -> float:
    times = np.array(history["times"])
    pop = np.array(history["population_size"])
    start_idx = int(len(times) * start_frac)
    if start_idx >= len(times) - 2:
        return 0.0
    t = times[start_idx:]
    n = pop[start_idx:]
    mask = n > 0
    if mask.sum() < 2:
        return 0.0
    log_n = np.log(n[mask])
    t_masked = t[mask]
    if len(t_masked) < 2:
        return 0.0
    slope = (log_n[-1] - log_n[0]) / (t_masked[-1] - t_masked[0])
    return float(slope)


def compute_extinction_probability(histories: List[Dict[str, Any]]) -> float:
    n_extinct = sum(1 for h in histories if any(n == 0 for n in h.get("population_size", [])))
    return n_extinct / len(histories)


def compute_time_to_threshold(history: Dict[str, Any], threshold: float, key: str = "population_size") -> Optional[float]:
    values = history.get(key, [])
    times = history.get("times", [])
    for t, v in zip(times, values):
        if v >= threshold:
            return float(t)
    return None


def compute_extinction_summary(histories: List[Dict[str, Any]]) -> Dict[str, Any]:
    extinct = []
    t_extinction = []
    last_k = []
    last_m = []
    for h in histories:
        times = np.array(h.get("times", []))
        pop = np.array(h.get("population_size", []))
        idx = np.where(pop == 0)[0]
        if len(idx) > 0:
            ext_idx = int(idx[0])
            extinct.append(True)
            t_extinction.append(float(times[ext_idx]) if len(times) > ext_idx else 0.0)
            alive_idx = np.where(pop > 0)[0]
            if len(alive_idx) > 0:
                last_idx = int(alive_idx[-1])
                last_k.append(h.get("k_distribution", [])[last_idx])
                last_m.append(h.get("m_distribution", [])[last_idx])
            else:
                last_k.append(np.array([], dtype=int))
                last_m.append(np.array([]))
        else:
            extinct.append(False)
            t_extinction.append(None)
            last_k.append(np.array([], dtype=int))
            last_m.append(np.array([]))
    t_max = max((h.get("times") or [0.0])[-1] for h in histories) if histories else 0.0
    return {
        "extinct": extinct,
        "t_extinction": t_extinction,
        "t_max": t_max,
        "last_k": last_k,
        "last_m": last_m,
    }


def compute_survival_curve(summary: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ext_times = [t for t in summary["t_extinction"] if t is not None]
    n_total = len(summary["t_extinction"])
    if n_total == 0:
        return np.array([0.0]), np.array([1.0])
    censor_times = [summary["t_max"]] * (n_total - len(ext_times))
    times = sorted(set(ext_times + censor_times))
    n_at_risk = n_total
    survival = []
    s = 1.0
    for t in times:
        d = sum(1 for et in ext_times if et == t)
        c = sum(1 for ct in censor_times if ct == t)
        if n_at_risk > 0:
            s *= 1.0 - d / n_at_risk
        survival.append(s)
        n_at_risk -= d + c
    if not times or times[0] != 0.0:
        times = [0.0] + times
        survival = [1.0] + survival
    return np.array(times), np.array(survival)


def compute_extinction_probability_ci(histories: List[Dict[str, Any]], alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    n = len(histories)
    if n == 0:
        return 0.0, (0.0, 1.0)
    k = sum(1 for h in histories if any(nv == 0 for nv in h.get("population_size", [])))
    p = k / n
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    return p, (max(0.0, center - half), min(1.0, center + half))


if __name__ == "__main__":
    params = ModelParameters()
    config = SimulationConfig(t_max=5.0, seed=123, record_interval=1.0)
    sim = EcDNASimulator(params, config)
    initial = [CellState(k=np.array([5])) for _ in range(5)]
    history = sim.simulate_population(initial)
    print(f"Final population: {history['population_size'][-1] if history['population_size'] else 0}")
