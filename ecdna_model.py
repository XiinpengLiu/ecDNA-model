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


@dataclass
class SimulationConfig:
    """Configuration for simulation."""

    t_max: float = 100.0
    seed: Optional[int] = None
    max_cells: Optional[int] = None  # if set, exceeding this raises an error
    record_interval: float = 1.0


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
        b_m = np.zeros(self.params.n_reg)
        b_e = np.zeros(self.params.n_env)
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
    def _next_true_event(self, state: CellState, t_current: float) -> Tuple[float, Optional[str], Optional[List[CellState]]]:
        while True:
            bar_r, _ = self.bounder.bounds(state)
            if bar_r <= 0:
                return math.inf, None, None
            delta = self.rng.exponential(1.0 / bar_r)
            t_candidate = t_current + delta
            state.y = self.params.propagate_y_exact(state, delta, self.rng)
            state.a += delta
            rates = self._compute_rates(state)
            if rates["total"] <= 0:
                return math.inf, None, None
            if self.rng.random() < rates["total"] / bar_r:
                event = self._choose_event(rates)
                daughters = None
                if event == "div":
                    d1, d2 = self._divide_cell(state)
                    daughters = [d1, d2]
                elif event == "death":
                    pass
                elif event.startswith("m_switch_"):
                    state.m = int(event.split("_")[-1])
                elif event.startswith("e_switch_"):
                    state.e = int(event.split("_")[-1])
                elif event.startswith("k_gain_"):
                    j = int(event.split("_")[-1])
                    state.k[j] = min(state.k[j] + 1, self.params.k_max[j])
                elif event.startswith("k_loss_"):
                    j = int(event.split("_")[-1])
                    state.k[j] = max(0, state.k[j] - 1)
                return t_candidate, event, daughters
            t_current = t_candidate

    # --- public APIs -------------------------------------------------------
    def simulate_single_cell_lineage(self, initial: CellState, daughter_rule: str = "random") -> List[Dict[str, Any]]:
        t = 0.0
        state = initial.copy()
        traj = [{"t": t, "state": state.copy()}]
        while t < self.config.t_max:
            t_event, event, daughters = self._next_true_event(state, t)
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
            traj.append({"t": t, "state": state.copy(), "event": event})
        return traj

    def simulate_population(self, initial_cells: List[CellState]) -> Dict[str, Any]:
        cells: Dict[int, CellState] = {i: c.copy() for i, c in enumerate(initial_cells)}
        heap: List[Tuple[float, int]] = []
        current_time = 0.0
        next_record = 0.0
        cell_counter = len(cells)
        history = {
            "times": [],
            "population_size": [],
            "mean_k": [],
            "std_k": [],
            "mean_age": [],
            "mean_y": [],
            "k_distribution": [],
            "m_distribution": [],
        }

        for cid in cells:
            heapq.heappush(heap, (0.0, cid))

        def record_snapshot(time_now: float):
            if not cells:
                return
            history["times"].append(time_now)
            history["population_size"].append(len(cells))
            k_vals = np.array([np.sum(c.k) for c in cells.values()])
            history["mean_k"].append(float(np.mean(k_vals)))
            history["std_k"].append(float(np.std(k_vals)))
            history["mean_age"].append(float(np.mean([c.a for c in cells.values()])))
            y_vals = np.array([c.y for c in cells.values()])
            history["mean_y"].append(np.mean(y_vals, axis=0))
            history["k_distribution"].append(k_vals.copy())
            m_counts = np.bincount([c.m for c in cells.values()], minlength=self.params.n_reg)
            history["m_distribution"].append(m_counts / len(cells))

        record_snapshot(current_time)
        next_record += self.config.record_interval

        while heap:
            t_candidate, cid = heapq.heappop(heap)
            if t_candidate > self.config.t_max:
                break
            if cid not in cells:
                continue
            current_time = t_candidate
            state = cells[cid]
            t_event, event, daughters = self._next_true_event(state, current_time)
            if t_event == math.inf or t_event > self.config.t_max:
                continue
            current_time = t_event
            if event == "death":
                del cells[cid]
            elif event == "div" and daughters is not None:
                del cells[cid]
                for d in daughters:
                    cells[cell_counter] = d
                    heapq.heappush(heap, (current_time, cell_counter))
                    cell_counter += 1
            else:
                cells[cid] = state
                heapq.heappush(heap, (current_time, cid))

            self.event_log.append({"t": current_time, "cell_id": cid, "event": event, "state": state.copy()})

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
    n_extinct = sum(1 for h in histories if h["population_size"] and h["population_size"][-1] == 0)
    return n_extinct / len(histories)


if __name__ == "__main__":
    params = ModelParameters()
    config = SimulationConfig(t_max=5.0, seed=123, record_interval=1.0)
    sim = EcDNASimulator(params, config)
    initial = [CellState(k=np.array([5])) for _ in range(5)]
    history = sim.simulate_population(initial)
    print(f"Final population: {history['population_size'][-1] if history['population_size'] else 0}")
