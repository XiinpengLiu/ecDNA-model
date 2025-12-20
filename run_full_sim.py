"""
Run the full multi-ecDNA model with parameters defined inline.
"""

import numpy as np

from ecdna_model import CellState, EcDNASimulator, ModelParameters, OUParameters, SimulationConfig


def main() -> None:
    n_env = 1
    n_reg = 1
    n_ecdna = 2

    ou_params = OUParameters(
        mean=np.zeros(1),
        rate=np.ones(1),
        diffusion=np.zeros(1),
    )

    params = ModelParameters(
        n_env=n_env,
        n_reg=n_reg,
        n_ecdna=n_ecdna,
        ou_params=ou_params,
        k_max=np.array([50, 50], dtype=int),
        div_rate_max=0.18,
        death_rate_max=0.03,
        gain_rate_max=0.002,
        loss_rate_max=0.002,
        reg_switch_max=0.02,
        env_switch_max=0.0,
        fitness_k_star=4.0,
        fitness_alpha=1.0,
        fitness_beta=0.02,
        fitness_weights=None,
        age_effect_rate=0.2,
        reg_switch_slope=0.5,
        env_switch_bias=0.0,
        amplification_scale=0.05,
        division_copy_factor=2,
        segregation_prob=0.5,
        post_segregation_loss_cap=0.5,
        post_segregation_loss_slope=0.01,
        post_segregation_loss_offset=1.0,
        daughter_y_noise_std=0.05,
    )

    config = SimulationConfig(
        t_max=100.0,
        seed=42,
        record_interval=5.0,
        max_cells=100000,
    )

    initial_k = np.array([2, 2], dtype=int)
    initial_count = 100

    sim = EcDNASimulator(params, config)
    initial_cells = [CellState(k=initial_k.copy()) for _ in range(initial_count)]
    history = sim.simulate_population(initial_cells)

    final_pop = history["population_size"][-1] if history["population_size"] else 0
    mean_k_species = history["mean_k_species"][-1] if history["mean_k_species"] else np.zeros(n_ecdna)

    print(f"Final population: {final_pop}")
    print(f"Final mean k per species: {mean_k_species}")


if __name__ == "__main__":
    main()
