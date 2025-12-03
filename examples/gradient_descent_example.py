#!/usr/bin/env python3
"""
Example script that refines an induction motor design using gradient descent.

The workflow runs a short genetic optimization to get a feasible starting point
and then applies the gradient-based optimizer to minimize the same cost
function, demonstrating how both solvers can complement each other.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt

# Allow running directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from induction_motor_design import (
    optimize_motor_genetic,
    optimize_motor_gradient_descent,
    hybrid_optimize_motor,
    solve_circuit,
    calculate_performance,
)

SLIP_CURVE_RANGE = (0.001, 1.0)
SLIP_CURVE_POINTS = 120
REQUESTED_SLIPS = [0.01, 0.03, 0.05, 0.08]


def _solve_performance(individual, optimizer, slips: Sequence[float]):
    circuit = individual.circuit
    if circuit is None:
        raise ValueError("Circuit parameters missing for performance evaluation.")
    nameplate = optimizer.nameplate
    P_mech_loss = (individual.final_state or {}).get('P_mech', 0.0)
    
    points = []
    for slip in slips:
        slip = max(1e-4, float(slip))
        sol = solve_circuit(circuit, nameplate.voltage_phase, slip)
        perf = calculate_performance(
            circuit,
            sol,
            nameplate.frequency,
            nameplate.pole_pairs,
            P_mech_loss
        )
        points.append(perf)
    return points


def summarize_slips(individual, optimizer):
    slips = sorted(set(REQUESTED_SLIPS + [optimizer.nameplate.slip_rated]))
    perfs = _solve_performance(individual, optimizer, slips)
    print("\nSlip summary (requested operating points)")
    print(f"{'Slip':>6} {'Speed [rpm]':>12} {'Pout [kW]':>12} {'Torque [Nm]':>14} {'cosφ':>8} {'η':>8}")
    print("-" * 68)
    for perf in perfs:
        print(
            f"{perf.slip:>6.3f} "
            f"{perf.speed_rpm:>12.0f} "
            f"{perf.P_output / 1000:>12.2f} "
            f"{perf.torque:>14.1f} "
            f"{perf.power_factor:>8.3f} "
            f"{perf.efficiency:>8.3f}"
        )


def slip_curve(individual, optimizer):
    min_slip, max_slip = SLIP_CURVE_RANGE
    slips = [
        min_slip + i * (max_slip - min_slip) / (SLIP_CURVE_POINTS - 1)
        for i in range(SLIP_CURVE_POINTS)
    ]
    return _solve_performance(individual, optimizer, slips)


def plot_slip_metrics(perfs, rated_torque: float):
    slips = [p.slip for p in perfs]
    powers = [p.P_output / 1000 for p in perfs]
    torques = [p.torque for p in perfs]
    efficiencies = [p.efficiency * 100 for p in perfs]
    cosphi = [p.power_factor for p in perfs]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    
    axes[0, 0].plot(slips, powers, color="tab:blue")
    axes[0, 0].set_ylabel("Potenza [kW]")
    axes[0, 0].set_title("Potenza vs Slip")
    
    axes[0, 1].plot(slips, torques, color="tab:orange")
    axes[0, 1].axhline(rated_torque, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("Coppia [Nm]")
    axes[0, 1].set_title("Coppia vs Slip")
    
    axes[1, 0].plot(slips, efficiencies, color="tab:green")
    axes[1, 0].set_ylabel("Efficienza [%]")
    axes[1, 0].set_xlabel("Slip")
    axes[1, 0].set_title("Efficienza vs Slip")
    
    axes[1, 1].plot(slips, cosphi, color="tab:red")
    axes[1, 1].set_ylabel("cosφ [-]")
    axes[1, 1].set_xlabel("Slip")
    axes[1, 1].set_title("Fattore di Potenza vs Slip")
    
    for ax in axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    fig.suptitle("Caratteristiche elettriche in funzione dello scorrimento", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def describe_individual(individual) -> None:
    assert individual.fitness is not None, "Fitness not evaluated"
    assert individual.lamination is not None, "Missing lamination data"
    
    lam = individual.lamination
    fit = individual.fitness
    
    print(
        f"Bore diameter: {lam.D_bore * 1e3:.1f} mm\n"
        f"Stack length:  {lam.L_stack * 1e3:.1f} mm\n"
        f"Slots / bars:  {lam.stator.N_slots} / {lam.rotor.N_bars}\n"
        f"Total fitness: {fit.total_fitness:.4f}\n"
        f"Power error:   {fit.power_error:.2%}\n"
        f"Slip error:    {fit.slip_error:.2%}\n"
        f"Valid design:  {'yes' if fit.is_valid else 'no'}"
    )


def main() -> None:
    spec = dict(
        power_kw=7.5,
        voltage=400,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82,
    )
    
    _print_section("HYBRID GA + GRADIENT DESCENT")
    best, info = hybrid_optimize_motor(
        **spec,
        top_candidates=10,
        ga_kwargs=dict(
            population_size=100,
            n_generations=50,
            mutation_rate=0.3,
            crossover_rate=0.7,
            elitism_count=4,
            random_injection_rate=0.15,
            verbose=False,
        ),
        gd_kwargs=dict(
            step_size=0.03,
            gradient_epsilon=5e-4,
            max_iterations=100,
            tolerance=1e-6,
            verbose=False,
        ),
        verbose=True,
    )
    describe_individual(best)
    
    ga_optimizer = info['ga_optimizer']
    if ga_optimizer.history:
        _print_section("GA FITNESS HISTORY (LAST 5 GENERATIONS)")
        for entry in ga_optimizer.history[-5:]:
            print(
                f"Gen {entry['generation']:>3}: "
                f"best={entry['best_fitness']:.3f} "
                f"avg={entry['avg_fitness']:.3f} "
                f"max_error={entry['best_max_error']:.2%}"
            )
    else:
        print("No GA history recorded.")
    
    best_optimizer = info.get('best_optimizer', ga_optimizer)
    summarize_slips(best, best_optimizer)
    curve = slip_curve(best, best_optimizer)
    plot_slip_metrics(curve, best_optimizer.nameplate.torque_rated)
    
    gd_results = info['gd_results']
    if gd_results:
        _print_section("BEST GRADIENT DESCENT RUN (LAST 5 ITERATIONS)")
        best_gd = min(
            gd_results,
            key=lambda item: item[0].fitness.total_fitness
            if item[0].fitness else float('inf')
        )
        history = best_gd[1].history
        for entry in history[-5:]:
            print(
                f"Iter {entry['iteration']:02d}: "
                f"fitness={entry['fitness']:.4f} "
                f"grad_norm={entry['grad_norm']:.3e} "
                f"step={entry['step_size']:.3f}"
            )
    else:
        print("No gradient-descent runs recorded.")
    
    _print_section("DONE")
    print(
        "Hybrid optimization complete. Adjust GA/GD settings in the script "
        "to explore different trade-offs."
    )


if __name__ == "__main__":
    main()
