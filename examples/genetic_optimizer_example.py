#!/usr/bin/env python3
"""
Example script that optimizes an induction motor geometry with the genetic
algorithm included in the project.

It shows how to run ``optimize_motor_genetic`` with a simple 7.5 kW target and
how to access the resulting individual, lamination data, and convergence
history.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt

# Allow running the script directly from the repository root without installing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from induction_motor_design import (
    optimize_motor_genetic,
    solve_circuit,
    calculate_performance
)

# Slip curve configuration (edit these to explore different regions)
SLIP_CURVE_RANGE = (0.005, 1)
SLIP_CURVE_POINTS = 200
REQUESTED_SLIPS = [0.01, 0.03, 0.05, 0.08]


def _print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def describe_individual(individual) -> None:
    assert individual.fitness is not None, "Fitness not evaluated"
    assert individual.lamination is not None, "Lamination not available"
    assert individual.circuit is not None, "Circuit parameters missing"
    
    lam = individual.lamination
    fit = individual.fitness
    
    print(
        f"Bore diameter: {lam.D_bore * 1e3:.1f} mm\n"
        f"Stack length:  {lam.L_stack * 1e3:.1f} mm\n"
        f"Slots / bars:  {lam.stator.N_slots} / {lam.rotor.N_bars}\n"
        f"Total fitness: {fit.total_fitness:.3f}\n"
        f"Max target error: {fit.max_error:.2%}\n"
        f"Valid design: {'yes' if fit.is_valid else 'no'}"
    )


def _solve_performance(
    individual,
    optimizer,
    slips: Sequence[float]
):
    """Return performance points for the requested slips."""
    circuit = individual.circuit
    if circuit is None:
        raise ValueError("Circuit parameters missing; optimizer has not evaluated this individual.")
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


def summarize_requested_slips(individual, optimizer):
    """Print table with power, torque, cosφ, η at requested slips."""
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


def generate_slip_curve(individual, optimizer):
    """Compute fine slip sweep for plotting."""
    min_slip, max_slip = SLIP_CURVE_RANGE
    n_points = max(2, SLIP_CURVE_POINTS)
    slips = [
        min_slip + i * (max_slip - min_slip) / (n_points - 1)
        for i in range(n_points)
    ]
    return _solve_performance(individual, optimizer, slips)


def plot_slip_metrics(perfs, rated_torque: float):
    """Plot power, torque, efficiency, and cosφ vs slip."""
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


def main() -> None:
    _print_section("RUN GENETIC OPTIMIZATION")
    best, optimizer = optimize_motor_genetic(
        power_kw=7.5,
        voltage=400,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82,
        population_size=500,
        n_generations=50,
        mutation_rate=0.4,
        crossover_rate=0.8,
        elitism_count=4,
        verbose=True
    )
    
    _print_section("BEST INDIVIDUAL SUMMARY")
    describe_individual(best)
    
    if optimizer.history:
        _print_section("FITNESS HISTORY (LAST 5 GENERATIONS)")
        for entry in optimizer.history[-5:]:
            print(
                f"Gen {entry['generation']:>3}: "
                f"best={entry['best_fitness']:.3f} "
                f"avg={entry['avg_fitness']:.3f} "
                f"max_error={entry['best_max_error']:.2%}"
            )
    else:
        print("No history recorded (optimizer stopped immediately).")
    
    _print_section("PERFORMANCE vs SLIP")
    summarize_requested_slips(best, optimizer)
    curve = generate_slip_curve(best, optimizer)
    plot_slip_metrics(curve, optimizer.nameplate.torque_rated)
    
    _print_section("DONE")
    print(
        "The genetic optimizer returned a candidate design. "
        "Adjust the GA settings above to explore different trade-offs."
    )


if __name__ == "__main__":
    main()
