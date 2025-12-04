#!/usr/bin/env python3
"""
Run the genetic optimizer with the thermal model enabled.

The script mirrors the standard GA example but penalizes designs whose
steady-state winding temperature exceeds the selected insulation class.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from induction_motor_design import (
    optimize_motor_genetic,
    InsulationClass,
)

from examples.genetic_optimizer_example import (
    describe_individual,
    summarize_requested_slips,
    generate_slip_curve,
    plot_slip_metrics,
)


def _print_section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


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
    
    _print_section("GENETIC OPTIMIZATION WITH THERMAL MODEL")
    best, optimizer = optimize_motor_genetic(
        **spec,
        population_size=30,
        n_generations=12,
        mutation_rate=0.18,
        crossover_rate=0.7,
        elitism_count=5,
        random_injection_rate=0.15,
        enable_thermal_model=True,
        insulation_class=InsulationClass.F,
        thermal_ambient=40.0,
        cooling_type="TEFC",
        verbose=True,
    )
    
    _print_section("BEST INDIVIDUAL SUMMARY")
    describe_individual(best)
    
    if best.thermal_result:
        result = best.thermal_result
        print(
            f"\nThermal:\n"
            f"  Winding temperature: {result.T_winding:.1f} °C "
            f"(limit {result.insulation_class.max_temp:.0f} °C)\n"
            f"  Frame temperature: {result.T_frame:.1f} °C\n"
            f"  Margin: {result.temperature_margin:.1f} K\n"
            f"  Within limits: {'yes' if result.is_within_limits else 'no'}"
        )
    
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
        print("No GA history recorded.")
    
    _print_section("PERFORMANCE vs SLIP")
    summarize_requested_slips(best, optimizer)
    curve = generate_slip_curve(best, optimizer)
    plot_slip_metrics(curve, optimizer.nameplate.torque_rated)
    
    _print_section("DONE")
    print("Thermally-aware GA optimization complete.")


if __name__ == "__main__":
    main()
