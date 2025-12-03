#!/usr/bin/env python3
"""
Smoke test for the induction motor design tool based on the README examples.

The script exercises the quick-start workflow, explicit nameplate creation,
custom material selection, and torque-speed curve generation so it is easy to
verify that the key flows in the documentation work end-to-end.
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

# Allow running the script from the repo root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from induction_motor_design import (
    COPPER,
    ALUMINUM,
    DesignInputs,
    InductionMotorDesigner,
    calculate_torque_speed_curve,
    create_M270_35A,
    create_nameplate_from_rpm,
    design_motor,
)


def _print_section(title: str) -> None:
    """Utility to format console sections consistently."""
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def run_quick_start() -> None:
    """Replicate the README quick-start snippet."""
    _print_section("README QUICK START")
    outputs = design_motor(
        power_kw=7.5,
        voltage=400,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82,
    )

    assert outputs.lamination.D_bore > 0, "Bore diameter must be positive"
    assert outputs.rated_performance.efficiency > 0, "Efficiency not computed"

    print(
        f"Bore diameter: {outputs.lamination.D_bore * 1e3:.1f} mm\n"
        f"Stack length: {outputs.lamination.L_stack * 1e3:.1f} mm\n"
        f"Rated efficiency: {outputs.rated_performance.efficiency:.2%}\n"
        f"Power factor: {outputs.rated_performance.power_factor:.3f}\n"
        f"Stator resistance Rs: {outputs.circuit.Rs:.3f} Ω"
    )


def run_custom_materials() -> tuple:
    """Follow the detailed usage example with explicit inputs."""
    _print_section("NAMEPLATE AND CUSTOM MATERIALS")
    nameplate = create_nameplate_from_rpm(
        power_kw=7.5,
        voltage_line=400,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82,
    )

    print("Derived nameplate data:")
    print(
        f"  Phase voltage: {nameplate.voltage_phase:.1f} V\n"
        f"  Rated current: {nameplate.current_line:.1f} A\n"
        f"  Synchronous speed: {nameplate.rpm_sync:.1f} rpm"
    )

    inputs = DesignInputs(
        nameplate=nameplate,
        steel=create_M270_35A(),
        stator_conductor=COPPER,
        rotor_conductor=ALUMINUM,
        Bm_target=0.85,
        fill_factor=0.42,
        short_pitch_slots=1,
    )

    designer = InductionMotorDesigner(inputs, verbose=False)
    outputs = designer.run()

    print(
        f"Final air-gap flux density: {outputs.Bm:.3f} T\n"
        f"Converged? {'yes' if outputs.converged else 'no'} "
        f"after {outputs.iterations} iterations"
    )

    return nameplate, outputs


def run_torque_speed_curve(nameplate, outputs) -> None:
    """Generate the torque-speed curve as in the README."""
    _print_section("TORQUE-SPEED CURVE")
    curve = calculate_torque_speed_curve(
        params=outputs.circuit,
        V_phase=nameplate.voltage_phase,
        frequency=nameplate.frequency,
        pole_pairs=nameplate.pole_pairs,
        slip_range=(0.001, 1.0),
        n_points=100,
    )

    assert curve, "Torque-speed curve is empty"
    print(f"{'Slip':>8} {'Speed [rpm]':>12} {'Torque [Nm]':>12} {'Current [A]':>12}")
    print("-" * 50)
    for point in curve:
        print(
            f"{point.slip:8.3f} {point.speed_rpm:12.0f} {point.torque:12.1f} "
            f"{point.current_line:12.1f}"
        )
    plot_slip_characteristics(curve, outputs.rated_performance.torque)


def plot_slip_characteristics(curve, rated_torque: float) -> None:
    """
    Plot torque, current, torque per unit, and power as a function of slip.
    
    Args:
        curve: Iterable of PerformancePoint results.
        rated_torque: Rated torque to normalize the per-unit panel.
    """
    slips = [point.slip for point in curve]
    torques = [point.torque for point in curve]
    currents = [point.current_line for point in curve]
    torque_pu = [t / rated_torque if rated_torque else 0.0 for t in torques]
    power_kw = [
        point.torque * 2 * math.pi * (point.speed_rpm / 60.0) / 1000.0
        for point in curve
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    
    axes[0, 0].plot(slips, torques, color="tab:blue")
    axes[0, 0].set_ylabel("Coppia [Nm]")
    axes[0, 0].set_title("Coppia vs Slip")
    
    axes[0, 1].plot(slips, currents, color="tab:orange")
    axes[0, 1].set_ylabel("Corrente [A]")
    axes[0, 1].set_title("Corrente vs Slip")
    
    axes[1, 0].plot(slips, torque_pu, color="tab:green")
    axes[1, 0].set_ylabel("Torque [p.u.]")
    axes[1, 0].set_title("Torque Normalizzato vs Slip")
    axes[1, 0].set_xlabel("Slip")
    
    axes[1, 1].plot(slips, power_kw, color="tab:red")
    axes[1, 1].set_ylabel("Potenza [kW]")
    axes[1, 1].set_title("Potenza vs Slip")
    axes[1, 1].set_xlabel("Slip")
    
    for ax in axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    fig.suptitle("Caratteristiche vs Slip", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main() -> None:
    run_quick_start()
    nameplate, outputs = run_custom_materials()
    run_torque_speed_curve(nameplate, outputs)
    _print_section("README TEST COMPLETE")
    print(
        "All README workflows executed successfully. "
        "Review the values above to confirm expected behavior."
    )


if __name__ == "__main__":
    main()
