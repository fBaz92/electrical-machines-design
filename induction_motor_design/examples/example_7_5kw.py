#!/usr/bin/env python3
"""
Example: Design of a 7.5 kW induction motor.

This example reproduces the design from the original MATLAB code,
but with a clean, modular approach.
"""

import sys
sys.path.insert(0, '/home/claude')

from induction_motor_design import (
    design_motor,
    DesignInputs,
    InductionMotorDesigner,
    create_nameplate_from_rpm,
    create_M400_50A,
    COPPER, ALUMINUM,
    calculate_torque_speed_curve
)


def main():
    """Run the example design."""
    
    print("=" * 70)
    print("INDUCTION MOTOR DESIGN - 7.5 kW Example")
    print("Reproducing design from original MATLAB code")
    print("=" * 70)
    
    # Method 1: Quick design using convenience function
    print("\n" + "="*70)
    print("METHOD 1: Quick Design")
    print("="*70)
    
    outputs = design_motor(
        power_kw=7.5,
        voltage=350,          # Line voltage [V]
        frequency=50,         # Hz
        pole_pairs=3,         # 6 poles
        rpm_rated=970,        # Rated speed
        efficiency=0.883,     # 88.3%
        power_factor=0.82     # cos(φ) = 0.82
    )
    
    # Method 2: More control with explicit inputs
    print("\n" + "="*70)
    print("METHOD 2: With Custom Materials")
    print("="*70)
    
    # Create nameplate
    nameplate = create_nameplate_from_rpm(
        power_kw=7.5,
        voltage_line=350,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82
    )
    
    print("\nNameplate Data:")
    print(nameplate)
    
    # Create design inputs with specific materials
    inputs = DesignInputs(
        nameplate=nameplate,
        steel=create_M400_50A(),  # M400-50A lamination steel
        stator_conductor=COPPER,
        rotor_conductor=ALUMINUM,
        Bm_target=0.9,           # Target air gap flux density
        fill_factor=0.4,         # Slot fill factor
        short_pitch_slots=1      # Coil shortening
    )
    
    # Run design
    designer = InductionMotorDesigner(inputs, verbose=True)
    outputs = designer.run()
    
    # Calculate torque-speed curve
    print("\n" + "="*70)
    print("TORQUE-SPEED CHARACTERISTIC")
    print("="*70)
    
    curve = calculate_torque_speed_curve(
        params=outputs.circuit,
        V_phase=nameplate.voltage_phase,
        frequency=nameplate.frequency,
        pole_pairs=nameplate.pole_pairs,
        slip_range=(0.001, 1.0),
        n_points=20
    )
    
    print(f"\n{'Slip':>8} {'Speed [rpm]':>12} {'Torque [Nm]':>12} {'Current [A]':>12} {'PF':>8}")
    print("-" * 56)
    for point in curve[::2]:  # Every other point
        print(f"{point.slip:8.3f} {point.speed_rpm:12.0f} {point.torque:12.1f} "
              f"{point.current_line:12.1f} {point.power_factor:8.3f}")
    
    # Show convergence history
    print("\n" + "="*70)
    print("CONVERGENCE HISTORY")
    print("="*70)
    
    history = outputs.history
    print(f"\n{'Iter':>4} {'η':>8} {'cosφ':>8} {'slip':>8} {'Bm [T]':>8}")
    print("-" * 40)
    for it in history.iterations[-10:]:  # Last 10 iterations
        print(f"{it.iteration:4d} {it.efficiency:8.4f} {it.power_factor:8.4f} "
              f"{it.slip:8.4f} {it.Bm:8.3f}")
    
    return outputs


def design_different_motors():
    """Show how to design motors with different specifications."""
    
    print("\n" + "="*70)
    print("DESIGNING DIFFERENT MOTORS")
    print("="*70)
    
    # Small motor: 1.5 kW, 4 poles
    print("\n--- Small Motor (1.5 kW, 4 poles) ---")
    small = design_motor(
        power_kw=1.5,
        voltage=400,
        frequency=50,
        pole_pairs=2,
        rpm_rated=1430,
        efficiency=0.82,
        power_factor=0.78
    )
    
    # Large motor: 45 kW, 4 poles
    print("\n--- Large Motor (45 kW, 4 poles) ---")
    large = design_motor(
        power_kw=45,
        voltage=400,
        frequency=50,
        pole_pairs=2,
        rpm_rated=1475,
        efficiency=0.93,
        power_factor=0.87
    )
    
    # 60 Hz motor (American standard)
    print("\n--- 60 Hz Motor (10 HP, 4 poles) ---")
    hz60 = design_motor(
        power_kw=7.46,  # 10 HP
        voltage=460,
        frequency=60,
        pole_pairs=2,
        rpm_rated=1750,
        efficiency=0.89,
        power_factor=0.84
    )
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\n{'Motor':>12} {'D [mm]':>10} {'L [mm]':>10} {'η':>8} {'cosφ':>8}")
    print("-" * 50)
    
    for name, out in [("1.5 kW", small), ("45 kW", large), ("60 Hz", hz60)]:
        print(f"{name:>12} {out.lamination.D_bore*1000:10.1f} "
              f"{out.lamination.L_stack*1000:10.1f} "
              f"{out.rated_performance.efficiency:8.3f} "
              f"{out.rated_performance.power_factor:8.3f}")


if __name__ == "__main__":
    outputs = main()
    
    # Optionally run comparison
    # design_different_motors()
