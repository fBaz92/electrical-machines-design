# Induction Motor Design Package

A comprehensive Python framework for designing three-phase squirrel-cage induction motors, refactored from MATLAB code (2016) based on course notes from Prof. Giovanni Serra, University of Bologna.

## Overview

This package implements an iterative design methodology for induction motors:

1. **Preliminary Sizing** - Uses Esson's output equation to calculate initial dimensions
2. **Lamination Setup** - Defines stator/rotor geometry (slots, bars, air gap)
3. **Winding Design** - Calculates conductors per slot, winding factor
4. **Iterative Refinement** - Converges on efficiency, power factor, and slip
5. **Performance Analysis** - Generates torque-speed curves and circuit parameters

## Installation

```bash
# Clone or copy the package
cp -r induction_motor_design /your/project/path/

# No external dependencies required (pure Python + standard library)
# Optional: matplotlib for plotting convergence
```

## Quick Start

```python
from induction_motor_design import design_motor

# Design a 7.5 kW motor
outputs = design_motor(
    power_kw=7.5,
    voltage=400,          # Line voltage [V]
    frequency=50,         # Hz
    pole_pairs=3,         # 6 poles
    rpm_rated=970,        # Target rated speed
    efficiency=0.883,     # Target efficiency
    power_factor=0.82     # Target power factor
)

# Access results
print(f"Bore diameter: {outputs.lamination.D_bore*1000:.1f} mm")
print(f"Efficiency: {outputs.rated_performance.efficiency:.1%}")
print(f"Circuit Rs={outputs.circuit.Rs:.3f} Ω")
```

## Detailed Usage

### 1. Define Nameplate Data

```python
from induction_motor_design import create_nameplate_from_rpm

nameplate = create_nameplate_from_rpm(
    power_kw=7.5,
    voltage_line=400,
    frequency=50,
    pole_pairs=3,
    rpm_rated=970,
    efficiency=0.883,
    power_factor=0.82
)
print(nameplate)  # Shows all derived quantities
```

### 2. Custom Materials

```python
from induction_motor_design import (
    DesignInputs, InductionMotorDesigner,
    create_M270_35A,  # Premium steel
    COPPER, ALUMINUM
)

inputs = DesignInputs(
    nameplate=nameplate,
    steel=create_M270_35A(),      # Low-loss lamination
    stator_conductor=COPPER,
    rotor_conductor=ALUMINUM,
    Bm_target=0.85,               # Air gap flux density [T]
    fill_factor=0.42,             # Slot fill factor
    short_pitch_slots=1           # Coil shortening
)

designer = InductionMotorDesigner(inputs, verbose=True)
outputs = designer.run()
```

### 3. Custom Lamination Geometry

```python
from induction_motor_design import (
    StatorSlot, StatorLamination, RotorLamination,
    RotorBar, EndRing, AirGap, LaminationAssembly
)

# Define stator slot
stator_slot = StatorSlot(
    h1=16.5e-3,      # Main slot height [m]
    h3=1e-3,         # Chamfer height [m]
    h4=0.7e-3,       # Opening height [m]
    a1=6.5e-3,       # Slot width [m]
    a_opening=2.5e-3 # Slot opening [m]
)

# Define stator lamination
stator = StatorLamination(
    D_bore=0.165,       # Bore diameter [m]
    D_outer=0.240,      # Outer diameter [m]
    L_stack=0.173,      # Stack length [m]
    N_slots=54,
    slot=stator_slot,
    tooth_width=5.8e-3,
    yoke_height=18.4e-3
)

# Similarly define rotor and combine
# ... (see examples/example_7_5kw.py)
```

### 4. Performance Curves

```python
from induction_motor_design import calculate_torque_speed_curve

curve = calculate_torque_speed_curve(
    params=outputs.circuit,
    V_phase=nameplate.voltage_phase,
    frequency=50,
    pole_pairs=3,
    slip_range=(0.001, 1.0),
    n_points=100
)

for point in curve:
    print(f"s={point.slip:.3f}, T={point.torque:.1f} Nm, "
          f"I={point.current_line:.1f} A")
```

## Package Structure

```
induction_motor_design/
├── __init__.py           # Main exports
├── models/
│   ├── nameplate.py      # Motor specifications
│   ├── materials.py      # Steel, copper, aluminum
│   ├── lamination.py     # Stator/rotor geometry
│   └── winding.py        # Winding configuration
├── calculations/
│   ├── preliminary.py    # Esson's equation
│   ├── magnetic.py       # MMF, Xm calculations
│   ├── losses.py         # Iron, copper, mechanical
│   └── equivalent_circuit.py  # Circuit model
├── core/
│   ├── convergence.py    # Iteration tracking
│   └── design_engine.py  # Main orchestrator
├── utils/
│   └── constants.py      # Physical constants
└── examples/
    └── example_7_5kw.py  # Example design
```

## Key Classes

| Class                    | Description                             |
| ------------------------ | --------------------------------------- |
| `NameplateData`        | Motor rated specifications              |
| `ElectricalSteel`      | Lamination material with B-H curve      |
| `StatorLamination`     | Stator geometry                         |
| `RotorLamination`      | Rotor geometry (cage)                   |
| `WindingConfiguration` | Winding layout                          |
| `CircuitParameters`    | Equivalent circuit (Rs, Xs, Rr, Xr, Xm) |
| `PerformancePoint`     | Operating point data                    |
| `DesignOutputs`        | Complete design results                 |

## Design Methodology

The design follows Prof. Serra's methodology:

1. **Output Equation** (Esson):

   ```
   D³ = P × p² × 2√2 / (π³ × Ka × η×cosφ × Bm × Δ × m × f)
   ```
2. **Conductors per Slot**:

   ```
   n = √2 × Vf × kE / (Ka × 2πf × q × Bm × L × D)
   ```
3. **Iterative Loop**:

   - Calculate current from η×cosφ
   - Calculate losses (Cu + Fe + mech)
   - Update slip, efficiency, power factor
   - Solve equivalent circuit
   - Check convergence
4. **Convergence Criteria**:

   - Relative change in η, cosφ, slip < 0.5%
   - Maximum 50 iterations
     ## Known Limitations
