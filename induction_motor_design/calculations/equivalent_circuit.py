"""
Equivalent circuit model for induction motors.

Implements the per-phase equivalent circuit and methods to calculate
performance characteristics from circuit parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import math
import cmath
from enum import Enum


class CircuitModel(Enum):
    """Available equivalent circuit models."""
    EXACT = "exact"          # Full T-equivalent circuit
    APPROXIMATE = "approximate"  # Approximate circuit (Xm to supply side)


@dataclass
class CircuitParameters:
    """
    Per-phase equivalent circuit parameters (all referred to stator).
    
    Circuit topology (Exact/T-model):
    
        Rs      Xs           Xr      Rr/s
    ○──┴┴┴──┬┬┬┬──┬──────┬┬┬┬──┴┴┴──○
                  │
                 ╱│╲  
                │ Xm │ // Rfe
                 ╲│╱
                  │
    ○─────────────┴─────────────────○
    
    Attributes:
        Rs: Stator resistance [Ω]
        Xs: Stator leakage reactance [Ω]
        Rr: Rotor resistance (referred to stator) [Ω]
        Xr: Rotor leakage reactance (referred to stator) [Ω]
        Xm: Magnetizing reactance [Ω]
        Rfe: Iron loss equivalent resistance [Ω] (optional)
    """
    Rs: float
    Xs: float
    Rr: float
    Xr: float
    Xm: float
    Rfe: float = float('inf')  # Infinite = no iron losses in circuit
    
    def __post_init__(self):
        """Validate parameters."""
        if any(x < 0 for x in [self.Rs, self.Xs, self.Rr, self.Xr, self.Xm]):
            raise ValueError("Circuit parameters must be non-negative")
    
    @property
    def Zs(self) -> complex:
        """Stator impedance."""
        return complex(self.Rs, self.Xs)
    
    def Zr(self, slip: float) -> complex:
        """Rotor impedance at given slip."""
        if slip == 0:
            return complex(float('inf'), self.Xr)
        return complex(self.Rr / slip, self.Xr)
    
    @property
    def Zm(self) -> complex:
        """Magnetizing branch impedance."""
        # Parallel combination of jXm and Rfe
        jXm = complex(0, self.Xm)
        if self.Rfe == float('inf'):
            return jXm
        return (jXm * self.Rfe) / (jXm + self.Rfe)


@dataclass
class CircuitSolution:
    """
    Solution of the equivalent circuit at a given operating point.
    
    All quantities are per-phase values unless noted.
    """
    slip: float
    
    # Voltages
    V_phase: float          # Applied phase voltage [V]
    E: complex              # Air gap EMF [V]
    
    # Currents  
    I_stator: complex       # Stator current [A]
    I_rotor: complex        # Rotor current (referred) [A]
    I_magnetizing: complex  # Magnetizing current [A]
    I_iron: complex         # Iron loss current [A]
    
    # Impedances at this operating point
    Z_total: complex        # Total impedance seen from supply [Ω]
    
    @property
    def I_stator_mag(self) -> float:
        """Stator current magnitude [A]."""
        return abs(self.I_stator)
    
    @property
    def I_rotor_mag(self) -> float:
        """Rotor current magnitude [A]."""
        return abs(self.I_rotor)
    
    @property
    def power_factor(self) -> float:
        """Power factor cos(φ)."""
        return math.cos(cmath.phase(self.Z_total))
    
    @property
    def phase_angle(self) -> float:
        """Phase angle φ [rad]."""
        return cmath.phase(self.Z_total)


@dataclass
class PerformancePoint:
    """Motor performance at a single operating point."""
    slip: float
    speed_rpm: float
    
    # Electrical
    current_line: float     # Line current [A]
    power_factor: float
    
    # Power
    P_input: float          # Input power [W] (3-phase)
    P_output: float         # Output mechanical power [W]
    P_airgap: float         # Air gap power [W]
    
    # Losses
    P_cu_stator: float      # Stator copper loss [W]
    P_cu_rotor: float       # Rotor copper loss [W]
    P_iron: float           # Iron loss [W]
    
    # Mechanical
    torque: float           # Electromagnetic torque [Nm]
    efficiency: float       # Overall efficiency


def solve_circuit(
    params: CircuitParameters,
    V_phase: float,
    slip: float,
    model: CircuitModel = CircuitModel.EXACT
) -> CircuitSolution:
    """
    Solve the equivalent circuit at a given slip.
    
    Args:
        params: Circuit parameters
        V_phase: Phase voltage [V]
        slip: Operating slip (0-1)
        model: Circuit model to use
    
    Returns:
        CircuitSolution with all electrical quantities
    """
    if slip < 1e-10:
        slip = 1e-10  # Avoid division by zero at synchronous speed
    
    # Impedances
    Zs = params.Zs
    Zr = params.Zr(slip)
    Zm = params.Zm
    
    if model == CircuitModel.EXACT:
        # T-equivalent: Zm in parallel with Zr, then series with Zs
        Z_parallel = (Zm * Zr) / (Zm + Zr)
        Z_total = Zs + Z_parallel
    else:
        # Approximate: Zm at input, ignore voltage drop effect
        Z_total = Zs + Zr + Zm  # Simplified
    
    # Stator current
    I_stator = V_phase / Z_total
    
    # Air gap EMF
    E = V_phase - Zs * I_stator
    
    # Current division between magnetizing and rotor branches
    # Using current divider
    I_magnetizing = E / complex(0, params.Xm)
    
    if params.Rfe < float('inf'):
        I_iron = E / params.Rfe
    else:
        I_iron = 0j
    
    I_rotor = I_stator - I_magnetizing - I_iron
    
    return CircuitSolution(
        slip=slip,
        V_phase=V_phase,
        E=E,
        I_stator=I_stator,
        I_rotor=I_rotor,
        I_magnetizing=I_magnetizing,
        I_iron=I_iron,
        Z_total=Z_total
    )


def calculate_performance(
    params: CircuitParameters,
    solution: CircuitSolution,
    frequency: float,
    pole_pairs: int,
    P_mech_loss: float = 0.0,
    phases: int = 3
) -> PerformancePoint:
    """
    Calculate motor performance from circuit solution.
    
    Args:
        params: Circuit parameters
        solution: Circuit solution at operating point
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        P_mech_loss: Mechanical losses [W]
        phases: Number of phases
    
    Returns:
        PerformancePoint with all performance metrics
    """
    slip = solution.slip
    
    # Synchronous speed
    n_sync = 60 * frequency / pole_pairs
    omega_sync = 2 * math.pi * frequency / pole_pairs
    
    # Operating speed
    n_rpm = n_sync * (1 - slip)
    omega_m = omega_sync * (1 - slip)
    
    # Stator copper loss (3-phase)
    I_s = solution.I_stator_mag
    P_cu_s = phases * params.Rs * I_s**2
    
    # Rotor copper loss
    I_r = solution.I_rotor_mag
    P_cu_r = phases * params.Rr * I_r**2
    
    # Iron loss
    if params.Rfe < float('inf'):
        P_fe = phases * params.Rfe * abs(solution.I_iron)**2
    else:
        P_fe = 0.0
    
    # Air gap power = Rotor input
    P_ag = phases * (params.Rr / slip) * I_r**2
    
    # Mechanical power developed
    P_mech = P_ag * (1 - slip)
    
    # Output power (after mechanical losses)
    P_out = P_mech - P_mech_loss
    P_out = max(0, P_out)  # Can't be negative
    
    # Input power
    P_in = phases * solution.V_phase * I_s * solution.power_factor
    
    # Torque
    if omega_m > 0:
        torque = P_mech / omega_m
    else:
        torque = P_ag / omega_sync  # At standstill
    
    # Efficiency
    if P_in > 0:
        efficiency = P_out / P_in
    else:
        efficiency = 0.0
    
    # Line current (for Y connection)
    I_line = I_s
    
    return PerformancePoint(
        slip=slip,
        speed_rpm=n_rpm,
        current_line=I_line,
        power_factor=solution.power_factor,
        P_input=P_in,
        P_output=P_out,
        P_airgap=P_ag,
        P_cu_stator=P_cu_s,
        P_cu_rotor=P_cu_r,
        P_iron=P_fe,
        torque=torque,
        efficiency=efficiency
    )


def calculate_torque_speed_curve(
    params: CircuitParameters,
    V_phase: float,
    frequency: float,
    pole_pairs: int,
    slip_range: Tuple[float, float] = (0.001, 1.0),
    n_points: int = 100
) -> List[PerformancePoint]:
    """
    Calculate torque-speed characteristic.
    
    Args:
        params: Circuit parameters
        V_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        slip_range: (min_slip, max_slip)
        n_points: Number of calculation points
    
    Returns:
        List of PerformancePoint objects
    """
    slips = [slip_range[0] + i * (slip_range[1] - slip_range[0]) / (n_points - 1)
             for i in range(n_points)]
    
    results = []
    for s in slips:
        solution = solve_circuit(params, V_phase, s)
        perf = calculate_performance(params, solution, frequency, pole_pairs)
        results.append(perf)
    
    return results


def find_rated_slip(
    params: CircuitParameters,
    V_phase: float,
    frequency: float,
    pole_pairs: int,
    P_target: float,
    P_mech_loss: float = 0.0,
    tol: float = 0.001
) -> float:
    """
    Find slip that gives target output power (Newton-Raphson).
    
    Args:
        params: Circuit parameters
        V_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        P_target: Target output power [W]
        P_mech_loss: Mechanical losses [W]
        tol: Convergence tolerance
    
    Returns:
        Slip at rated power
    """
    # Initial guess
    s = 0.05
    
    for _ in range(50):
        sol = solve_circuit(params, V_phase, s)
        perf = calculate_performance(params, sol, frequency, pole_pairs, P_mech_loss)
        
        error = perf.P_output - P_target
        
        if abs(error) < tol * P_target:
            return s
        
        # Numerical derivative
        ds = 0.001
        sol2 = solve_circuit(params, V_phase, s + ds)
        perf2 = calculate_performance(params, sol2, frequency, pole_pairs, P_mech_loss)
        
        dP_ds = (perf2.P_output - perf.P_output) / ds
        
        if abs(dP_ds) < 1e-10:
            break
        
        s = s - error / dP_ds
        s = max(0.001, min(0.5, s))  # Keep slip in reasonable range
    
    return s


def calculate_breakdown_torque(
    params: CircuitParameters,
    V_phase: float,
    frequency: float,
    pole_pairs: int
) -> Tuple[float, float]:
    """
    Calculate breakdown (maximum) torque and corresponding slip.
    
    For the simplified circuit (ignoring Rs):
    s_max = Rr / sqrt(Xs + Xr)²
    T_max ∝ 1 / (2 * (Xs + Xr))
    
    Args:
        params: Circuit parameters
        V_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
    
    Returns:
        Tuple of (breakdown_torque [Nm], slip_at_breakdown)
    """
    # Thevenin equivalent for more accurate calculation
    # Approximate slip at max torque
    X_total = params.Xs + params.Xr
    s_max = params.Rr / math.sqrt(params.Rs**2 + X_total**2)
    
    # Calculate torque at this slip
    sol = solve_circuit(params, V_phase, s_max)
    perf = calculate_performance(params, sol, frequency, pole_pairs)
    
    return perf.torque, s_max


def calculate_starting_performance(
    params: CircuitParameters,
    V_phase: float,
    frequency: float,
    pole_pairs: int
) -> PerformancePoint:
    """
    Calculate starting (locked rotor) performance.
    
    Args:
        params: Circuit parameters
        V_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
    
    Returns:
        PerformancePoint at standstill (slip = 1)
    """
    sol = solve_circuit(params, V_phase, slip=1.0)
    return calculate_performance(params, sol, frequency, pole_pairs)


def refer_rotor_to_stator(
    R_rotor_actual: float,
    X_rotor_actual: float,
    winding_factor_stator: float,
    conductors_per_phase_stator: int,
    N_bars: int,
    skew_factor: float,
    phases: int = 3
) -> Tuple[float, float]:
    """
    Refer rotor parameters to stator.
    
    The transformation ratio is:
    k² = (m₁/N_b) * (K_a * N / K_skew)²
    
    Args:
        R_rotor_actual: Actual rotor resistance [Ω]
        X_rotor_actual: Actual rotor reactance [Ω]
        winding_factor_stator: Stator winding factor
        conductors_per_phase_stator: Stator conductors per phase
        N_bars: Number of rotor bars
        skew_factor: Rotor skew factor
        phases: Number of stator phases
    
    Returns:
        Tuple of (R_rotor_referred, X_rotor_referred) [Ω]
    """
    # Transformation ratio squared
    k_sq = (phases / N_bars) * (winding_factor_stator * conductors_per_phase_stator / 
                                 skew_factor)**2
    
    R_referred = R_rotor_actual * k_sq
    X_referred = X_rotor_actual * k_sq
    
    return R_referred, X_referred
