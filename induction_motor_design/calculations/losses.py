"""
Loss calculations for induction motors.

Includes:
- Iron losses (hysteresis + eddy current)
- Copper losses (stator)
- Rotor losses (cage)
- Mechanical/windage losses
"""

from dataclasses import dataclass
from typing import Optional
import math

from ..models.materials import ElectricalSteel, ConductorMaterial, COPPER, ALUMINUM
from ..models.lamination import StatorLamination, RotorLamination, RotorBar, EndRing
from ..utils.constants import DENSITY_FE


@dataclass
class IronLosses:
    """Iron loss breakdown."""
    P_teeth_stator: float   # Stator teeth losses [W]
    P_yoke_stator: float    # Stator yoke losses [W]
    P_teeth_rotor: float    # Rotor teeth losses [W] (usually negligible)
    P_yoke_rotor: float     # Rotor yoke losses [W] (usually negligible)
    
    @property
    def total_stator(self) -> float:
        """Total stator iron loss [W]."""
        return self.P_teeth_stator + self.P_yoke_stator
    
    @property
    def total(self) -> float:
        """Total iron loss [W]."""
        return (self.P_teeth_stator + self.P_yoke_stator + 
                self.P_teeth_rotor + self.P_yoke_rotor)


@dataclass 
class CopperLosses:
    """Copper/conductor loss breakdown."""
    P_stator: float    # Stator copper losses [W]
    P_rotor_bars: float    # Rotor bar losses [W]
    P_rotor_rings: float   # End ring losses [W]
    
    @property
    def P_rotor(self) -> float:
        """Total rotor losses [W]."""
        return self.P_rotor_bars + self.P_rotor_rings
    
    @property
    def total(self) -> float:
        """Total copper losses [W]."""
        return self.P_stator + self.P_rotor_bars + self.P_rotor_rings


@dataclass
class TotalLosses:
    """Complete loss breakdown."""
    iron: IronLosses
    copper: CopperLosses
    mechanical: float  # Mechanical/windage losses [W]
    stray: float = 0.0  # Stray load losses [W]
    
    @property
    def total(self) -> float:
        """Total losses [W]."""
        return self.iron.total + self.copper.total + self.mechanical + self.stray
    
    @property
    def P_joule_rotor(self) -> float:
        """Rotor Joule losses (for slip calculation) [W]."""
        return self.copper.P_rotor


def calculate_iron_losses(
    stator: StatorLamination,
    rotor: RotorLamination,
    Bm: float,
    frequency: float,
    steel: ElectricalSteel,
    pole_pairs: int,
    machining_factor: float = 1.5
) -> IronLosses:
    """
    Calculate iron losses in stator and rotor.
    
    Uses the loss model: P = k_i * f * B^α + k_cp * f² * B²
    
    Args:
        stator: Stator lamination
        rotor: Rotor lamination
        Bm: Air gap flux density [T]
        frequency: Supply frequency [Hz]
        steel: Electrical steel properties
        pole_pairs: Number of pole pairs
        machining_factor: Factor for increased losses due to punching (1.3-1.8)
    
    Returns:
        IronLosses breakdown
    """
    # Flux densities
    B_tooth_s = stator.tooth_flux_density(Bm)
    B_yoke_s = stator.yoke_flux_density(Bm, pole_pairs)
    B_tooth_r = rotor.tooth_flux_density(Bm)
    B_yoke_r = rotor.yoke_flux_density(Bm, pole_pairs)
    
    # Masses
    mass_teeth_s = stator.tooth_volume * steel.density * steel.stacking_factor
    mass_yoke_s = stator.yoke_volume * steel.density * steel.stacking_factor
    
    # Rotor volumes (simplified)
    vol_teeth_r = rotor.N_bars * rotor.tooth_width * rotor.slot_height * rotor.L_stack
    vol_yoke_r = math.pi / 4 * ((rotor.D_inner + 2*rotor.yoke_height)**2 - 
                                 rotor.D_inner**2) * rotor.L_stack
    mass_teeth_r = vol_teeth_r * steel.density * steel.stacking_factor
    mass_yoke_r = vol_yoke_r * steel.density * steel.stacking_factor
    
    # Specific losses [W/kg]
    p_teeth_s = steel.specific_loss(B_tooth_s, frequency) * machining_factor
    p_yoke_s = steel.specific_loss(B_yoke_s, frequency) * machining_factor
    
    # Rotor iron losses are at slip frequency (usually negligible)
    # At rated slip ~3%, rotor frequency is ~1.5 Hz
    slip_typical = 0.03
    f_rotor = slip_typical * frequency
    p_teeth_r = steel.specific_loss(B_tooth_r, f_rotor)
    p_yoke_r = steel.specific_loss(B_yoke_r, f_rotor)
    
    return IronLosses(
        P_teeth_stator=p_teeth_s * mass_teeth_s,
        P_yoke_stator=p_yoke_s * mass_yoke_s,
        P_teeth_rotor=p_teeth_r * mass_teeth_r,
        P_yoke_rotor=p_yoke_r * mass_yoke_r
    )


def calculate_stator_copper_loss(
    current_phase: float,
    resistance_phase: float,
    phases: int = 3
) -> float:
    """
    Calculate stator copper losses.
    
    P_cu = m * R_s * I_f²
    
    Args:
        current_phase: Phase current [A]
        resistance_phase: Phase resistance [Ω]
        phases: Number of phases
    
    Returns:
        Stator copper loss [W]
    """
    return phases * resistance_phase * current_phase**2


def calculate_rotor_bar_current(
    current_phase: float,
    power_factor: float,
    winding_factor: float,
    total_conductors: int,
    N_bars: int,
    skew_factor: float,
    phases: int = 3
) -> float:
    """
    Calculate rotor bar current.
    
    I_b = (m₁ * K_a * N * I_f * cos(φ)) / (N_b * K_skew)
    
    Args:
        current_phase: Stator phase current [A]
        power_factor: Power factor cos(φ)
        winding_factor: Stator winding factor K_a
        total_conductors: Total stator conductors per phase N
        N_bars: Number of rotor bars
        skew_factor: Rotor skew factor K_skew
        phases: Number of stator phases
    
    Returns:
        Bar current [A]
    """
    I_bar = (phases * winding_factor * total_conductors * current_phase * power_factor /
             (N_bars * skew_factor))
    return I_bar


def calculate_end_ring_current(
    bar_current: float,
    pole_pairs: int,
    N_bars: int
) -> float:
    """
    Calculate end ring current.
    
    I_a = I_b / (2 * sin(p * π / N_b))
    
    Args:
        bar_current: Bar current [A]
        pole_pairs: Number of pole pairs
        N_bars: Number of rotor bars
    
    Returns:
        End ring current [A]
    """
    return bar_current / (2 * math.sin(pole_pairs * math.pi / N_bars))


def calculate_rotor_losses(
    bar_current: float,
    ring_current: float,
    bar: RotorBar,
    end_ring: EndRing,
    L_stack: float,
    N_bars: int,
    conductor: ConductorMaterial = ALUMINUM
) -> tuple:
    """
    Calculate rotor bar and end ring losses.
    
    Args:
        bar_current: Bar current [A]
        ring_current: End ring current [A]
        bar: Bar geometry
        end_ring: End ring geometry
        L_stack: Stack length [m]
        N_bars: Number of bars
        conductor: Conductor material (usually aluminum)
    
    Returns:
        Tuple of (P_bars, P_rings) [W]
    """
    # Bar resistance and losses
    R_bar = conductor.resistance(L_stack, bar.area)
    P_bars = N_bars * R_bar * bar_current**2
    
    # End ring resistance (both rings)
    # Ring length per bar section
    ring_section_length = math.pi * end_ring.mean_diameter / N_bars
    R_ring_section = conductor.resistance(ring_section_length, end_ring.area)
    
    # Total ring losses (2 rings)
    P_rings = 2 * N_bars * R_ring_section * ring_current**2
    
    return P_bars, P_rings


def calculate_mechanical_losses(
    power_rated: float,
    rpm_rated: float
) -> float:
    """
    Calculate mechanical (windage + friction) losses.
    
    Empirical formula: P_mech = 0.7 * P_n * √(rpm) * 10⁻³
    
    This is a simplified estimate. More accurate calculation
    requires knowledge of bearing type, ventilation system, etc.
    
    Args:
        power_rated: Rated output power [W]
        rpm_rated: Rated speed [rpm]
    
    Returns:
        Mechanical losses [W]
    """
    return 0.7 * power_rated * math.sqrt(rpm_rated) * 1e-3


def calculate_slip_from_losses(
    P_joule_rotor: float,
    P_output: float,
    P_mechanical: float
) -> float:
    """
    Calculate slip from rotor losses.
    
    s = P_jr / (P_out + P_jr + P_mech)
    
    The slip is determined by the rotor losses because:
    P_jr = s * P_air_gap
    
    Args:
        P_joule_rotor: Rotor Joule losses [W]
        P_output: Output mechanical power [W]
        P_mechanical: Mechanical losses [W]
    
    Returns:
        Slip (0-1)
    """
    P_transmitted = P_output + P_joule_rotor + P_mechanical
    return P_joule_rotor / P_transmitted


def calculate_efficiency(
    P_output: float,
    losses: TotalLosses
) -> float:
    """
    Calculate motor efficiency.
    
    η = P_out / (P_out + P_losses)
    
    Args:
        P_output: Output mechanical power [W]
        losses: Total losses breakdown
    
    Returns:
        Efficiency (0-1)
    """
    P_input = P_output + losses.total
    return P_output / P_input


def calculate_power_factor_from_reactive(
    P_active: float,
    Q_reactive: float
) -> float:
    """
    Calculate power factor from active and reactive power.
    
    cos(φ) = P / √(P² + Q²)
    
    Args:
        P_active: Active power [W]
        Q_reactive: Reactive power [VAr]
    
    Returns:
        Power factor (0-1)
    """
    S_apparent = math.sqrt(P_active**2 + Q_reactive**2)
    return P_active / S_apparent


def calculate_reactive_power(
    I_magnetizing: float,
    X_magnetizing: float,
    I_stator: float,
    X_stator: float,
    I_rotor: float,
    X_rotor: float,
    phases: int = 3
) -> float:
    """
    Calculate total reactive power.
    
    Q = Q_m + Q_s + Q_r
      = m * (X_m * I_μ² + X_s * I_s² + X_r * I_r²)
    
    Args:
        I_magnetizing: Magnetizing current [A]
        X_magnetizing: Magnetizing reactance [Ω]
        I_stator: Stator current [A]
        X_stator: Stator leakage reactance [Ω]
        I_rotor: Rotor current (referred to stator) [A]
        X_rotor: Rotor leakage reactance (referred to stator) [Ω]
        phases: Number of phases
    
    Returns:
        Total reactive power [VAr]
    """
    Q = phases * (X_magnetizing * I_magnetizing**2 +
                  X_stator * I_stator**2 +
                  X_rotor * I_rotor**2)
    return Q


@dataclass
class ResistanceFe:
    """Iron loss equivalent resistance for circuit model."""
    R_fe: float  # Equivalent resistance [Ω]
    
    @staticmethod
    def from_losses(voltage_phase: float, P_iron: float) -> 'ResistanceFe':
        """
        Calculate equivalent iron loss resistance.
        
        R_fe = V² / P_fe
        
        Args:
            voltage_phase: Phase voltage [V]
            P_iron: Total iron losses [W]
        
        Returns:
            ResistanceFe object
        """
        R_fe = voltage_phase**2 / P_iron if P_iron > 0 else float('inf')
        return ResistanceFe(R_fe=R_fe)
