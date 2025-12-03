"""
Preliminary sizing calculations using Esson's output equation.

This module implements the initial sizing of an induction motor based on
the output equation that relates power to machine dimensions and 
electromagnetic loadings.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math

from ..models.nameplate import NameplateData
from ..utils.constants import DesignRanges


@dataclass
class PreliminarySizing:
    """
    Results of preliminary sizing calculations.
    
    Attributes:
        D_bore: Bore diameter [m]
        L_stack: Stack length [m]
        pole_pitch: Pole pitch τ [m]
        aspect_ratio: L/τ ratio
        Bm_assumed: Assumed air gap flux density [T]
        delta_assumed: Assumed linear current density [A/m]
    """
    D_bore: float
    L_stack: float
    pole_pitch: float
    aspect_ratio: float
    Bm_assumed: float
    delta_assumed: float
    
    @property
    def D_outer_estimate(self) -> float:
        """
        Estimate outer diameter.
        Typically D_outer ≈ 1.4-1.6 * D_bore for small motors.
        """
        return 1.5 * self.D_bore
    
    @property
    def volume_rotor(self) -> float:
        """Rotor volume D²L [m³]."""
        return self.D_bore ** 2 * self.L_stack


def esson_output_equation(
    power: float,
    frequency: float,
    pole_pairs: int,
    Ka: float,
    eta_cosfi: float,
    Bm: float,
    delta: float,
    m: float
) -> float:
    """
    Calculate D³ from Esson's output equation.
    
    The output equation relates mechanical power to machine dimensions:
    
    P = (π²/√2) * Ka * η*cosφ * Bm * Δ * m * f * D³ / p²
    
    Rearranging:
    D³ = P * p² * √2 / (π² * Ka * η*cosφ * Bm * Δ * m * f)
    
    Note: The factor 2√2/π³ comes from the relationship between 
    apparent power and the output coefficient C.
    
    Args:
        power: Output power [W]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        Ka: Winding factor (initial guess ~0.92-0.96)
        eta_cosfi: Product of efficiency and power factor
        Bm: Air gap flux density [T]
        delta: Linear current density [A/m]
        m: Aspect ratio L/τ
    
    Returns:
        D³ value [m³]
    """
    numerator = power * (pole_pairs ** 2) * 2 * math.sqrt(2)
    denominator = (math.pi ** 3) * Ka * eta_cosfi * Bm * delta * m * frequency
    
    return numerator / denominator


def calculate_preliminary_dimensions(
    nameplate: NameplateData,
    Ka: float = 0.92,
    Bm: float = None,
    delta: float = None,
    m: float = None
) -> PreliminarySizing:
    """
    Calculate preliminary motor dimensions from nameplate data.
    
    This is the first step in the design process, giving initial
    values for bore diameter and stack length.
    
    Args:
        nameplate: Motor nameplate specifications
        Ka: Assumed winding factor (typically 0.92-0.96)
        Bm: Air gap flux density [T] (None = use typical value)
        delta: Linear current density [A/m] (None = use typical value)
        m: Aspect ratio L/τ (None = use empirical formula)
    
    Returns:
        PreliminarySizing with calculated dimensions
    """
    # Use default values if not provided
    if Bm is None:
        Bm = DesignRanges.BM_TYPICAL
    
    if delta is None:
        delta = DesignRanges.DELTA_TYPICAL
    
    if m is None:
        m = DesignRanges.aspect_ratio_typical(nameplate.pole_pairs)
    
    # Calculate D³ from output equation
    D_cubed = esson_output_equation(
        power=nameplate.power_w,
        frequency=nameplate.frequency,
        pole_pairs=nameplate.pole_pairs,
        Ka=Ka,
        eta_cosfi=nameplate.eta_cosfi,
        Bm=Bm,
        delta=delta,
        m=m
    )
    
    # Bore diameter
    D_bore = D_cubed ** (1/3)
    
    # Pole pitch
    tau = math.pi * D_bore / (2 * nameplate.pole_pairs)
    
    # Stack length from aspect ratio
    L_stack = m * tau
    
    return PreliminarySizing(
        D_bore=D_bore,
        L_stack=L_stack,
        pole_pitch=tau,
        aspect_ratio=m,
        Bm_assumed=Bm,
        delta_assumed=delta
    )


def scale_to_lamination(
    preliminary: PreliminarySizing,
    D_lamination: float,
    preserve_volume: bool = True
) -> Tuple[float, float]:
    """
    Scale preliminary dimensions to match an available lamination.
    
    When selecting from a catalog of standard laminations, the bore
    diameter is fixed. The stack length must be adjusted to maintain
    the electromagnetic loading.
    
    Args:
        preliminary: Preliminary sizing results
        D_lamination: Available lamination bore diameter [m]
        preserve_volume: If True, adjust L to keep D²L constant
                        If False, adjust L to keep D³ constant
    
    Returns:
        Tuple of (D_new, L_new) [m]
    """
    D_old = preliminary.D_bore
    L_old = preliminary.L_stack
    
    if preserve_volume:
        # Keep D²L constant (constant rotor volume)
        L_new = L_old * (D_old / D_lamination) ** 2
    else:
        # Keep D³ constant (more conservative)
        L_new = L_old * (D_old / D_lamination) ** 3
    
    return D_lamination, L_new


def verify_sizing_limits(sizing: PreliminarySizing) -> dict:
    """
    Verify that preliminary sizing is within acceptable limits.
    
    Args:
        sizing: Preliminary sizing to verify
    
    Returns:
        Dictionary with verification results
    """
    issues = []
    warnings = []
    
    # Check flux density
    if sizing.Bm_assumed < DesignRanges.BM_MIN:
        warnings.append(f"Bm={sizing.Bm_assumed:.2f}T is below typical minimum {DesignRanges.BM_MIN}T")
    elif sizing.Bm_assumed > DesignRanges.BM_MAX:
        warnings.append(f"Bm={sizing.Bm_assumed:.2f}T exceeds typical maximum {DesignRanges.BM_MAX}T")
    
    # Check linear current density
    if sizing.delta_assumed < DesignRanges.DELTA_MIN:
        warnings.append(f"Δ={sizing.delta_assumed/1000:.1f}kA/m is below typical minimum")
    elif sizing.delta_assumed > DesignRanges.DELTA_MAX:
        warnings.append(f"Δ={sizing.delta_assumed/1000:.1f}kA/m exceeds typical maximum")
    
    # Check aspect ratio (should be reasonable)
    if sizing.aspect_ratio < 0.5:
        warnings.append(f"Aspect ratio m={sizing.aspect_ratio:.2f} is very low (pancake shape)")
    elif sizing.aspect_ratio > 3.0:
        warnings.append(f"Aspect ratio m={sizing.aspect_ratio:.2f} is very high (elongated)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


def calculate_output_coefficient(
    power: float,
    rpm_sync: float,
    D_bore: float,
    L_stack: float
) -> float:
    """
    Calculate the output coefficient C = P / (n_s * D² * L).
    
    This is useful for comparing designs and checking against
    typical values for different motor classes.
    
    Args:
        power: Output power [W]
        rpm_sync: Synchronous speed [rpm]
        D_bore: Bore diameter [m]
        L_stack: Stack length [m]
    
    Returns:
        Output coefficient C [W/(rpm·m³)]
    """
    return power / (rpm_sync * D_bore**2 * L_stack)


def specific_magnetic_loading(Bm: float) -> float:
    """
    Calculate specific magnetic loading.
    
    This is essentially the average flux density over the pole pitch.
    For sinusoidal distribution: B_avg = (2/π) * Bm
    
    Args:
        Bm: Peak air gap flux density [T]
    
    Returns:
        Average flux density [T]
    """
    return (2 / math.pi) * Bm


def specific_electric_loading(
    current_phase: float,
    conductors_per_phase: float,
    D_bore: float
) -> float:
    """
    Calculate specific electric loading (linear current density).
    
    Δ = (m * N * I_f) / (π * D)
    
    where m is number of phases, N is total conductors per phase,
    I_f is phase current.
    
    Args:
        current_phase: Phase current [A]
        conductors_per_phase: Total series conductors per phase
        D_bore: Bore diameter [m]
    
    Returns:
        Linear current density [A/m]
    """
    # For 3-phase
    return (3 * conductors_per_phase * current_phase) / (math.pi * D_bore)
