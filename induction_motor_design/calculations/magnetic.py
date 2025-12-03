"""
Magnetic calculations for induction motor design.

Includes:
- MMF (ampere-turns) calculations for magnetic circuit
- Magnetizing reactance calculation
- Flux density distribution
"""

from dataclasses import dataclass
from typing import Optional
import math

from ..models.materials import ElectricalSteel, BHCurve
from ..models.lamination import StatorLamination, RotorLamination, AirGap
from ..utils.constants import MU_0


@dataclass
class MagneticCircuitMMF:
    """
    MMF (Ampere-turns) breakdown for the magnetic circuit.
    
    Total MMF = A_gap + A_teeth_s + A_teeth_r + A_yoke_s + A_yoke_r
    """
    A_air_gap: float       # Air gap MMF [A]
    A_teeth_stator: float  # Stator teeth MMF [A]
    A_teeth_rotor: float   # Rotor teeth MMF [A]
    A_yoke_stator: float   # Stator yoke MMF [A]
    A_yoke_rotor: float    # Rotor yoke MMF [A]
    
    @property
    def total(self) -> float:
        """Total MMF [A]."""
        return (self.A_air_gap + self.A_teeth_stator + self.A_teeth_rotor +
                self.A_yoke_stator + self.A_yoke_rotor)
    
    @property
    def saturation_factor(self) -> float:
        """
        Saturation factor k_s = (A_gap + A_teeth) / A_gap.
        
        Indicates how much the iron contributes to total MMF.
        k_s = 1 means no saturation (all MMF in air gap).
        Typical range: 1.2 - 1.8
        """
        A_teeth = self.A_teeth_stator + self.A_teeth_rotor
        return (self.A_air_gap + A_teeth) / self.A_air_gap


def calculate_air_gap_mmf(
    Bm: float,
    air_gap_effective: float
) -> float:
    """
    Calculate air gap MMF.
    
    A_gap = Bm * δ_eff / μ₀
    
    Args:
        Bm: Peak air gap flux density [T]
        air_gap_effective: Effective air gap with Carter factor [m]
    
    Returns:
        Air gap MMF [A]
    """
    return Bm * air_gap_effective / MU_0


def calculate_tooth_mmf(
    B_tooth: float,
    tooth_height: float,
    bh_curve: BHCurve,
    iterations: int = 20
) -> float:
    """
    Calculate tooth MMF accounting for flux density variation.
    
    The tooth flux density varies along the height due to leakage
    and fringing. An iterative approach is used to find the 
    effective MMF.
    
    Args:
        B_tooth: Flux density at tooth base (from air gap Bm) [T]
        tooth_height: Tooth height [m]
        bh_curve: B-H curve of the lamination steel
        iterations: Number of iterations for convergence
    
    Returns:
        Tooth MMF [A]
    """
    # Simple approach: assume average flux density
    # More accurate: integrate H along tooth height
    
    # Iterative calculation considering flux density drop
    B = B_tooth
    tol = 0.01
    
    for _ in range(iterations):
        H = bh_curve.H_from_B(B)
        # Flux density drops slightly due to MMF consumed
        B_new = B_tooth - MU_0 * H * 0.1  # Empirical correction
        
        if abs(B_new - B) / B < tol:
            break
        B = max(0.1, B_new)  # Prevent negative
    
    H_effective = bh_curve.H_from_B(B)
    return H_effective * tooth_height


def calculate_yoke_mmf(
    B_yoke: float,
    yoke_path_length: float,
    bh_curve: BHCurve,
    correction_factor: float = 0.85
) -> float:
    """
    Calculate yoke (back iron) MMF.
    
    The flux path in the yoke is approximately half the pole pitch.
    A correction factor accounts for non-uniform flux distribution.
    
    Args:
        B_yoke: Yoke flux density [T]
        yoke_path_length: Half the pole pitch in yoke [m]
        bh_curve: B-H curve of the lamination steel
        correction_factor: Accounts for flux non-uniformity (0.8-0.9)
    
    Returns:
        Yoke MMF [A]
    """
    # Apply correction for non-uniform distribution
    B_effective = B_yoke * correction_factor
    H = bh_curve.H_from_B(B_effective)
    return H * yoke_path_length


def calculate_total_mmf(
    Bm: float,
    stator: StatorLamination,
    rotor: RotorLamination,
    air_gap: AirGap,
    steel: ElectricalSteel,
    pole_pairs: int
) -> MagneticCircuitMMF:
    """
    Calculate complete magnetic circuit MMF.
    
    Args:
        Bm: Peak air gap flux density [T]
        stator: Stator lamination
        rotor: Rotor lamination
        air_gap: Air gap specification
        steel: Electrical steel properties
        pole_pairs: Number of pole pairs
    
    Returns:
        MagneticCircuitMMF breakdown
    """
    # Effective air gap with Carter factor
    delta_eff = air_gap.effective_length(
        stator.slot.a_opening,
        stator.slot_pitch
    )
    
    # Air gap MMF
    A_gap = calculate_air_gap_mmf(Bm, delta_eff)
    
    # Stator tooth flux density and MMF
    B_tooth_s = stator.tooth_flux_density(Bm)
    A_teeth_s = calculate_tooth_mmf(
        B_tooth_s, 
        stator.tooth_height,
        steel.bh_curve
    )
    
    # Rotor tooth flux density and MMF
    B_tooth_r = rotor.tooth_flux_density(Bm)
    A_teeth_r = calculate_tooth_mmf(
        B_tooth_r,
        rotor.slot_height,
        steel.bh_curve
    )
    
    # Stator yoke
    B_yoke_s = stator.yoke_flux_density(Bm, pole_pairs)
    tau_yoke_s = math.pi * stator.yoke_mean_diameter / (2 * 2 * pole_pairs)
    A_yoke_s = calculate_yoke_mmf(B_yoke_s, tau_yoke_s, steel.bh_curve)
    
    # Rotor yoke
    B_yoke_r = rotor.yoke_flux_density(Bm, pole_pairs)
    tau_yoke_r = math.pi * (rotor.D_inner + rotor.yoke_height) / (2 * 2 * pole_pairs)
    A_yoke_r = calculate_yoke_mmf(B_yoke_r, tau_yoke_r, steel.bh_curve)
    
    return MagneticCircuitMMF(
        A_air_gap=A_gap,
        A_teeth_stator=A_teeth_s,
        A_teeth_rotor=A_teeth_r,
        A_yoke_stator=A_yoke_s,
        A_yoke_rotor=A_yoke_r
    )


def calculate_magnetizing_current(
    mmf: MagneticCircuitMMF,
    winding_factor: float,
    conductors_per_slot: int,
    slots_per_pole_per_phase: float,
    phases: int = 3
) -> float:
    """
    Calculate magnetizing current from total MMF.
    
    I_μ = A_total * π / (3 * n * √2 * q * K_a)
    
    where n is conductors per slot, q is slots/pole/phase.
    
    Args:
        mmf: Magnetic circuit MMF
        winding_factor: Winding factor K_a
        conductors_per_slot: Conductors per slot n
        slots_per_pole_per_phase: Slots per pole per phase q
        phases: Number of phases
    
    Returns:
        Magnetizing current (phase value) [A]
    """
    I_mu = (mmf.total * math.pi / 
            (phases * conductors_per_slot * math.sqrt(2) * 
             slots_per_pole_per_phase * winding_factor))
    return I_mu


def calculate_magnetizing_reactance_simple(
    voltage_phase: float,
    Bm: float,
    D_bore: float,
    L_stack: float,
    air_gap: float,
    carter_factor: float = 1.7
) -> float:
    """
    Calculate magnetizing reactance using simplified formula.
    
    Xm = V² / Qm
    
    where Qm is the reactive power for magnetization.
    
    Args:
        voltage_phase: Phase voltage [V]
        Bm: Air gap flux density [T]
        D_bore: Bore diameter [m]
        L_stack: Stack length [m]
        air_gap: Air gap length [m]
        carter_factor: Carter correction factor
    
    Returns:
        Magnetizing reactance [Ω]
    """
    # Magnetic energy in air gap
    # W = (1/2) * B² / μ₀ * Volume
    volume_gap = math.pi * D_bore * L_stack * air_gap * carter_factor
    Qm = (0.5 * Bm**2 / MU_0) * volume_gap * 2 * math.pi * 50  # Assuming 50Hz
    
    return voltage_phase**2 / Qm


def calculate_magnetizing_reactance_rigorous(
    voltage_phase: float,
    magnetizing_current: float
) -> float:
    """
    Calculate magnetizing reactance from voltage and current.
    
    Xm = V_f / I_μ
    
    This is the rigorous calculation after determining the
    magnetizing current from MMF analysis.
    
    Args:
        voltage_phase: Phase voltage [V]
        magnetizing_current: Magnetizing current [A]
    
    Returns:
        Magnetizing reactance [Ω]
    """
    return voltage_phase / magnetizing_current


def calculate_flux_per_pole(
    Bm: float,
    pole_pitch: float,
    L_stack: float
) -> float:
    """
    Calculate flux per pole.
    
    Φ = (2/π) * Bm * τ * L
    
    Args:
        Bm: Peak air gap flux density [T]
        pole_pitch: Pole pitch [m]
        L_stack: Stack length [m]
    
    Returns:
        Flux per pole [Wb]
    """
    return (2 / math.pi) * Bm * pole_pitch * L_stack


def calculate_induced_emf(
    flux_per_pole: float,
    frequency: float,
    winding_factor: float,
    turns_per_phase: int
) -> float:
    """
    Calculate induced EMF per phase.
    
    E = 4.44 * f * Φ * K_a * N
    
    Args:
        flux_per_pole: Flux per pole [Wb]
        frequency: Supply frequency [Hz]
        winding_factor: Winding factor K_a
        turns_per_phase: Turns per phase
    
    Returns:
        Induced EMF [V]
    """
    return 4.44 * frequency * flux_per_pole * winding_factor * turns_per_phase


def calculate_Bm_from_voltage(
    voltage_phase: float,
    frequency: float,
    winding_factor: float,
    slots_per_pole_per_phase: float,
    conductors_per_slot: int,
    L_stack: float,
    D_bore: float,
    emf_factor: float = 0.95
) -> float:
    """
    Calculate required air gap flux density from voltage.
    
    Inverse of the EMF equation:
    Bm = √2 * V * k_E / (K_a * 2πf * q * n * L * D)
    
    Args:
        voltage_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        winding_factor: Winding factor K_a
        slots_per_pole_per_phase: Slots per pole per phase q
        conductors_per_slot: Conductors per slot n
        L_stack: Stack length [m]
        D_bore: Bore diameter [m]
        emf_factor: E/V ratio
    
    Returns:
        Required air gap flux density [T]
    """
    Bm = (math.sqrt(2) * voltage_phase * emf_factor /
          (winding_factor * 2 * math.pi * frequency * 
           slots_per_pole_per_phase * conductors_per_slot * L_stack * D_bore))
    return Bm


def verify_flux_densities(
    Bm: float,
    stator: StatorLamination,
    rotor: RotorLamination,
    pole_pairs: int
) -> dict:
    """
    Verify that flux densities are within acceptable limits.
    
    Args:
        Bm: Air gap flux density [T]
        stator: Stator lamination
        rotor: Rotor lamination  
        pole_pairs: Number of pole pairs
    
    Returns:
        Dictionary with verification results
    """
    from ..utils.constants import DesignRanges
    
    results = {
        'Bm': Bm,
        'B_tooth_stator': stator.tooth_flux_density(Bm),
        'B_tooth_rotor': rotor.tooth_flux_density(Bm),
        'B_yoke_stator': stator.yoke_flux_density(Bm, pole_pairs),
        'B_yoke_rotor': rotor.yoke_flux_density(Bm, pole_pairs),
        'warnings': []
    }
    
    # Check limits
    if results['B_tooth_stator'] > DesignRanges.BD_STATOR_MAX:
        results['warnings'].append(
            f"Stator tooth B={results['B_tooth_stator']:.2f}T exceeds limit"
        )
    
    if results['B_yoke_stator'] < DesignRanges.BC_STATOR_MIN:
        results['warnings'].append(
            f"Stator yoke B={results['B_yoke_stator']:.2f}T below typical minimum"
        )
    elif results['B_yoke_stator'] > DesignRanges.BC_STATOR_MAX:
        results['warnings'].append(
            f"Stator yoke B={results['B_yoke_stator']:.2f}T exceeds typical maximum"
        )
    
    results['valid'] = len(results['warnings']) == 0
    
    return results
