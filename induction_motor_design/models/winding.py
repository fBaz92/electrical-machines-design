"""
Stator winding model for induction motors.
Handles winding configuration, distribution, and pitch factors.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math
from ..utils.constants import MU_0


@dataclass
class WindingConfiguration:
    """
    Stator winding configuration.
    
    Attributes:
        N_slots: Total number of stator slots
        pole_pairs: Number of pole pairs
        phases: Number of phases (typically 3)
        layers: Number of winding layers (1 = single, 2 = double)
        coil_pitch_slots: Coil pitch in number of slots (for short pitching)
        parallel_paths: Number of parallel paths per phase
        conductors_per_slot: Number of conductors per slot
    """
    N_slots: int
    pole_pairs: int
    phases: int = 3
    layers: int = 2  # Double layer is most common
    coil_pitch_slots: Optional[int] = None  # None = full pitch
    parallel_paths: int = 1
    conductors_per_slot: int = 1
    
    # Derived quantities
    q: float = field(init=False)  # Slots per pole per phase
    alpha_e: float = field(init=False)  # Electrical angle between slots
    beta_e: float = field(init=False)  # Short pitch angle
    
    def __post_init__(self):
        self._compute_winding_parameters()
        self._validate()
    
    def _compute_winding_parameters(self):
        """Compute winding configuration parameters."""
        # Slots per pole per phase
        self.q = self.N_slots / (2 * self.pole_pairs * self.phases)
        
        # Electrical angle between adjacent slots [rad]
        self.alpha_e = 2 * math.pi * self.pole_pairs / self.N_slots
        
        # Full pitch in slots
        full_pitch = self.N_slots / (2 * self.pole_pairs)
        
        # Coil pitch (default to full pitch)
        if self.coil_pitch_slots is None:
            self.coil_pitch_slots = int(full_pitch)
        
        # Short pitch angle [rad electrical]
        shortening = full_pitch - self.coil_pitch_slots
        self.beta_e = shortening * self.alpha_e
    
    def _validate(self):
        """Validate winding configuration."""
        if self.N_slots % (2 * self.pole_pairs * self.phases) != 0:
            # Fractional slot winding - allowed but noted
            pass
        
        if self.coil_pitch_slots > self.N_slots / (2 * self.pole_pairs):
            raise ValueError("Coil pitch cannot exceed full pitch")
    
    @property
    def distribution_factor(self) -> float:
        """
        Distribution (breadth) factor K_d.
        
        K_d = sin(q * α/2) / (q * sin(α/2))
        
        Accounts for the fact that coil sides are distributed
        across multiple slots rather than concentrated.
        """
        if self.q == 1:
            return 1.0
        
        half_alpha = self.alpha_e / 2
        return math.sin(self.q * half_alpha) / (self.q * math.sin(half_alpha))
    
    @property
    def pitch_factor(self) -> float:
        """
        Pitch (chording) factor K_p.
        
        K_p = cos(β/2)
        
        Accounts for short-pitched (chorded) coils.
        """
        return math.cos(self.beta_e / 2)
    
    @property
    def winding_factor(self) -> float:
        """
        Total winding factor K_a = K_d * K_p.
        
        Combines distribution and pitch factors.
        Typical range: 0.85 - 0.96
        """
        return self.distribution_factor * self.pitch_factor
    
    def harmonic_winding_factor(self, harmonic: int) -> float:
        """
        Winding factor for a specific harmonic.
        
        Args:
            harmonic: Harmonic order (1 = fundamental, 5, 7, 11, 13, ...)
        
        Returns:
            Winding factor for that harmonic
        """
        # Distribution factor for harmonic
        half_alpha_nu = harmonic * self.alpha_e / 2
        if self.q == 1:
            k_d_nu = 1.0
        else:
            k_d_nu = math.sin(self.q * half_alpha_nu) / (self.q * math.sin(half_alpha_nu))
        
        # Pitch factor for harmonic
        k_p_nu = math.cos(harmonic * self.beta_e / 2)
        
        return k_d_nu * k_p_nu
    
    @property
    def total_conductors_per_phase(self) -> float:
        """Total series conductors per phase N = 2pqn."""
        return 2 * self.pole_pairs * self.q * self.conductors_per_slot
    
    @property
    def total_turns_per_phase(self) -> float:
        """Total series turns per phase (N/2)."""
        return self.total_conductors_per_phase / 2
    
    @property
    def effective_turns(self) -> float:
        """Effective turns per phase (K_a * N / parallel_paths)."""
        return self.winding_factor * self.total_conductors_per_phase / self.parallel_paths


@dataclass
class WindingDesign:
    """
    Complete winding design including conductor sizing.
    
    Attributes:
        config: Winding configuration
        conductor_area: Conductor cross-sectional area [m²]
        conductor_diameter: Conductor diameter [m] (for round wire)
        strands_in_hand: Number of parallel strands per conductor
        fill_factor: Slot fill factor (copper area / slot area)
        end_turn_length: One-side end turn length [m]
    """
    config: WindingConfiguration
    conductor_area: float
    conductor_diameter: Optional[float] = None
    strands_in_hand: int = 1
    fill_factor: float = 0.4
    end_turn_length: Optional[float] = None
    
    def __post_init__(self):
        # Calculate conductor diameter if not provided (round wire)
        if self.conductor_diameter is None:
            self.conductor_diameter = 2 * math.sqrt(self.conductor_area / math.pi)
    
    @property
    def total_conductor_area(self) -> float:
        """Total conductor area including strands [m²]."""
        return self.conductor_area * self.strands_in_hand
    
    def required_slot_area(self) -> float:
        """
        Required slot area based on fill factor [m²].
        """
        copper_per_slot = self.config.conductors_per_slot * self.total_conductor_area
        return copper_per_slot / self.fill_factor
    
    def phase_resistance(
        self, 
        stack_length: float, 
        resistivity: float,
        D_bore: float = None
    ) -> float:
        """
        Calculate phase resistance.
        
        R = ρ * L_total / (S_cond * n_parallel)
        
        where L_total is total conductor length per phase.
        
        Args:
            stack_length: Lamination stack length [m]
            resistivity: Conductor resistivity [Ω·m]
            D_bore: Bore diameter for end turn estimation [m]
        
        Returns:
            Phase resistance [Ω]
        """
        # Estimate end turn length if not provided
        if self.end_turn_length is None:
            if D_bore is None:
                raise ValueError("Need D_bore to estimate end turn length")
            # Empirical: L_end ≈ coil_span + some overhang
            pole_pitch = math.pi * D_bore / (2 * self.config.pole_pairs)
            coil_pitch_ratio = self.config.coil_pitch_slots / (
                self.config.N_slots / (2 * self.config.pole_pairs)
            )
            # End turn length per coil side (one direction)
            end_turn = 0.5 * pole_pitch * coil_pitch_ratio + 0.02  # + 20mm overhang
        else:
            end_turn = self.end_turn_length
        
        # Length of one conductor (in slot + 2 end turns)
        length_per_conductor = stack_length + 2 * end_turn
        
        # Number of series conductors per phase (accounting for parallel paths)
        N_series = self.config.total_conductors_per_phase / self.config.parallel_paths
        
        # Total length per phase
        total_length = N_series * length_per_conductor
        
        # R = ρL/A, with parallel strands
        R = resistivity * total_length / self.total_conductor_area
        
        return R


@dataclass
class LeakagePermeances:
    """
    Slot leakage permeance coefficients.
    
    Used for calculating leakage reactances.
    """
    slot_permeance: float      # λ_c - slot leakage
    end_turn_permeance: float  # λ_t - end turn (overhang) leakage
    harmonic_permeance: float  # λ_δ - differential/zigzag leakage
    
    @property
    def total(self) -> float:
        """Total permeance coefficient."""
        return self.slot_permeance + self.end_turn_permeance + self.harmonic_permeance


def calculate_stator_leakage_permeances(
    winding: WindingConfiguration,
    slot_permeance: float,
    pole_pitch: float,
    air_gap: float
) -> LeakagePermeances:
    """
    Calculate stator leakage permeance coefficients.
    
    Args:
        winding: Winding configuration
        slot_permeance: Slot geometry permeance [H/m]
        pole_pitch: Pole pitch τ [m]
        air_gap: Air gap length [m]
    
    Returns:
        LeakagePermeances object
    """
    # End turn permeance (empirical)
    lambda_t = MU_0 * 0.3 * winding.q
    
    # Harmonic permeance (belt leakage)
    # Sum contributions from 5th, 7th, 11th, 13th, etc.
    harmonics = [5, 7, 11, 13, 17, 19]
    contributions = 0
    
    for nu in harmonics:
        k_a_nu = winding.harmonic_winding_factor(nu)
        contributions += (k_a_nu / nu) ** 2
    
    lambda_delta = MU_0 * pole_pitch / air_gap * 3 / (math.pi ** 2) * contributions
    
    return LeakagePermeances(
        slot_permeance=slot_permeance,
        end_turn_permeance=lambda_t,
        harmonic_permeance=lambda_delta
    )


def calculate_stator_leakage_reactance(
    winding: WindingDesign,
    permeances: LeakagePermeances,
    frequency: float,
    stack_length: float,
    pole_pitch: float
) -> float:
    """
    Calculate stator leakage reactance X_s.
    
    X_s = X_c + X_t + X_δ
    
    Args:
        winding: Winding design
        permeances: Leakage permeances
        frequency: Supply frequency [Hz]
        stack_length: Stack length [m]
        pole_pitch: Pole pitch [m]
    
    Returns:
        Stator leakage reactance [Ω]
    """
    omega = 2 * math.pi * frequency
    p = winding.config.pole_pairs
    q = winding.config.q
    n = winding.config.conductors_per_slot
    
    # Slot reactance
    X_c = omega * 2 * p * q * n**2 * stack_length * permeances.slot_permeance / 3
    
    # End turn reactance  
    X_t = omega * 2 * p * q**2 * n**2 * pole_pitch * permeances.end_turn_permeance / 3
    
    # Harmonic reactance
    X_delta = omega * 2 * p * q**2 * n**2 * stack_length * permeances.harmonic_permeance / 3
    
    return X_c + X_t + X_delta


def calculate_conductors_per_slot(
    voltage_phase: float,
    frequency: float,
    winding_factor: float,
    q: float,
    Bm: float,
    stack_length: float,
    D_bore: float,
    emf_factor: float = 0.95
) -> float:
    """
    Calculate number of conductors per slot from design equation.
    
    n = √2 * V_f * k_E / (K_a * 2πf * q * B_m * L * D)
    
    Args:
        voltage_phase: Phase voltage [V]
        frequency: Supply frequency [Hz]
        winding_factor: Winding factor K_a
        q: Slots per pole per phase
        Bm: Air gap flux density [T]
        stack_length: Stack length [m]
        D_bore: Bore diameter [m]
        emf_factor: E/V ratio (typically 0.94-0.98)
    
    Returns:
        Number of conductors per slot (float - needs rounding)
    """
    n = (math.sqrt(2) * voltage_phase * emf_factor / 
         (winding_factor * 2 * math.pi * frequency * q * Bm * stack_length * D_bore))
    return n


def round_conductors(n: float, double_layer: bool = True) -> int:
    """
    Round conductors per slot to valid integer.
    
    For double-layer windings, must be even.
    
    Args:
        n: Calculated conductors per slot
        double_layer: True if double-layer winding
    
    Returns:
        Valid integer number of conductors
    """
    n_int = int(n)
    
    if double_layer and n_int % 2 != 0:
        # Round to nearest even number
        if n - n_int > 0.5:
            n_int += 1
        else:
            n_int -= 1
        n_int = max(2, n_int)  # Minimum 2 for double layer
    
    return max(1, n_int)
