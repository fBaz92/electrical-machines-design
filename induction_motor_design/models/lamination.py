"""
Lamination geometry models for stator and rotor.
Defines slot shapes, dimensions, and derived geometric properties.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math
from ..utils.constants import MU_0


# =============================================================================
# STATOR SLOT GEOMETRY
# =============================================================================

@dataclass
class StatorSlot:
    """
    Stator slot geometry (trapezoidal semi-closed slot).
    
    Geometry parameters (from bottom to top):
        h1: Main slot height (winding area) [m]
        h2: Slot wedge height [m]
        h3: Slot opening chamfer height [m]
        h4: Slot opening height [m]
        a1: Slot width at bottom [m]
        a2: Slot width at top (if different from a1) [m]
        a_opening: Slot opening width [m]
    
    ASCII representation:
        ___a_opening___
       |     h4       |
        `----h3-----'
        |     h2     |
        |            |
        |     h1     |  <- Main winding area
        |            |
        |____a1_____|
    """
    h1: float           # Main slot height [m]
    h2: float = 0.0     # Wedge height [m]
    h3: float = 1e-3    # Chamfer height [m]
    h4: float = 0.7e-3  # Opening height [m]
    a1: float = 6.5e-3  # Slot width [m]
    a2: Optional[float] = None  # Top width (None = same as a1)
    a_opening: float = 2.5e-3   # Slot opening [m]
    
    def __post_init__(self):
        if self.a2 is None:
            self.a2 = self.a1
    
    @property
    def total_height(self) -> float:
        """Total slot height [m]."""
        return self.h1 + self.h2 + self.h3 + self.h4
    
    @property
    def area(self) -> float:
        """Approximate slot area [m²]."""
        # Trapezoidal approximation
        avg_width = (self.a1 + self.a2) / 2
        return avg_width * self.h1 + self.a2 * (self.h2 + self.h3) + self.a_opening * self.h4
    
    @property
    def winding_area(self) -> float:
        """Area available for winding [m²]."""
        return (self.a1 + self.a2) / 2 * self.h1
    
    def slot_permeance(self) -> float:
        """
        Slot permeance coefficient λ_c for leakage reactance calculation.
        
        Formula from course notes for trapezoidal semi-closed slots:
        λ_c = μ₀ * (h1/(3*a1) + h2/a1 + 2.3*h3/(a1+a_opening) + h4/a_opening)
        """
        lambda_c = MU_0 * (
            self.h1 / (3 * self.a1) +
            self.h2 / self.a1 +
            2.3 * self.h3 / (self.a1 + self.a_opening) +
            self.h4 / self.a_opening
        )
        return lambda_c


# =============================================================================
# ROTOR SLOT/BAR GEOMETRY
# =============================================================================

@dataclass 
class RotorBar:
    """
    Rotor bar geometry for squirrel cage rotors.
    
    Attributes:
        area: Bar cross-sectional area [m²]
        height: Bar height (slot depth) [m]
        width_top: Width at air gap side [m]
        width_bottom: Width at shaft side [m]
    """
    area: float
    height: float
    width_top: Optional[float] = None
    width_bottom: Optional[float] = None
    
    @property
    def average_width(self) -> float:
        """Average bar width [m]."""
        if self.width_top and self.width_bottom:
            return (self.width_top + self.width_bottom) / 2
        # Estimate from area and height
        return self.area / self.height


@dataclass
class EndRing:
    """
    End ring geometry for squirrel cage rotors.
    
    Attributes:
        area: Cross-sectional area [m²]
        mean_diameter: Mean diameter of the ring [m]
    """
    area: float
    mean_diameter: float
    
    @property
    def length(self) -> float:
        """Circumferential length [m]."""
        return math.pi * self.mean_diameter


# =============================================================================
# STATOR LAMINATION
# =============================================================================

@dataclass
class StatorLamination:
    """
    Complete stator lamination geometry.
    
    Attributes:
        D_bore: Bore diameter (air gap side) [m]
        D_outer: Outer diameter [m]
        L_stack: Axial stack length [m]
        N_slots: Number of stator slots
        slot: Slot geometry
        tooth_width: Tooth width [m]
        yoke_height: Back iron (yoke) height [m]
    """
    D_bore: float
    D_outer: float
    L_stack: float
    N_slots: int
    slot: StatorSlot
    tooth_width: float
    yoke_height: float
    
    # Derived quantities
    slot_pitch: float = field(init=False)
    pole_pitch: float = field(init=False, default=None)  # Set after knowing p
    
    def __post_init__(self):
        self.slot_pitch = math.pi * self.D_bore / self.N_slots
    
    def set_pole_pitch(self, pole_pairs: int):
        """Set pole pitch based on number of pole pairs."""
        self.pole_pitch = math.pi * self.D_bore / (2 * pole_pairs)
    
    @property
    def slot_area(self) -> float:
        """Slot cross-sectional area [m²]."""
        return self.slot.area
    
    @property
    def tooth_height(self) -> float:
        """Tooth height (same as slot total height) [m]."""
        return self.slot.total_height
    
    @property
    def yoke_mean_diameter(self) -> float:
        """Mean diameter of yoke [m]."""
        return self.D_outer - self.yoke_height
    
    @property
    def tooth_volume(self) -> float:
        """Total volume of all teeth [m³]."""
        return self.N_slots * self.tooth_width * self.tooth_height * self.L_stack
    
    @property
    def yoke_volume(self) -> float:
        """Volume of yoke [m³]."""
        return math.pi / 4 * (self.D_outer**2 - (self.D_outer - 2*self.yoke_height)**2) * self.L_stack
    
    def tooth_flux_density(self, Bm: float) -> float:
        """
        Calculate tooth flux density from air gap flux density.
        
        Args:
            Bm: Air gap flux density [T]
        
        Returns:
            Tooth flux density [T]
        """
        return Bm * self.slot_pitch / self.tooth_width
    
    def yoke_flux_density(self, Bm: float, pole_pairs: int) -> float:
        """
        Calculate yoke flux density from air gap flux density.
        
        Args:
            Bm: Air gap flux density [T]
            pole_pairs: Number of pole pairs
        
        Returns:
            Yoke flux density [T]
        """
        return Bm * self.D_bore / (2 * pole_pairs * self.yoke_height)


# =============================================================================
# ROTOR LAMINATION
# =============================================================================

@dataclass
class RotorLamination:
    """
    Rotor lamination geometry for squirrel cage motor.
    
    Attributes:
        D_outer: Outer diameter (air gap side) [m]
        D_inner: Inner diameter (shaft bore) [m]
        L_stack: Axial stack length [m]
        N_bars: Number of rotor bars
        bar: Bar geometry
        end_ring: End ring geometry
        tooth_width: Rotor tooth width [m]
        yoke_height: Rotor yoke height [m]
        skew_slots: Number of slots of skew (0 = no skew)
    """
    D_outer: float
    D_inner: float
    L_stack: float
    N_bars: int
    bar: RotorBar
    end_ring: Optional[EndRing] = None
    tooth_width: float = None
    yoke_height: float = None
    skew_slots: float = 0.0
    
    # Derived
    bar_pitch: float = field(init=False)
    
    def __post_init__(self):
        self.bar_pitch = math.pi * self.D_outer / self.N_bars
        
        # Calculate yoke height if not provided
        if self.yoke_height is None:
            self.yoke_height = (self.D_outer - self.D_inner) / 2 - self.bar.height
    
    @property
    def slot_height(self) -> float:
        """Rotor slot height [m]."""
        return self.bar.height
    
    def skew_factor(self, pole_pairs: int) -> float:
        """
        Calculate skew factor K_skew.
        
        Args:
            pole_pairs: Number of pole pairs
        
        Returns:
            Skew factor (0-1)
        """
        if self.skew_slots == 0:
            return 1.0
        
        # Skew angle in electrical radians
        alpha_skew = 2 * math.pi * pole_pairs * self.skew_slots / self.N_bars
        
        # sin(α/2) / (α/2)
        return math.sin(alpha_skew / 2) / (alpha_skew / 2)
    
    def tooth_flux_density(self, Bm: float) -> float:
        """
        Calculate rotor tooth flux density.
        
        Args:
            Bm: Air gap flux density [T]
        
        Returns:
            Tooth flux density [T]
        """
        return Bm * self.bar_pitch / self.tooth_width
    
    def yoke_flux_density(self, Bm: float, pole_pairs: int) -> float:
        """
        Calculate rotor yoke flux density.
        
        Args:
            Bm: Air gap flux density [T]
            pole_pairs: Number of pole pairs
        
        Returns:
            Yoke flux density [T]
        """
        return Bm * self.D_outer / (2 * pole_pairs * self.yoke_height)


# =============================================================================
# AIR GAP
# =============================================================================

@dataclass
class AirGap:
    """
    Air gap specification.
    
    Attributes:
        length: Mechanical air gap length [m]
        
    The effective air gap (considering Carter factor) is calculated
    based on slot openings.
    """
    length: float
    
    @staticmethod
    def empirical_minimum(D_bore: float) -> float:
        """
        Calculate empirical minimum air gap from bore diameter.
        
        Formula: δ_min = 3.06 - 6560/(D + 2280) [mm]
        where D is in mm.
        
        Args:
            D_bore: Bore diameter [m]
        
        Returns:
            Minimum air gap [m]
        """
        D_mm = D_bore * 1000
        delta_mm = 3.06 - 6560 / (D_mm + 2280)
        return delta_mm / 1000
    
    def carter_factor(self, slot_opening: float, slot_pitch: float) -> float:
        """
        Calculate Carter coefficient for effective air gap.
        
        Args:
            slot_opening: Slot opening width [m]
            slot_pitch: Slot pitch [m]
        
        Returns:
            Carter factor (> 1)
        """
        # γ = (b_o/δ) / (5 + b_o/δ)
        gamma = (slot_opening / self.length) / (5 + slot_opening / self.length)
        
        # k_c = 1 / (1 - γ * b_o/τ_s)
        k_c = 1 / (1 - gamma * slot_opening / slot_pitch)
        
        return k_c
    
    def effective_length(self, slot_opening: float, slot_pitch: float) -> float:
        """
        Calculate effective air gap length (with Carter factor).
        
        Args:
            slot_opening: Slot opening width [m]
            slot_pitch: Slot pitch [m]
        
        Returns:
            Effective air gap [m]
        """
        return self.length * self.carter_factor(slot_opening, slot_pitch)


# =============================================================================
# COMPLETE LAMINATION ASSEMBLY
# =============================================================================

@dataclass
class LaminationAssembly:
    """
    Complete lamination assembly (stator + rotor + air gap).
    
    This class combines stator and rotor laminations with the air gap
    and provides methods for geometric validation and calculations.
    """
    stator: StatorLamination
    rotor: RotorLamination
    air_gap: AirGap
    
    def __post_init__(self):
        self._validate_geometry()
    
    def _validate_geometry(self):
        """Validate geometric consistency."""
        # Check air gap consistency
        expected_gap = (self.stator.D_bore - self.rotor.D_outer) / 2
        if abs(expected_gap - self.air_gap.length) > 1e-6:
            # Auto-correct rotor outer diameter
            self.rotor.D_outer = self.stator.D_bore - 2 * self.air_gap.length
        
        # Check stack length consistency
        if self.stator.L_stack != self.rotor.L_stack:
            raise ValueError(
                f"Stack length mismatch: stator={self.stator.L_stack}, "
                f"rotor={self.rotor.L_stack}"
            )
    
    @property
    def D_bore(self) -> float:
        """Bore diameter [m]."""
        return self.stator.D_bore
    
    @property
    def L_stack(self) -> float:
        """Stack length [m]."""
        return self.stator.L_stack
    
    def carter_factor_total(self) -> float:
        """
        Calculate total Carter factor (stator + rotor contribution).
        """
        k_cs = self.air_gap.carter_factor(
            self.stator.slot.a_opening,
            self.stator.slot_pitch
        )
        # Rotor contribution (if slotted, otherwise 1)
        k_cr = 1.0  # For closed rotor slots
        
        return k_cs * k_cr
    
    def effective_air_gap(self) -> float:
        """Effective air gap with Carter factor [m]."""
        return self.air_gap.length * self.carter_factor_total()
