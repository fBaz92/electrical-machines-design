"""
Material models for induction motor design.
Includes electrical steel (laminations), copper, and aluminum properties.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import math
from ..utils.constants import (
    MU_0, RHO_CU_75, RHO_AL_75, DENSITY_CU, DENSITY_AL, DENSITY_FE,
    TEMP_COEFF_CU, TEMP_COEFF_AL, resistivity_at_temp
)


# =============================================================================
# B-H CURVE MODELS
# =============================================================================

@dataclass
class BHCurve:
    """
    B-H curve model for electrical steel.
    
    Uses empirical formula: H = qn*B + qm*B^qk
    This captures both linear and saturation regions.
    
    Attributes:
        qn: Linear coefficient [A/m/T]
        qm: Saturation coefficient
        qk: Saturation exponent (typically 10-15)
        name: Material identifier
    """
    qn: float = 150.0       # Linear region slope
    qm: float = 4.461872    # Saturation coefficient
    qk: float = 13.05286    # Saturation exponent
    name: str = "M400-50A"
    
    def H_from_B(self, B: float) -> float:
        """
        Calculate magnetic field H from flux density B.
        
        Args:
            B: Flux density [T]
        
        Returns:
            Magnetic field intensity [A/m]
        """
        if B < 0:
            raise ValueError(f"B must be non-negative, got {B}")
        return self.qn * B + self.qm * (B ** self.qk)
    
    def B_from_H(self, H: float, tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Calculate flux density B from magnetic field H (inverse).
        Uses Newton-Raphson iteration.
        
        Args:
            H: Magnetic field intensity [A/m]
            tol: Convergence tolerance
            max_iter: Maximum iterations
        
        Returns:
            Flux density [T]
        """
        if H < 0:
            raise ValueError(f"H must be non-negative, got {H}")
        if H == 0:
            return 0.0
        
        # Initial guess (linear approximation)
        B = H / self.qn
        
        for _ in range(max_iter):
            f = self.H_from_B(B) - H
            df = self.qn + self.qm * self.qk * (B ** (self.qk - 1))
            
            if abs(df) < 1e-12:
                break
            
            B_new = B - f / df
            B_new = max(0, B_new)  # Ensure non-negative
            
            if abs(B_new - B) < tol:
                return B_new
            B = B_new
        
        return B
    
    def mu_r(self, B: float) -> float:
        """
        Calculate relative permeability at given B.
        
        Args:
            B: Flux density [T]
        
        Returns:
            Relative permeability (dimensionless)
        """
        if B < 1e-6:
            return self.qn / MU_0  # Linear region
        H = self.H_from_B(B)
        return B / (MU_0 * H)


# =============================================================================
# ELECTRICAL STEEL (LAMINATION)
# =============================================================================

@dataclass
class ElectricalSteel:
    """
    Electrical steel properties for laminations.
    
    Loss model: P_fe = k_i * f * B^α + k_cp * f² * B²  [W/kg]
    Where:
        - First term: Hysteresis losses
        - Second term: Eddy current losses
    
    Attributes:
        name: Material grade identifier
        thickness_mm: Lamination thickness [mm]
        density: Material density [kg/m³]
        p_1T_50Hz: Specific loss at 1T, 50Hz [W/kg]
        p_1_5T_50Hz: Specific loss at 1.5T, 50Hz [W/kg]
        p_1T_60Hz: Specific loss at 1T, 60Hz [W/kg]
        bh_curve: B-H curve model
        stacking_factor: Lamination stacking factor (0.95-0.97 typical)
    """
    name: str = "M400-50A"
    thickness_mm: float = 0.5
    density: float = DENSITY_FE
    p_1T_50Hz: float = 1.7      # W/kg
    p_1_5T_50Hz: float = 4.0    # W/kg  
    p_1T_60Hz: float = 2.18     # W/kg
    bh_curve: BHCurve = field(default_factory=BHCurve)
    stacking_factor: float = 0.96
    
    # Loss coefficients (computed in __post_init__)
    k_i: float = field(init=False)      # Hysteresis coefficient
    k_cp: float = field(init=False)     # Eddy current coefficient
    alpha: float = field(init=False)    # Hysteresis exponent
    
    def __post_init__(self):
        """Calculate loss coefficients from datasheet values."""
        self._compute_loss_coefficients()
    
    def _compute_loss_coefficients(self):
        """
        Derive k_i, k_cp, and alpha from datasheet loss values.
        
        From the course notes:
        - k_i = (1.44 * p_1T_50Hz - p_1T_60Hz) / 12
        - k_cp = (p_1T_60Hz - 60 * k_i) / 3600
        - alpha from: p_1.5T_50Hz = k_i * 50 * 1.5^α + k_cp * 50² * 1.5²
        """
        self.k_i = (1.44 * self.p_1T_50Hz - self.p_1T_60Hz) / 12
        self.k_cp = (self.p_1T_60Hz - 60 * self.k_i) / 3600
        
        # Solve for alpha
        # p_1.5T_50Hz - k_cp * 2500 * 2.25 = k_i * 50 * 1.5^α
        eddy_at_1_5T = self.k_cp * 50**2 * 1.5**2
        hysteresis_at_1_5T = self.p_1_5T_50Hz - eddy_at_1_5T
        
        if hysteresis_at_1_5T > 0 and self.k_i * 50 > 0:
            self.alpha = math.log(hysteresis_at_1_5T / (self.k_i * 50)) / math.log(1.5)
        else:
            self.alpha = 2.0  # Default Steinmetz exponent
    
    def specific_loss(self, B: float, f: float) -> float:
        """
        Calculate specific iron loss [W/kg].
        
        Args:
            B: Peak flux density [T]
            f: Frequency [Hz]
        
        Returns:
            Specific loss [W/kg]
        """
        p_hysteresis = self.k_i * f * (B ** self.alpha)
        p_eddy = self.k_cp * (f ** 2) * (B ** 2)
        return p_hysteresis + p_eddy
    
    def iron_loss(self, B: float, f: float, mass_kg: float) -> float:
        """
        Calculate total iron loss [W].
        
        Args:
            B: Peak flux density [T]
            f: Frequency [Hz]
            mass_kg: Iron mass [kg]
        
        Returns:
            Total loss [W]
        """
        return self.specific_loss(B, f) * mass_kg


# =============================================================================
# CONDUCTOR MATERIALS
# =============================================================================

@dataclass
class ConductorMaterial:
    """
    Conductor material properties (copper or aluminum).
    
    Attributes:
        name: Material name
        resistivity_20C: Resistivity at 20°C [Ω·m]
        temp_coefficient: Temperature coefficient [1/K]
        density: Material density [kg/m³]
        operating_temp: Default operating temperature [°C]
    """
    name: str
    resistivity_20C: float
    temp_coefficient: float
    density: float
    operating_temp: float = 75.0
    
    @property
    def resistivity(self) -> float:
        """Resistivity at operating temperature [Ω·m]."""
        return resistivity_at_temp(
            self.resistivity_20C, 
            self.operating_temp, 
            self.temp_coefficient
        )
    
    def resistance(self, length: float, area: float) -> float:
        """
        Calculate resistance of a conductor.
        
        Args:
            length: Conductor length [m]
            area: Cross-sectional area [m²]
        
        Returns:
            Resistance [Ω]
        """
        return self.resistivity * length / area
    
    def joule_loss(self, current: float, length: float, area: float) -> float:
        """
        Calculate Joule loss [W].
        
        Args:
            current: RMS current [A]
            length: Conductor length [m]
            area: Cross-sectional area [m²]
        
        Returns:
            Loss [W]
        """
        R = self.resistance(length, area)
        return R * current**2


# Pre-defined conductor materials
COPPER = ConductorMaterial(
    name="Copper",
    resistivity_20C=1.724e-8,
    temp_coefficient=TEMP_COEFF_CU,
    density=DENSITY_CU
)

ALUMINUM = ConductorMaterial(
    name="Aluminum", 
    resistivity_20C=2.65e-8,
    temp_coefficient=TEMP_COEFF_AL,
    density=DENSITY_AL
)


# =============================================================================
# STANDARD LAMINATION GRADES
# =============================================================================

def create_M400_50A() -> ElectricalSteel:
    """Create M400-50A electrical steel (common grade)."""
    return ElectricalSteel(
        name="M400-50A",
        thickness_mm=0.5,
        p_1T_50Hz=1.7,
        p_1_5T_50Hz=4.0,
        p_1T_60Hz=2.18
    )

def create_M270_35A() -> ElectricalSteel:
    """Create M270-35A electrical steel (premium grade)."""
    return ElectricalSteel(
        name="M270-35A",
        thickness_mm=0.35,
        p_1T_50Hz=1.1,
        p_1_5T_50Hz=2.7,
        p_1T_60Hz=1.4
    )

def create_M600_50A() -> ElectricalSteel:
    """Create M600-50A electrical steel (economy grade)."""
    return ElectricalSteel(
        name="M600-50A",
        thickness_mm=0.5,
        p_1T_50Hz=2.4,
        p_1_5T_50Hz=5.7,
        p_1T_60Hz=3.1
    )
