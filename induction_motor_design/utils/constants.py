"""
Physical constants and typical design values for induction motor design.
Based on course notes from Prof. Giovanni Serra - University of Bologna.
"""

import math

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

MU_0 = 4 * math.pi * 1e-7  # Vacuum permeability [H/m]
RHO_CU_20 = 1.724e-8       # Copper resistivity at 20°C [Ω·m]
RHO_CU_75 = 2.112e-8       # Copper resistivity at 75°C [Ω·m]
RHO_AL_20 = 2.65e-8        # Aluminum resistivity at 20°C [Ω·m]
RHO_AL_75 = 3.45e-8        # Aluminum resistivity at 75°C [Ω·m]
DENSITY_CU = 8960          # Copper density [kg/m³]
DENSITY_AL = 2700          # Aluminum density [kg/m³]
DENSITY_FE = 7650          # Electrical steel density [kg/m³]


# =============================================================================
# TYPICAL DESIGN RANGES (for validation and initial guesses)
# =============================================================================

class DesignRanges:
    """Typical ranges for design parameters based on experience and course notes."""
    
    # Air gap flux density [T]
    BM_MIN = 0.7
    BM_MAX = 0.95
    BM_TYPICAL = 0.85
    
    # Stator tooth flux density [T]
    BD_STATOR_MAX = 1.8
    BD_STATOR_TYPICAL = 1.6
    
    # Stator yoke flux density [T]
    BC_STATOR_MIN = 1.1
    BC_STATOR_MAX = 1.5
    BC_STATOR_TYPICAL = 1.3
    
    # Rotor tooth flux density [T]
    BD_ROTOR_MAX = 1.9
    BD_ROTOR_TYPICAL = 1.7
    
    # Linear current density [A/m]
    DELTA_MIN = 15000
    DELTA_MAX = 45000
    DELTA_TYPICAL = 25000
    
    # Current density in conductors [A/m²]
    J_MIN = 3e6
    J_MAX = 8e6
    J_TYPICAL = 5e6
    
    # Slot fill factor (copper area / slot area)
    FILL_FACTOR_MIN = 0.35
    FILL_FACTOR_MAX = 0.45
    FILL_FACTOR_TYPICAL = 0.40
    
    # Aspect ratio m = L/τ (axial length / pole pitch)
    # Rule: m ≈ 0.3 * 2p for p < 4
    @staticmethod
    def aspect_ratio_typical(pole_pairs: int) -> float:
        """Typical aspect ratio based on number of pole pairs."""
        return 0.3 * 2 * pole_pairs


# =============================================================================
# EMF FACTOR
# =============================================================================

# Factor accounting for voltage drop on stator resistance and reactance
# E/V ratio typically 0.94-0.98
EMF_FACTOR_MIN = 0.94
EMF_FACTOR_MAX = 0.98
EMF_FACTOR_TYPICAL = 0.95


# =============================================================================
# CARTER FACTOR (Air gap correction)
# =============================================================================

CARTER_FACTOR_TYPICAL = 1.1  # Range: 1.05-1.2 depending on slot opening


# =============================================================================
# TEMPERATURE COEFFICIENTS
# =============================================================================

TEMP_COEFF_CU = 0.00393  # Temperature coefficient for copper [1/K]
TEMP_COEFF_AL = 0.00403  # Temperature coefficient for aluminum [1/K]


def resistivity_at_temp(rho_20: float, temp: float, temp_coeff: float) -> float:
    """
    Calculate resistivity at a given temperature.
    
    Args:
        rho_20: Resistivity at 20°C [Ω·m]
        temp: Operating temperature [°C]
        temp_coeff: Temperature coefficient [1/K]
    
    Returns:
        Resistivity at given temperature [Ω·m]
    """
    return rho_20 * (1 + temp_coeff * (temp - 20))
