"""
Thermal Model for Induction Motor Steady-State Temperature Estimation.

This module implements a lumped-parameter thermal model to estimate the steady-state
temperature of an induction motor based on operating conditions, particularly the
current magnitude (both active and reactive components).

The model accounts for:
- Joule losses in stator windings (copper losses)
- Joule losses in rotor (bars and end rings)
- Iron losses (hysteresis + eddy current)
- Mechanical losses (windage + friction)
- Thermal resistance from windings to ambient
- Thermal capacity for transient analysis

Based on thermal analysis methods from CEI EN60034-1/IEC 34-1 standards.

Author: Motor Design Project
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import math

from ..models.lamination import LaminationAssembly
from ..models.winding import WindingDesign
from ..models.materials import ElectricalSteel, ConductorMaterial


# =============================================================================
# INSULATION CLASSES (per IEC 60034-1)
# =============================================================================

class InsulationClass(Enum):
    """
    Insulation classes with maximum temperature limits.
    
    The temperature limits include:
    - Reference ambient temperature: 40°C
    - Allowable temperature rise
    - Hot-spot allowance
    """
    A = ("A", 105, 60)      # 105°C max, 60K rise
    E = ("E", 120, 75)      # 120°C max, 75K rise
    B = ("B", 130, 80)      # 130°C max, 80K rise
    F = ("F", 155, 105)     # 155°C max, 105K rise
    H = ("H", 180, 125)     # 180°C max, 125K rise
    
    def __init__(self, class_name: str, max_temp: float, temp_rise: float):
        self.class_name = class_name
        self.max_temp = max_temp        # Maximum winding temperature [°C]
        self.temp_rise = temp_rise      # Allowable temperature rise [K]


# =============================================================================
# MATERIAL THERMAL PROPERTIES
# =============================================================================

@dataclass
class ThermalMaterial:
    """
    Thermal properties of motor materials.
    
    Attributes:
        name: Material identifier
        specific_heat: Specific heat capacity [J/(kg·K)]
        thermal_conductivity: Thermal conductivity [W/(m·K)]
        density: Material density [kg/m³]
    """
    name: str
    specific_heat: float        # c [J/(kg·K)]
    thermal_conductivity: float  # λ [W/(m·K)]
    density: float              # ρ [kg/m³]


# Standard materials with thermal properties
COPPER = ThermalMaterial(
    name="Copper",
    specific_heat=385,          # J/(kg·K)
    thermal_conductivity=380,   # W/(m·K) - from course notes
    density=8900                # kg/m³
)

ALUMINUM = ThermalMaterial(
    name="Aluminum",
    specific_heat=897,          # J/(kg·K)
    thermal_conductivity=218,   # W/(m·K) - from course notes
    density=2700                # kg/m³
)

ELECTRICAL_STEEL = ThermalMaterial(
    name="Electrical Steel",
    specific_heat=480,          # J/(kg·K)
    thermal_conductivity=45,    # W/(m·K) - typical for laminations
    density=7650                # kg/m³
)

INSULATION_CLASS_A = ThermalMaterial(
    name="Insulation Class A",
    specific_heat=1000,         # J/(kg·K) - typical for organic materials
    thermal_conductivity=0.1,   # W/(m·K) - from course notes
    density=1200                # kg/m³
)

INSULATION_CLASS_B = ThermalMaterial(
    name="Insulation Class B",
    specific_heat=1000,         # J/(kg·K)
    thermal_conductivity=0.15,  # W/(m·K) - from course notes
    density=1400                # kg/m³
)

AIR_40C = ThermalMaterial(
    name="Air at 40°C",
    specific_heat=1007,         # J/(kg·K)
    thermal_conductivity=0.027, # W/(m·K) - from course notes
    density=1.127               # kg/m³
)


# =============================================================================
# HEAT TRANSFER COEFFICIENTS
# =============================================================================

@dataclass
class ConvectionCoefficients:
    """
    Convection heat transfer coefficients for different motor surfaces.
    
    Based on empirical correlations from course notes.
    """
    
    @staticmethod
    def natural_convection_vertical(
        delta_T: float,
        height: float,
        T_ambient_K: float = 313.15,  # 40°C in Kelvin
        pressure_mmHg: float = 760
    ) -> float:
        """
        Natural convection coefficient for vertical surfaces.
        
        k = 2.6 * (ΔT / (T² * h))^(1/4) * (B/760)^(5/6)
        
        Args:
            delta_T: Temperature difference [K]
            height: Surface height [m]
            T_ambient_K: Ambient temperature [K]
            pressure_mmHg: Barometric pressure [mmHg]
        
        Returns:
            Convection coefficient [W/(m²·K)]
        """
        if delta_T <= 0 or height <= 0:
            return 5.0  # Default minimum value
        
        k = 2.6 * (delta_T / (T_ambient_K**2 * height))**0.25 * (pressure_mmHg / 760)**0.833
        return max(k, 2.0)  # Minimum practical value
    
    @staticmethod
    def forced_convection_surface(velocity: float) -> float:
        """
        Forced convection for air flowing over a surface.
        
        k_f = 0.01 * v^(1/3)  [W/(m²·K)]
        
        Args:
            velocity: Air velocity [m/s]
        
        Returns:
            Convection coefficient [W/(m²·K)]
        """
        if velocity <= 0:
            return 5.0  # Natural convection default
        return 0.01 * velocity**(1/3) * 1000  # Convert to W/(m²·K)
    
    @staticmethod
    def rotating_cylinder(peripheral_velocity: float, C1: float = 4.0) -> float:
        """
        Convection coefficient for rotating cylindrical surfaces.
        
        k_f = C1 * v^(1/3) * 10^(-5)
        
        Args:
            peripheral_velocity: Surface velocity [m/s]
            C1: Empirical constant (3-5 typical)
        
        Returns:
            Convection coefficient [W/(m²·K)]
        """
        if peripheral_velocity <= 0:
            return 5.0
        return C1 * peripheral_velocity**(1/3) * 1e-5 * 1e6  # Scaled for practical values
    
    @staticmethod
    def forced_ventilation(velocity: float, C2: float = 1.0) -> float:
        """
        Convection for strongly ventilated machines.
        
        k_f = 50 * (1 + C2 * v)
        
        Args:
            velocity: Air velocity [m/s]
            C2: Ventilation efficiency factor (0.6-1.3)
        
        Returns:
            Convection coefficient [W/(m²·K)]
        """
        return 50 * (1 + C2 * velocity)
    
    @staticmethod
    def channel_flow(velocity: float, hydraulic_diameter: float) -> float:
        """
        Forced convection in cooling channels (Gotter formula).
        
        k_f = M * (v/d)^0.75 * d^0.22
        
        Args:
            velocity: Flow velocity [m/s]
            hydraulic_diameter: Hydraulic diameter [m]
        
        Returns:
            Convection coefficient [W/(m²·K)]
        """
        M = 6.3  # Empirical constant
        if velocity <= 0 or hydraulic_diameter <= 0:
            return 10.0
        return M * (velocity / hydraulic_diameter)**0.75 * hydraulic_diameter**0.22


# =============================================================================
# THERMAL RESISTANCES
# =============================================================================

@dataclass
class ThermalResistance:
    """
    Thermal resistance calculation methods.
    
    R_th = ΔT / Q  [K/W]
    """
    
    @staticmethod
    def conduction(thickness: float, area: float, conductivity: float) -> float:
        """
        Thermal resistance for conduction through a plane wall.
        
        R = s / (λ * S)
        
        Args:
            thickness: Wall thickness [m]
            area: Heat transfer area [m²]
            conductivity: Thermal conductivity [W/(m·K)]
        
        Returns:
            Thermal resistance [K/W]
        """
        if area <= 0 or conductivity <= 0:
            return float('inf')
        return thickness / (conductivity * area)
    
    @staticmethod
    def convection(area: float, h_conv: float) -> float:
        """
        Thermal resistance for convection from a surface.
        
        R = 1 / (h * S)
        
        Args:
            area: Heat transfer area [m²]
            h_conv: Convection coefficient [W/(m²·K)]
        
        Returns:
            Thermal resistance [K/W]
        """
        if area <= 0 or h_conv <= 0:
            return float('inf')
        return 1.0 / (h_conv * area)
    
    @staticmethod
    def cylindrical_conduction(
        r_inner: float,
        r_outer: float,
        length: float,
        conductivity: float
    ) -> float:
        """
        Thermal resistance for radial conduction through a cylinder.
        
        R = ln(r_o/r_i) / (2π * λ * L)
        
        Args:
            r_inner: Inner radius [m]
            r_outer: Outer radius [m]
            length: Cylinder length [m]
            conductivity: Thermal conductivity [W/(m·K)]
        
        Returns:
            Thermal resistance [K/W]
        """
        if r_outer <= r_inner or length <= 0 or conductivity <= 0:
            return float('inf')
        return math.log(r_outer / r_inner) / (2 * math.pi * conductivity * length)
    
    @staticmethod
    def series(*resistances: float) -> float:
        """Total resistance for resistances in series."""
        return sum(r for r in resistances if r < float('inf'))
    
    @staticmethod
    def parallel(*resistances: float) -> float:
        """Total resistance for resistances in parallel."""
        valid = [r for r in resistances if r > 0 and r < float('inf')]
        if not valid:
            return float('inf')
        return 1.0 / sum(1.0/r for r in valid)


# =============================================================================
# THERMAL CAPACITY
# =============================================================================

@dataclass
class ThermalCapacity:
    """
    Thermal capacity (heat capacity) calculation.
    
    C = Σ(c_i * m_i)  [J/K]
    """
    
    @staticmethod
    def from_mass(mass: float, specific_heat: float) -> float:
        """
        Calculate thermal capacity from mass and specific heat.
        
        C = c * m
        
        Args:
            mass: Component mass [kg]
            specific_heat: Specific heat [J/(kg·K)]
        
        Returns:
            Thermal capacity [J/K]
        """
        return mass * specific_heat
    
    @staticmethod
    def from_volume(
        volume: float,
        density: float,
        specific_heat: float
    ) -> float:
        """
        Calculate thermal capacity from volume and material properties.
        
        C = c * ρ * V
        
        Args:
            volume: Component volume [m³]
            density: Material density [kg/m³]
            specific_heat: Specific heat [J/(kg·K)]
        
        Returns:
            Thermal capacity [J/K]
        """
        return specific_heat * density * volume


# =============================================================================
# MOTOR LOSSES CALCULATION
# =============================================================================

@dataclass
class MotorLosses:
    """
    Motor losses breakdown for thermal analysis.
    
    All losses in Watts.
    """
    P_cu_stator: float = 0.0    # Stator copper losses
    P_cu_rotor: float = 0.0     # Rotor copper losses (bars + rings)
    P_iron: float = 0.0         # Iron losses (stator + rotor)
    P_mechanical: float = 0.0   # Mechanical losses (friction + windage)
    P_stray: float = 0.0        # Stray load losses
    
    @property
    def total(self) -> float:
        """Total losses [W]."""
        return (self.P_cu_stator + self.P_cu_rotor + 
                self.P_iron + self.P_mechanical + self.P_stray)
    
    @property
    def winding_losses(self) -> float:
        """Total winding losses (creates heat in windings) [W]."""
        return self.P_cu_stator + self.P_cu_rotor
    
    @property
    def core_losses(self) -> float:
        """Core losses (creates heat in laminations) [W]."""
        return self.P_iron


def calculate_joule_losses(
    current: float,
    resistance: float,
    phases: int = 3
) -> float:
    """
    Calculate Joule (I²R) losses.
    
    P_j = m * R * I²
    
    Args:
        current: RMS current per phase [A]
        resistance: Resistance per phase [Ω]
        phases: Number of phases
    
    Returns:
        Joule losses [W]
    """
    return phases * resistance * current**2


def calculate_current_dependent_losses(
    I_active: float,
    I_reactive: float,
    R_stator: float,
    R_rotor: float,
    phases: int = 3
) -> Tuple[float, float]:
    """
    Calculate losses from active and reactive current components.
    
    The total current I = √(I_a² + I_r²)
    Both components contribute to Joule losses.
    
    Args:
        I_active: Active current component [A]
        I_reactive: Reactive (magnetizing) current component [A]
        R_stator: Stator resistance per phase [Ω]
        R_rotor: Rotor resistance per phase (referred to stator) [Ω]
        phases: Number of phases
    
    Returns:
        Tuple of (stator_losses, rotor_losses) [W]
    """
    # Total stator current magnitude
    I_stator = math.sqrt(I_active**2 + I_reactive**2)
    
    # Stator copper losses (from total current)
    P_cu_stator = phases * R_stator * I_stator**2
    
    # Rotor losses mainly from active component (load current)
    # Reactive current flows through magnetizing branch, not rotor
    P_cu_rotor = phases * R_rotor * I_active**2
    
    return P_cu_stator, P_cu_rotor


def estimate_resistance_at_temperature(
    R_20C: float,
    temperature: float,
    material: str = "copper"
) -> float:
    """
    Calculate resistance at operating temperature.
    
    R(T) = R_20 * (1 + α * (T - 20))
    
    Args:
        R_20C: Resistance at 20°C [Ω]
        temperature: Operating temperature [°C]
        material: "copper" or "aluminum"
    
    Returns:
        Resistance at temperature [Ω]
    """
    alpha = 0.00393 if material == "copper" else 0.00403
    return R_20C * (1 + alpha * (temperature - 20))


# =============================================================================
# MOTOR GEOMETRY FOR THERMAL MODEL
# =============================================================================

@dataclass
class MotorThermalGeometry:
    """
    Motor geometry parameters relevant for thermal analysis.
    
    All dimensions in SI units (meters, m²).
    """
    # Stator dimensions
    D_outer: float              # Stator outer diameter [m]
    D_bore: float               # Stator bore (inner) diameter [m]
    L_stack: float              # Lamination stack length [m]
    
    # Rotor dimensions
    D_rotor: float              # Rotor outer diameter [m]
    D_shaft: float              # Shaft diameter [m]
    
    # Material masses
    mass_stator_iron: float     # Stator lamination mass [kg]
    mass_rotor_iron: float      # Rotor lamination mass [kg]
    mass_stator_copper: float   # Stator winding copper mass [kg]
    mass_rotor_conductor: float # Rotor conductor mass [kg]
    
    # Optional: Detailed geometry
    slot_insulation_thickness: float = 0.0003   # Slot insulation [m]
    end_winding_surface: Optional[float] = None # End winding cooling surface [m²]
    frame_surface: Optional[float] = None       # Frame external surface [m²]
    
    @property
    def air_gap(self) -> float:
        """Air gap length [m]."""
        return (self.D_bore - self.D_rotor) / 2
    
    @property
    def stator_yoke_thickness(self) -> float:
        """Approximate stator yoke thickness [m]."""
        return (self.D_outer - self.D_bore) / 4  # Rough estimate
    
    @property
    def rotor_yoke_thickness(self) -> float:
        """Approximate rotor yoke thickness [m]."""
        return (self.D_rotor - self.D_shaft) / 4  # Rough estimate
    
    @property
    def rotor_surface_area(self) -> float:
        """Rotor cylindrical surface area [m²]."""
        return math.pi * self.D_rotor * self.L_stack
    
    @property
    def stator_bore_surface(self) -> float:
        """Stator bore surface area [m²]."""
        return math.pi * self.D_bore * self.L_stack
    
    @property
    def frame_surface_area(self) -> float:
        """External frame surface area [m²]."""
        if self.frame_surface is not None:
            return self.frame_surface
        # Estimate: cylinder + two end caps
        return (math.pi * self.D_outer * (self.L_stack + 0.1) + 
                math.pi * self.D_outer**2 / 2)


# =============================================================================
# LUMPED PARAMETER THERMAL MODEL
# =============================================================================

@dataclass
class ThermalModelParameters:
    """
    Parameters for the lumped thermal model.
    
    Simplified model with main thermal nodes:
    1. Stator winding (hottest point)
    2. Stator iron
    3. Rotor
    4. Frame/ambient
    """
    # Thermal resistances [K/W]
    R_winding_to_stator: float      # Winding to stator iron
    R_stator_to_frame: float        # Stator iron to frame
    R_frame_to_ambient: float       # Frame to ambient
    R_rotor_to_airgap: float        # Rotor to air gap
    R_airgap_to_stator: float       # Air gap to stator bore
    
    # Thermal capacities [J/K]
    C_stator_winding: float         # Stator winding thermal capacity
    C_stator_iron: float            # Stator iron thermal capacity
    C_rotor: float                  # Rotor thermal capacity
    
    # Convection parameters
    h_frame_natural: float = 10.0   # Natural convection at frame [W/(m²·K)]
    h_frame_forced: float = 50.0    # Forced convection (with fan) [W/(m²·K)]
    h_airgap: float = 100.0         # Air gap convection [W/(m²·K)]


@dataclass
class ThermalModelResult:
    """
    Results from thermal model calculation.
    """
    # Temperatures [°C]
    T_winding: float            # Stator winding temperature
    T_stator_iron: float        # Stator iron temperature
    T_rotor: float              # Rotor temperature
    T_frame: float              # Frame temperature
    T_ambient: float            # Ambient temperature
    
    # Temperature rises [K]
    delta_T_winding: float      # Winding temperature rise
    delta_T_stator: float       # Stator iron temperature rise
    delta_T_rotor: float        # Rotor temperature rise
    
    # Thermal parameters
    R_th_total: float           # Total thermal resistance [K/W]
    C_th_total: float           # Total thermal capacity [J/K]
    tau_thermal: float          # Thermal time constant [s]
    
    # Power dissipation
    P_total_losses: float       # Total losses [W]
    
    # Status
    insulation_class: InsulationClass
    temperature_margin: float   # Margin below max temperature [K]
    is_within_limits: bool      # Whether temperature is acceptable
    
    def __str__(self) -> str:
        return (
            f"Thermal Model Results:\n"
            f"  Winding Temperature: {self.T_winding:.1f}°C "
            f"(rise: {self.delta_T_winding:.1f}K)\n"
            f"  Stator Iron Temperature: {self.T_stator_iron:.1f}°C\n"
            f"  Rotor Temperature: {self.T_rotor:.1f}°C\n"
            f"  Frame Temperature: {self.T_frame:.1f}°C\n"
            f"  Ambient: {self.T_ambient:.1f}°C\n"
            f"  ---\n"
            f"  Insulation Class: {self.insulation_class.class_name}\n"
            f"  Max Allowed: {self.insulation_class.max_temp}°C\n"
            f"  Temperature Margin: {self.temperature_margin:.1f}K\n"
            f"  Within Limits: {'Yes' if self.is_within_limits else 'NO!'}\n"
            f"  ---\n"
            f"  Total Losses: {self.P_total_losses:.1f}W\n"
            f"  Thermal Resistance: {self.R_th_total:.4f} K/W\n"
            f"  Thermal Capacity: {self.C_th_total:.1f} J/K\n"
            f"  Time Constant: {self.tau_thermal/3600:.2f} hours"
        )


class MotorThermalModel:
    """
    Lumped-parameter thermal model for induction motors.
    
    Calculates steady-state temperatures based on losses and thermal
    network parameters. Also provides thermal capacity and resistance
    estimates for transient analysis.
    
    The model uses a simplified thermal network:
    
        P_cu_s     P_fe
          ↓         ↓
    [Winding]--R1--[Stator Iron]--R2--[Frame]--R3--[Ambient]
                        ↑
                       R4
                        |
                    [Air Gap]
                        |
                       R5
                        ↓
                    [Rotor] ← P_cu_r + P_mech
    
    """
    
    def __init__(
        self,
        geometry: MotorThermalGeometry,
        insulation_class: InsulationClass = InsulationClass.F,
        cooling_type: str = "TEFC",  # Totally Enclosed Fan Cooled
        T_ambient: float = 40.0
    ):
        """
        Initialize thermal model.
        
        Args:
            geometry: Motor thermal geometry
            insulation_class: Winding insulation class
            cooling_type: "TEFC", "ODP", "TENV" (Open Drip Proof, Totally Enclosed Non-Ventilated)
            T_ambient: Ambient temperature [°C]
        """
        self.geometry = geometry
        self.insulation_class = insulation_class
        self.cooling_type = cooling_type
        self.T_ambient = T_ambient
        
        # Calculate thermal parameters
        self._calculate_thermal_parameters()
    
    def _calculate_thermal_parameters(self):
        """Calculate thermal resistances and capacities."""
        g = self.geometry
        
        # =====================================================================
        # THERMAL CAPACITIES
        # =====================================================================
        
        # Stator winding thermal capacity
        self.C_winding = ThermalCapacity.from_mass(
            g.mass_stator_copper, 
            COPPER.specific_heat
        )
        
        # Stator iron thermal capacity
        self.C_stator_iron = ThermalCapacity.from_mass(
            g.mass_stator_iron,
            ELECTRICAL_STEEL.specific_heat
        )
        
        # Rotor thermal capacity (iron + conductor)
        self.C_rotor = (
            ThermalCapacity.from_mass(g.mass_rotor_iron, ELECTRICAL_STEEL.specific_heat) +
            ThermalCapacity.from_mass(g.mass_rotor_conductor, ALUMINUM.specific_heat)
        )
        
        # Total thermal capacity
        self.C_total = self.C_winding + self.C_stator_iron + self.C_rotor
        
        # =====================================================================
        # THERMAL RESISTANCES
        # =====================================================================
        
        # Slot insulation resistance (winding to stator iron)
        # Through slot liner + impregnation
        insulation_area = g.L_stack * 0.05 * 48  # Rough estimate: slot perimeter × slots
        self.R_slot_insulation = ThermalResistance.conduction(
            g.slot_insulation_thickness,
            insulation_area,
            INSULATION_CLASS_B.thermal_conductivity
        )
        
        # Stator iron radial resistance
        self.R_stator_radial = ThermalResistance.cylindrical_conduction(
            g.D_bore / 2,
            g.D_outer / 2,
            g.L_stack,
            ELECTRICAL_STEEL.thermal_conductivity
        )
        
        # Frame convection resistance
        h_conv = self._get_frame_convection_coefficient()
        self.R_frame_convection = ThermalResistance.convection(
            g.frame_surface_area,
            h_conv
        )
        
        # Air gap thermal resistance
        # Combination of convection on both sides
        h_airgap = self._estimate_airgap_convection()
        self.R_airgap = (
            ThermalResistance.convection(g.stator_bore_surface, h_airgap) +
            ThermalResistance.convection(g.rotor_surface_area, h_airgap)
        )
        
        # Total thermal resistance (simplified path: winding → ambient)
        self.R_total = (
            self.R_slot_insulation + 
            self.R_stator_radial + 
            self.R_frame_convection
        )
        
        # Thermal time constant
        self.tau_thermal = self.R_total * self.C_total
    
    def _get_frame_convection_coefficient(self) -> float:
        """Get convection coefficient based on cooling type."""
        if self.cooling_type == "TEFC":
            return 50.0  # Fan-cooled, forced convection
        elif self.cooling_type == "ODP":
            return 30.0  # Partially open, mixed convection
        else:  # TENV or natural
            return 10.0  # Natural convection only
    
    def _estimate_airgap_convection(self, rpm: float = 1500) -> float:
        """Estimate air gap convection coefficient based on rotor speed."""
        # Peripheral velocity
        v_periph = math.pi * self.geometry.D_rotor * rpm / 60
        
        # Taylor-Couette flow approximation
        return 20 + 0.5 * v_periph  # Simplified correlation
    
    def calculate_steady_state(
        self,
        losses: MotorLosses,
        rpm: float = 1500
    ) -> ThermalModelResult:
        """
        Calculate steady-state temperatures.
        
        At steady state: ΔT = P × R_th
        
        Args:
            losses: Motor losses breakdown
            rpm: Operating speed [rpm]
        
        Returns:
            ThermalModelResult with all temperatures
        """
        # Update air gap convection for actual speed
        h_airgap = self._estimate_airgap_convection(rpm)
        
        # =====================================================================
        # TEMPERATURE CALCULATION (simplified network)
        # =====================================================================
        
        # Frame temperature rise (from ambient)
        # All losses eventually reach the frame
        delta_T_frame = losses.total * self.R_frame_convection
        T_frame = self.T_ambient + delta_T_frame
        
        # Stator iron temperature
        # Receives iron losses directly + winding losses through insulation
        delta_T_stator_iron = (
            losses.total * (self.R_stator_radial + self.R_frame_convection) +
            losses.P_iron * 0.1  # Additional rise from iron losses
        )
        T_stator_iron = self.T_ambient + delta_T_stator_iron
        
        # Winding temperature (hottest point)
        # Must account for all thermal resistances from winding to ambient
        delta_T_winding = (
            losses.P_cu_stator * self.R_slot_insulation +  # Copper losses through insulation
            delta_T_stator_iron  # Plus stator iron rise
        )
        T_winding = self.T_ambient + delta_T_winding
        
        # Rotor temperature
        # Heat path: rotor → air gap → stator bore
        # The rotor is cooled by rotating air gap flow (Taylor-Couette)
        # Limit effective resistance for realistic results
        R_rotor_effective = min(self.R_airgap, 0.15)  # Cap at realistic value
        delta_T_rotor = (
            losses.P_cu_rotor * R_rotor_effective +  # Rotor copper losses
            losses.P_mechanical * R_rotor_effective * 0.3 +  # Mechanical losses (mostly at bearings)
            delta_T_stator_iron * 0.6  # Thermal coupling through air gap
        )
        T_rotor = self.T_ambient + delta_T_rotor
        
        # =====================================================================
        # CHECK AGAINST LIMITS
        # =====================================================================
        
        temp_margin = self.insulation_class.max_temp - T_winding
        is_within_limits = T_winding <= self.insulation_class.max_temp
        
        return ThermalModelResult(
            T_winding=T_winding,
            T_stator_iron=T_stator_iron,
            T_rotor=T_rotor,
            T_frame=T_frame,
            T_ambient=self.T_ambient,
            delta_T_winding=delta_T_winding,
            delta_T_stator=delta_T_stator_iron,
            delta_T_rotor=delta_T_rotor,
            R_th_total=self.R_total,
            C_th_total=self.C_total,
            tau_thermal=self.tau_thermal,
            P_total_losses=losses.total,
            insulation_class=self.insulation_class,
            temperature_margin=temp_margin,
            is_within_limits=is_within_limits
        )
    
    def calculate_from_current(
        self,
        I_phase: float,
        R_stator: float,
        R_rotor: float,
        P_iron: float,
        P_mechanical: float,
        power_factor: float = 0.85,
        rpm: float = 1500
    ) -> ThermalModelResult:
        """
        Calculate steady-state temperature from phase current.
        
        This is the main interface for estimating temperature based on
        operating current magnitude.
        
        Args:
            I_phase: Phase current magnitude [A]
            R_stator: Stator resistance at operating temperature [Ω]
            R_rotor: Rotor resistance (referred to stator) [Ω]
            P_iron: Iron losses [W]
            P_mechanical: Mechanical losses [W]
            power_factor: Operating power factor
            rpm: Operating speed [rpm]
        
        Returns:
            ThermalModelResult
        """
        # Decompose current into active and reactive components
        I_active = I_phase * power_factor
        I_reactive = I_phase * math.sqrt(1 - power_factor**2)
        
        # Calculate losses
        P_cu_stator, P_cu_rotor = calculate_current_dependent_losses(
            I_active, I_reactive, R_stator, R_rotor
        )
        
        # Stray losses (typically 0.5-1% of rated power, approximated)
        P_stray = 0.005 * (P_cu_stator + P_cu_rotor + P_iron)
        
        losses = MotorLosses(
            P_cu_stator=P_cu_stator,
            P_cu_rotor=P_cu_rotor,
            P_iron=P_iron,
            P_mechanical=P_mechanical,
            P_stray=P_stray
        )
        
        return self.calculate_steady_state(losses, rpm)
    
    def estimate_max_current(
        self,
        R_stator: float,
        R_rotor: float,
        P_iron: float,
        P_mechanical: float,
        power_factor: float = 0.85,
        rpm: float = 1500,
        safety_margin: float = 5.0
    ) -> float:
        """
        Estimate maximum allowable current based on thermal limits.
        
        Finds the current that brings winding temperature to the
        insulation class limit minus safety margin.
        
        Args:
            R_stator: Stator resistance [Ω]
            R_rotor: Rotor resistance [Ω]
            P_iron: Iron losses [W]
            P_mechanical: Mechanical losses [W]
            power_factor: Operating power factor
            rpm: Operating speed [rpm]
            safety_margin: Temperature safety margin [K]
        
        Returns:
            Maximum allowable phase current [A]
        """
        T_max = self.insulation_class.max_temp - safety_margin
        
        # Binary search for maximum current
        I_low, I_high = 0.0, 1000.0
        
        for _ in range(50):
            I_mid = (I_low + I_high) / 2
            result = self.calculate_from_current(
                I_mid, R_stator, R_rotor, P_iron, P_mechanical, power_factor, rpm
            )
            
            if result.T_winding < T_max:
                I_low = I_mid
            else:
                I_high = I_mid
            
            if abs(I_high - I_low) < 0.1:
                break
        
        return I_low
    
    def transient_response(
        self,
        losses: MotorLosses,
        time_points: List[float],
        T_initial: float = None
    ) -> List[float]:
        """
        Calculate transient temperature response.
        
        T(t) = T_ambient + ΔT_ss * (1 - exp(-t/τ)) + (T_0 - T_ambient) * exp(-t/τ)
        
        Args:
            losses: Motor losses
            time_points: Time points for evaluation [s]
            T_initial: Initial winding temperature [°C]
        
        Returns:
            List of winding temperatures at each time point [°C]
        """
        if T_initial is None:
            T_initial = self.T_ambient
        
        # Steady-state temperature rise
        result = self.calculate_steady_state(losses)
        T_ss = result.T_winding
        
        # Calculate transient
        temperatures = []
        for t in time_points:
            exp_term = math.exp(-t / self.tau_thermal)
            T = self.T_ambient + (T_ss - self.T_ambient) * (1 - exp_term) + \
                (T_initial - self.T_ambient) * exp_term
            temperatures.append(T)
        
        return temperatures


# =============================================================================
# SIMPLIFIED THERMAL MODEL ESTIMATOR
# =============================================================================

def estimate_thermal_parameters(
    P_rated: float,
    V_rated: float,
    I_rated: float,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    D_outer: float,
    D_bore: float,
    L_stack: float,
    pole_pairs: int = 2,
    insulation_class: InsulationClass = InsulationClass.F
) -> Tuple[float, float, float]:
    """
    Estimate thermal parameters from basic motor specifications.
    
    This function provides quick estimates when detailed geometry
    is not available.
    
    Args:
        P_rated: Rated output power [W]
        V_rated: Rated line voltage [V]
        I_rated: Rated line current [A]
        rpm_rated: Rated speed [rpm]
        efficiency: Motor efficiency (0-1)
        power_factor: Power factor (0-1)
        D_outer: Stator outer diameter [m]
        D_bore: Stator bore diameter [m]
        L_stack: Stack length [m]
        pole_pairs: Number of pole pairs
        insulation_class: Insulation class
    
    Returns:
        Tuple of (R_thermal, C_thermal, tau_thermal) in [K/W], [J/K], [s]
    """
    # Estimate total losses
    P_input = P_rated / efficiency
    P_losses = P_input - P_rated
    
    # Temperature rise at rated load
    T_rise = insulation_class.temp_rise
    
    # Estimate thermal resistance
    R_thermal = T_rise / P_losses
    
    # Estimate mass (empirical correlation)
    # Motor mass approximately proportional to D²L
    volume_active = math.pi / 4 * D_outer**2 * L_stack
    density_avg = 5000  # kg/m³ average for motor
    mass_total = density_avg * volume_active * 1.5  # Include frame, etc.
    
    # Estimate thermal capacity
    c_avg = 500  # J/(kg·K) average specific heat
    C_thermal = mass_total * c_avg
    
    # Thermal time constant
    tau_thermal = R_thermal * C_thermal
    
    return R_thermal, C_thermal, tau_thermal


def quick_temperature_estimate(
    I_current: float,
    I_rated: float,
    T_rise_rated: float,
    T_ambient: float = 40.0
) -> float:
    """
    Quick temperature estimate using current scaling.
    
    Temperature rise is approximately proportional to I²:
    ΔT = ΔT_rated * (I/I_rated)²
    
    Args:
        I_current: Operating current [A]
        I_rated: Rated current [A]
        T_rise_rated: Temperature rise at rated current [K]
        T_ambient: Ambient temperature [°C]
    
    Returns:
        Estimated winding temperature [°C]
    """
    T_rise = T_rise_rated * (I_current / I_rated)**2
    return T_ambient + T_rise


def build_thermal_geometry_from_design(
    lamination: LaminationAssembly,
    winding: WindingDesign,
    stator_conductor: ConductorMaterial,
    rotor_conductor: ConductorMaterial,
    steel: ElectricalSteel,
    slot_insulation_thickness: float = 0.0003
) -> MotorThermalGeometry:
    """Construct MotorThermalGeometry from electromagnetic design data."""
    stacking = getattr(steel, 'stacking_factor', 1.0)
    
    stator_volume = (lamination.stator.tooth_volume + lamination.stator.yoke_volume) * stacking
    mass_stator_iron = stator_volume * steel.density
    
    slot_copper_area = lamination.stator.slot_area * winding.fill_factor
    end_turn = winding.end_turn_length_effective(lamination.stator.D_bore)
    effective_length = lamination.L_stack + 2 * end_turn
    copper_volume = slot_copper_area * lamination.stator.N_slots * effective_length
    mass_stator_copper = copper_volume * stator_conductor.density
    
    rotor_total_volume = (
        math.pi / 4 * (lamination.rotor.D_outer**2 - lamination.rotor.D_inner**2) *
        lamination.rotor.L_stack
    )
    bar_volume = lamination.rotor.bar.area * lamination.rotor.L_stack * lamination.rotor.N_bars
    end_ring_volume = 0.0
    if lamination.rotor.end_ring is not None:
        end_ring_volume = lamination.rotor.end_ring.area * lamination.rotor.end_ring.length * 2
    conductor_volume = bar_volume + end_ring_volume
    iron_volume = max(rotor_total_volume - bar_volume, 0.0)
    
    mass_rotor_iron = iron_volume * steel.density
    mass_rotor_conductor = conductor_volume * rotor_conductor.density
    
    frame_surface = (
        math.pi * lamination.stator.D_outer * (lamination.L_stack + 0.1) +
        math.pi * lamination.stator.D_outer**2 / 2
    )
    
    return MotorThermalGeometry(
        D_outer=lamination.stator.D_outer,
        D_bore=lamination.stator.D_bore,
        L_stack=lamination.L_stack,
        D_rotor=lamination.rotor.D_outer,
        D_shaft=lamination.rotor.D_inner,
        mass_stator_iron=mass_stator_iron,
        mass_rotor_iron=mass_rotor_iron,
        mass_stator_copper=mass_stator_copper,
        mass_rotor_conductor=mass_rotor_conductor,
        slot_insulation_thickness=slot_insulation_thickness,
        frame_surface=frame_surface
    )


def evaluate_design_thermal_performance(
    lamination: LaminationAssembly,
    winding: WindingDesign,
    stator_conductor: ConductorMaterial,
    rotor_conductor: ConductorMaterial,
    steel: ElectricalSteel,
    losses: MotorLosses,
    rpm: float,
    insulation_class: InsulationClass,
    ambient_temp: float,
    cooling_type: str = "TEFC"
) -> ThermalModelResult:
    """Evaluate steady-state temperature for a completed design."""
    geometry = build_thermal_geometry_from_design(
        lamination=lamination,
        winding=winding,
        stator_conductor=stator_conductor,
        rotor_conductor=rotor_conductor,
        steel=steel
    )
    model = MotorThermalModel(
        geometry=geometry,
        insulation_class=insulation_class,
        cooling_type=cooling_type,
        T_ambient=ambient_temp
    )
    return model.calculate_steady_state(losses, rpm=rpm)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: 7.5 kW, 400V, 4-pole motor
    
    # Define motor geometry
    geometry = MotorThermalGeometry(
        D_outer=0.200,              # 200 mm outer diameter
        D_bore=0.110,               # 110 mm bore diameter
        L_stack=0.120,              # 120 mm stack length
        D_rotor=0.1094,             # 109.4 mm rotor diameter (0.3mm air gap)
        D_shaft=0.040,              # 40 mm shaft
        mass_stator_iron=15.0,      # 15 kg stator laminations
        mass_rotor_iron=8.0,        # 8 kg rotor laminations
        mass_stator_copper=4.0,     # 4 kg stator copper
        mass_rotor_conductor=2.0,   # 2 kg rotor aluminum
        slot_insulation_thickness=0.0003  # 0.3 mm
    )
    
    # Create thermal model
    model = MotorThermalModel(
        geometry=geometry,
        insulation_class=InsulationClass.F,
        cooling_type="TEFC",
        T_ambient=40.0
    )
    
    print("=" * 60)
    print("INDUCTION MOTOR THERMAL MODEL")
    print("=" * 60)
    
    # Print thermal parameters
    print(f"\nThermal Parameters:")
    print(f"  Total Thermal Capacity: {model.C_total:.1f} J/K")
    print(f"  Total Thermal Resistance: {model.R_total:.4f} K/W")
    print(f"  Thermal Time Constant: {model.tau_thermal/3600:.2f} hours")
    
    # Calculate at rated current
    print("\n" + "-" * 60)
    print("Steady-State Analysis at Rated Load")
    print("-" * 60)
    
    result = model.calculate_from_current(
        I_phase=14.0,           # Rated current ~14A for 7.5kW motor
        R_stator=0.5,           # Stator resistance
        R_rotor=0.35,           # Rotor resistance (referred)
        P_iron=200,             # Iron losses
        P_mechanical=50,        # Mechanical losses
        power_factor=0.85,
        rpm=1450
    )
    
    print(result)
    
    # Calculate at overload
    print("\n" + "-" * 60)
    print("Steady-State Analysis at 120% Overload")
    print("-" * 60)
    
    result_overload = model.calculate_from_current(
        I_phase=14.0 * 1.2,     # 120% current
        R_stator=0.5,
        R_rotor=0.35,
        P_iron=200,
        P_mechanical=50,
        power_factor=0.82,      # Lower PF at overload
        rpm=1440
    )
    
    print(result_overload)
    
    # Estimate maximum current
    print("\n" + "-" * 60)
    print("Maximum Allowable Current")
    print("-" * 60)
    
    I_max = model.estimate_max_current(
        R_stator=0.5,
        R_rotor=0.35,
        P_iron=200,
        P_mechanical=50,
        power_factor=0.85,
        rpm=1450,
        safety_margin=10.0
    )
    
    print(f"Maximum allowable current (10K margin): {I_max:.1f} A")
    print(f"Ratio to rated: {I_max/14.0:.2f}")
    
    # Quick estimate comparison
    print("\n" + "-" * 60)
    print("Quick Estimate Method")
    print("-" * 60)
    
    T_quick = quick_temperature_estimate(
        I_current=14.0,
        I_rated=14.0,
        T_rise_rated=105,  # Class F rise
        T_ambient=40.0
    )
    print(f"Quick estimate at rated: {T_quick:.1f}°C")
    
    T_quick_120 = quick_temperature_estimate(
        I_current=14.0 * 1.2,
        I_rated=14.0,
        T_rise_rated=105,
        T_ambient=40.0
    )
    print(f"Quick estimate at 120%: {T_quick_120:.1f}°C")
