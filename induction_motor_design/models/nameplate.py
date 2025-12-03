"""
Nameplate data model for induction motors.
Contains all rated specifications that define the motor requirements.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class NameplateData:
    """
    Motor nameplate (rated) specifications.
    
    These are the input requirements that drive the entire design process.
    All other parameters will be calculated to meet these specifications.
    
    Attributes:
        power_kw: Rated mechanical output power [kW]
        voltage_line: Rated line-to-line voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs p (poles = 2p)
        efficiency: Target efficiency at rated load (0-1)
        power_factor: Target power factor at rated load (0-1)
        rpm_rated: Rated speed [rpm] (optional - can be calculated from slip)
        slip_rated: Rated slip (optional - alternative to rpm_rated)
        phases: Number of phases (default 3)
        connection: Winding connection 'Y' (star) or 'D' (delta)
    """
    
    # Required parameters
    power_kw: float
    voltage_line: float
    frequency: float
    pole_pairs: int
    efficiency: float
    power_factor: float
    
    # Optional parameters
    rpm_rated: Optional[float] = None
    slip_rated: Optional[float] = None
    phases: int = 3
    connection: str = 'Y'
    
    # Derived quantities (computed in __post_init__)
    power_w: float = field(init=False)
    voltage_phase: float = field(init=False)
    omega_sync: float = field(init=False)
    rpm_sync: float = field(init=False)
    
    def __post_init__(self):
        """Validate inputs and compute derived quantities."""
        self._validate()
        self._compute_derived()
    
    def _validate(self):
        """Validate input parameters."""
        if self.power_kw <= 0:
            raise ValueError(f"Power must be positive, got {self.power_kw}")
        if self.voltage_line <= 0:
            raise ValueError(f"Voltage must be positive, got {self.voltage_line}")
        if self.frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency}")
        if self.pole_pairs < 1:
            raise ValueError(f"Pole pairs must be >= 1, got {self.pole_pairs}")
        if not 0 < self.efficiency <= 1:
            raise ValueError(f"Efficiency must be in (0, 1], got {self.efficiency}")
        if not 0 < self.power_factor <= 1:
            raise ValueError(f"Power factor must be in (0, 1], got {self.power_factor}")
        if self.phases not in [1, 3]:
            raise ValueError(f"Phases must be 1 or 3, got {self.phases}")
        if self.connection not in ['Y', 'D']:
            raise ValueError(f"Connection must be 'Y' or 'D', got {self.connection}")
        
        # Check that either rpm_rated or slip_rated is provided
        if self.rpm_rated is None and self.slip_rated is None:
            raise ValueError("Either rpm_rated or slip_rated must be provided")
    
    def _compute_derived(self):
        """Compute derived electrical quantities."""
        # Power in Watts
        self.power_w = self.power_kw * 1000
        
        # Synchronous speed
        self.rpm_sync = 60 * self.frequency / self.pole_pairs
        self.omega_sync = 2 * math.pi * self.frequency / self.pole_pairs
        
        # Phase voltage depends on connection
        if self.connection == 'Y':
            self.voltage_phase = self.voltage_line / math.sqrt(3)
        else:  # Delta
            self.voltage_phase = self.voltage_line
        
        # Compute slip if not provided
        if self.slip_rated is None:
            self.slip_rated = (self.rpm_sync - self.rpm_rated) / self.rpm_sync
        elif self.rpm_rated is None:
            self.rpm_rated = self.rpm_sync * (1 - self.slip_rated)
    
    @property
    def omega_rated(self) -> float:
        """Rated mechanical angular velocity [rad/s]."""
        return 2 * math.pi * self.rpm_rated / 60
    
    @property
    def torque_rated(self) -> float:
        """Rated torque [Nm]."""
        return self.power_w / self.omega_rated
    
    @property
    def apparent_power(self) -> float:
        """Apparent power [VA]."""
        return self.power_w / (self.efficiency * self.power_factor)
    
    @property
    def current_line(self) -> float:
        """Rated line current [A]."""
        return self.apparent_power / (math.sqrt(3) * self.voltage_line)
    
    @property
    def current_phase(self) -> float:
        """Rated phase current [A]."""
        if self.connection == 'Y':
            return self.current_line
        else:  # Delta
            return self.current_line / math.sqrt(3)
    
    @property
    def eta_cosfi(self) -> float:
        """Product of efficiency and power factor (useful for sizing)."""
        return self.efficiency * self.power_factor
    
    def __repr__(self) -> str:
        return (
            f"NameplateData(\n"
            f"  Power: {self.power_kw} kW @ {self.voltage_line} V, {self.frequency} Hz\n"
            f"  Poles: {2 * self.pole_pairs} ({self.pole_pairs} pairs)\n"
            f"  Sync speed: {self.rpm_sync:.0f} rpm, Rated: {self.rpm_rated:.0f} rpm\n"
            f"  Slip: {self.slip_rated:.4f} ({self.slip_rated*100:.2f}%)\n"
            f"  Efficiency: {self.efficiency:.3f}, Power Factor: {self.power_factor:.3f}\n"
            f"  Rated current: {self.current_line:.2f} A (line)\n"
            f"  Rated torque: {self.torque_rated:.2f} Nm\n"
            f")"
        )


# Factory functions for common motor types
def create_nameplate_from_rpm(
    power_kw: float,
    voltage_line: float,
    frequency: float,
    pole_pairs: int,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    connection: str = 'Y'
) -> NameplateData:
    """Create nameplate data specifying rated RPM."""
    return NameplateData(
        power_kw=power_kw,
        voltage_line=voltage_line,
        frequency=frequency,
        pole_pairs=pole_pairs,
        efficiency=efficiency,
        power_factor=power_factor,
        rpm_rated=rpm_rated,
        connection=connection
    )


def create_nameplate_from_slip(
    power_kw: float,
    voltage_line: float,
    frequency: float,
    pole_pairs: int,
    slip_rated: float,
    efficiency: float,
    power_factor: float,
    connection: str = 'Y'
) -> NameplateData:
    """Create nameplate data specifying rated slip."""
    return NameplateData(
        power_kw=power_kw,
        voltage_line=voltage_line,
        frequency=frequency,
        pole_pairs=pole_pairs,
        efficiency=efficiency,
        power_factor=power_factor,
        slip_rated=slip_rated,
        connection=connection
    )
