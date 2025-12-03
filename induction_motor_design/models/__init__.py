"""Data models for induction motor design."""

from .nameplate import (
    NameplateData,
    create_nameplate_from_rpm,
    create_nameplate_from_slip
)
from .materials import (
    BHCurve,
    ElectricalSteel,
    ConductorMaterial,
    COPPER, ALUMINUM,
    create_M400_50A, create_M270_35A, create_M600_50A
)
from .lamination import (
    StatorSlot,
    RotorBar,
    EndRing,
    StatorLamination,
    RotorLamination,
    AirGap,
    LaminationAssembly
)
from .winding import (
    WindingConfiguration,
    WindingDesign,
    LeakagePermeances,
    calculate_stator_leakage_permeances,
    calculate_stator_leakage_reactance,
    calculate_conductors_per_slot,
    round_conductors
)

__all__ = [
    # Nameplate
    'NameplateData',
    'create_nameplate_from_rpm',
    'create_nameplate_from_slip',
    # Materials
    'BHCurve',
    'ElectricalSteel', 
    'ConductorMaterial',
    'COPPER', 'ALUMINUM',
    'create_M400_50A', 'create_M270_35A', 'create_M600_50A',
    # Lamination
    'StatorSlot',
    'RotorBar',
    'EndRing',
    'StatorLamination',
    'RotorLamination',
    'AirGap',
    'LaminationAssembly',
    # Winding
    'WindingConfiguration',
    'WindingDesign',
    'LeakagePermeances',
    'calculate_stator_leakage_permeances',
    'calculate_stator_leakage_reactance',
    'calculate_conductors_per_slot',
    'round_conductors'
]
