"""
Induction Motor Design Package

A comprehensive framework for designing squirrel-cage induction motors.
Based on course notes from Prof. Giovanni Serra - University of Bologna.

Usage:
    from induction_motor_design import design_motor
    
    outputs = design_motor(
        power_kw=7.5,
        voltage=400,
        frequency=50,
        pole_pairs=3,
        rpm_rated=970,
        efficiency=0.883,
        power_factor=0.82
    )
"""

from .models import (
    # Nameplate
    NameplateData,
    create_nameplate_from_rpm,
    create_nameplate_from_slip,
    # Materials
    BHCurve,
    ElectricalSteel,
    ConductorMaterial,
    COPPER, ALUMINUM,
    create_M400_50A, create_M270_35A, create_M600_50A,
    # Lamination
    StatorSlot,
    RotorBar,
    EndRing,
    StatorLamination,
    RotorLamination,
    AirGap,
    LaminationAssembly,
    # Winding
    WindingConfiguration,
    WindingDesign
)

from .calculations import (
    # Preliminary
    PreliminarySizing,
    calculate_preliminary_dimensions,
    # Magnetic
    MagneticCircuitMMF,
    calculate_total_mmf,
    calculate_magnetizing_current,
    # Losses
    IronLosses,
    CopperLosses,
    TotalLosses,
    calculate_iron_losses,
    # Circuit
    CircuitParameters,
    CircuitSolution,
    PerformancePoint,
    solve_circuit,
    calculate_performance,
    calculate_torque_speed_curve
)

from .core import (
    DesignInputs,
    DesignOutputs,
    InductionMotorDesigner,
    design_motor,
    ConvergenceTracker,
    DesignHistory,
    GeneticOptimizer,
    GradientDescentOptimizer,
    DesignGenes,
    DesignFitness,
    Individual,
    optimize_motor_genetic,
    optimize_motor_gradient_descent,
    hybrid_optimize_motor
)

from .core.thermal_model import InsulationClass

from .utils import (
    MU_0,
    DesignRanges
)

__version__ = "1.0.0"
__author__ = "Refactored from MATLAB code (2016)"

__all__ = [
    # Main entry point
    'design_motor',
    'InductionMotorDesigner',
    'DesignInputs',
    'DesignOutputs',
    'GeneticOptimizer',
    'GradientDescentOptimizer',
    'DesignGenes',
    'DesignFitness',
    'Individual',
    'optimize_motor_genetic',
    'optimize_motor_gradient_descent',
    'hybrid_optimize_motor',
    'InsulationClass',
    
    # Models
    'NameplateData',
    'create_nameplate_from_rpm',
    'create_nameplate_from_slip',
    'ElectricalSteel',
    'ConductorMaterial',
    'COPPER', 'ALUMINUM',
    'StatorLamination',
    'RotorLamination',
    'LaminationAssembly',
    'WindingConfiguration',
    'WindingDesign',
    
    # Calculations
    'CircuitParameters',
    'PerformancePoint',
    'solve_circuit',
    'calculate_torque_speed_curve',
    
    # Utils
    'DesignRanges'
]
