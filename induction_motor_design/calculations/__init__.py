"""Calculation modules for induction motor design."""

from .preliminary import (
    PreliminarySizing,
    esson_output_equation,
    calculate_preliminary_dimensions,
    scale_to_lamination,
    verify_sizing_limits,
    calculate_output_coefficient,
    specific_magnetic_loading,
    specific_electric_loading
)

from .magnetic import (
    MagneticCircuitMMF,
    calculate_air_gap_mmf,
    calculate_tooth_mmf,
    calculate_yoke_mmf,
    calculate_total_mmf,
    calculate_magnetizing_current,
    calculate_magnetizing_reactance_simple,
    calculate_magnetizing_reactance_rigorous,
    calculate_flux_per_pole,
    calculate_induced_emf,
    calculate_Bm_from_voltage,
    verify_flux_densities
)

from .losses import (
    IronLosses,
    CopperLosses,
    TotalLosses,
    ResistanceFe,
    calculate_iron_losses,
    calculate_stator_copper_loss,
    calculate_rotor_bar_current,
    calculate_end_ring_current,
    calculate_rotor_losses,
    calculate_mechanical_losses,
    calculate_slip_from_losses,
    calculate_efficiency,
    calculate_power_factor_from_reactive,
    calculate_reactive_power
)

from .equivalent_circuit import (
    CircuitModel,
    CircuitParameters,
    CircuitSolution,
    PerformancePoint,
    solve_circuit,
    calculate_performance,
    calculate_torque_speed_curve,
    find_rated_slip,
    calculate_breakdown_torque,
    calculate_starting_performance,
    refer_rotor_to_stator
)

__all__ = [
    # Preliminary
    'PreliminarySizing',
    'esson_output_equation',
    'calculate_preliminary_dimensions',
    'scale_to_lamination',
    'verify_sizing_limits',
    'calculate_output_coefficient',
    'specific_magnetic_loading',
    'specific_electric_loading',
    
    # Magnetic
    'MagneticCircuitMMF',
    'calculate_air_gap_mmf',
    'calculate_tooth_mmf',
    'calculate_yoke_mmf',
    'calculate_total_mmf',
    'calculate_magnetizing_current',
    'calculate_magnetizing_reactance_simple',
    'calculate_magnetizing_reactance_rigorous',
    'calculate_flux_per_pole',
    'calculate_induced_emf',
    'calculate_Bm_from_voltage',
    'verify_flux_densities',
    
    # Losses
    'IronLosses',
    'CopperLosses',
    'TotalLosses',
    'ResistanceFe',
    'calculate_iron_losses',
    'calculate_stator_copper_loss',
    'calculate_rotor_bar_current',
    'calculate_end_ring_current',
    'calculate_rotor_losses',
    'calculate_mechanical_losses',
    'calculate_slip_from_losses',
    'calculate_efficiency',
    'calculate_power_factor_from_reactive',
    'calculate_reactive_power',
    
    # Equivalent circuit
    'CircuitModel',
    'CircuitParameters',
    'CircuitSolution',
    'PerformancePoint',
    'solve_circuit',
    'calculate_performance',
    'calculate_torque_speed_curve',
    'find_rated_slip',
    'calculate_breakdown_torque',
    'calculate_starting_performance',
    'refer_rotor_to_stator'
]
