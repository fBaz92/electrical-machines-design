"""Core design engine and convergence tracking."""

from .convergence import (
    ConvergenceState,
    ConvergenceTracker,
    create_standard_tracker,
    DesignIteration,
    DesignHistory
)

from .design_engine import (
    DesignInputs,
    DesignOutputs,
    InductionMotorDesigner,
    design_motor
)

__all__ = [
    # Convergence
    'ConvergenceState',
    'ConvergenceTracker',
    'create_standard_tracker',
    'DesignIteration',
    'DesignHistory',
    
    # Design engine
    'DesignInputs',
    'DesignOutputs',
    'InductionMotorDesigner',
    'design_motor'
]
