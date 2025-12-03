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

from .genetic_optimizer import (
    GeneticOptimizer,
    GradientDescentOptimizer,
    DesignGenes,
    DesignFitness,
    Individual,
    optimize_motor_genetic,
    optimize_motor_gradient_descent,
    hybrid_optimize_motor
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
    'design_motor',
    
    # Genetic optimizer
    'GeneticOptimizer',
    'GradientDescentOptimizer',
    'DesignGenes',
    'DesignFitness',
    'Individual',
    'optimize_motor_genetic',
    'optimize_motor_gradient_descent',
    'hybrid_optimize_motor'
]
