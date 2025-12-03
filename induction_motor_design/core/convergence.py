"""
Convergence criteria and iteration control for design process.
"""

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class ConvergenceState:
    """
    State of a single design variable being tracked for convergence.
    
    Attributes:
        name: Variable name (e.g., 'efficiency', 'power_factor')
        current_value: Current iteration value
        previous_value: Previous iteration value
        target_value: Target value (if applicable)
        tolerance: Convergence tolerance (relative)
    """
    name: str
    current_value: float
    previous_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.01  # 1% default
    
    @property
    def relative_change(self) -> float:
        """Relative change from previous iteration."""
        if self.previous_value is None or self.previous_value == 0:
            return float('inf')
        return abs(self.current_value - self.previous_value) / abs(self.previous_value)
    
    @property
    def is_converged(self) -> bool:
        """Check if variable has converged."""
        return self.relative_change < self.tolerance
    
    @property
    def error_from_target(self) -> Optional[float]:
        """Relative error from target value."""
        if self.target_value is None or self.target_value == 0:
            return None
        return abs(self.current_value - self.target_value) / abs(self.target_value)
    
    def update(self, new_value: float):
        """Update with new iteration value."""
        self.previous_value = self.current_value
        self.current_value = new_value


@dataclass
class ConvergenceTracker:
    """
    Tracks convergence of multiple design variables.
    
    The design loop converges when ALL tracked variables have converged.
    """
    variables: dict  # name -> ConvergenceState
    max_iterations: int = 50
    current_iteration: int = 0
    history: List[dict] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def add_variable(
        self,
        name: str,
        initial_value: float,
        target_value: Optional[float] = None,
        tolerance: float = 0.01
    ):
        """Add a variable to track."""
        self.variables[name] = ConvergenceState(
            name=name,
            current_value=initial_value,
            target_value=target_value,
            tolerance=tolerance
        )
    
    def update(self, **values):
        """
        Update tracked variables with new values.
        
        Args:
            **values: Variable names and their new values
        """
        self.current_iteration += 1
        
        iteration_record = {'iteration': self.current_iteration}
        
        for name, value in values.items():
            if name in self.variables:
                self.variables[name].update(value)
                iteration_record[name] = value
        
        self.history.append(iteration_record)
    
    @property
    def is_converged(self) -> bool:
        """Check if all variables have converged."""
        if self.current_iteration < 2:
            return False
        return all(v.is_converged for v in self.variables.values())
    
    @property
    def max_iterations_reached(self) -> bool:
        """Check if maximum iterations exceeded."""
        return self.current_iteration >= self.max_iterations
    
    @property
    def should_continue(self) -> bool:
        """Check if iteration should continue."""
        return not self.is_converged and not self.max_iterations_reached
    
    def get_summary(self) -> dict:
        """Get convergence summary."""
        return {
            'iteration': self.current_iteration,
            'converged': self.is_converged,
            'variables': {
                name: {
                    'value': state.current_value,
                    'change': state.relative_change,
                    'converged': state.is_converged,
                    'target_error': state.error_from_target
                }
                for name, state in self.variables.items()
            }
        }
    
    def __str__(self) -> str:
        lines = [f"Iteration {self.current_iteration}:"]
        for name, state in self.variables.items():
            status = "✓" if state.is_converged else "○"
            lines.append(
                f"  {status} {name}: {state.current_value:.4f} "
                f"(Δ={state.relative_change*100:.2f}%)"
            )
        return "\n".join(lines)


def create_standard_tracker(
    initial_eta: float,
    initial_cosfi: float,
    target_eta: float,
    target_cosfi: float,
    max_iterations: int = 50,
    tolerance: float = 0.005
) -> ConvergenceTracker:
    """
    Create a standard convergence tracker for motor design.
    
    Tracks:
    - Efficiency η
    - Power factor cos(φ)
    - Slip s
    
    Args:
        initial_eta: Initial guess for efficiency
        initial_cosfi: Initial guess for power factor
        target_eta: Target efficiency (from nameplate)
        target_cosfi: Target power factor (from nameplate)
        max_iterations: Maximum allowed iterations
        tolerance: Convergence tolerance (relative)
    
    Returns:
        Configured ConvergenceTracker
    """
    tracker = ConvergenceTracker(
        variables={},
        max_iterations=max_iterations
    )
    
    tracker.add_variable(
        'efficiency',
        initial_value=initial_eta,
        target_value=target_eta,
        tolerance=tolerance
    )
    
    tracker.add_variable(
        'power_factor',
        initial_value=initial_cosfi,
        target_value=target_cosfi,
        tolerance=tolerance
    )
    
    tracker.add_variable(
        'slip',
        initial_value=0.03,  # Typical initial guess
        tolerance=tolerance
    )
    
    return tracker


@dataclass
class DesignIteration:
    """
    Record of a single design iteration.
    
    Contains all computed values at that iteration.
    """
    iteration: int
    
    # Primary design variables
    Bm: float              # Air gap flux density [T]
    J: float               # Current density [A/m²]
    delta: float           # Linear current density [A/m]
    
    # Currents
    I_phase: float         # Phase current [A]
    I_magnetizing: float   # Magnetizing current [A]
    
    # Losses
    P_cu_stator: float     # Stator copper losses [W]
    P_cu_rotor: float      # Rotor copper losses [W]
    P_iron: float          # Iron losses [W]
    P_mech: float          # Mechanical losses [W]
    
    # Performance
    efficiency: float
    power_factor: float
    slip: float
    
    # Circuit parameters
    Rs: float              # Stator resistance [Ω]
    Rr: float              # Rotor resistance (referred) [Ω]
    Xs: float              # Stator reactance [Ω]
    Xr: float              # Rotor reactance (referred) [Ω]
    Xm: float              # Magnetizing reactance [Ω]


class DesignHistory:
    """
    Stores complete history of design iterations.
    """
    
    def __init__(self):
        self.iterations: List[DesignIteration] = []
    
    def add(self, iteration: DesignIteration):
        """Add an iteration record."""
        self.iterations.append(iteration)
    
    @property
    def current(self) -> Optional[DesignIteration]:
        """Get most recent iteration."""
        return self.iterations[-1] if self.iterations else None
    
    @property
    def count(self) -> int:
        """Number of iterations."""
        return len(self.iterations)
    
    def get_convergence_data(self, variable: str) -> List[float]:
        """
        Get convergence history for a specific variable.
        
        Args:
            variable: Name of the variable (must be an attribute of DesignIteration)
        
        Returns:
            List of values across iterations
        """
        return [getattr(it, variable) for it in self.iterations]
    
    def plot_convergence(self, variable: str):
        """
        Plot convergence of a variable (requires matplotlib).
        """
        try:
            import matplotlib.pyplot as plt
            
            values = self.get_convergence_data(variable)
            iterations = list(range(1, len(values) + 1))
            
            plt.figure(figsize=(8, 4))
            plt.plot(iterations, values, 'b-o')
            plt.xlabel('Iteration')
            plt.ylabel(variable)
            plt.title(f'Convergence of {variable}')
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
