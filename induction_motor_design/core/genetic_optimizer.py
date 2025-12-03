"""
Genetic Algorithm optimizer for induction motor design.

This module implements a GA-based approach to find optimal motor geometry
that satisfies nameplate requirements (power, torque, efficiency, power factor, speed).

The GA explores the design space by varying:
- Bore diameter D
- Stack length L
- Number of stator slots N_s
- Number of rotor bars N_r
- Conductors per slot n
- Target flux density Bm
- Slot geometry parameters

For each candidate design (individual), an internal convergence loop
stabilizes the electrical variables (slip, currents, losses).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random
import math
import copy

from ..models.nameplate import NameplateData
from ..models.materials import (
    ElectricalSteel,
    ConductorMaterial,
    COPPER,
    ALUMINUM,
    create_M400_50A
)
from ..models.lamination import (
    StatorSlot, StatorLamination, RotorLamination, 
    AirGap, LaminationAssembly, RotorBar, EndRing
)
from ..models.winding import WindingConfiguration, WindingDesign
from ..calculations.equivalent_circuit import (
    CircuitParameters, solve_circuit, calculate_performance
)
from ..utils.constants import DesignRanges, MU_0
from .convergence import DesignIteration


@dataclass
class DesignGenes:
    """
    Genetic encoding of motor design parameters.
    
    These are the "genes" that the GA will optimize.
    """
    # Main dimensions
    D_bore: float              # Bore diameter [m]
    L_stack: float             # Stack length [m]
    
    # Slot configuration
    N_slots: int               # Number of stator slots
    N_bars: int                # Number of rotor bars
    conductors_per_slot: int   # Conductors per slot
    
    # Electromagnetic loadings
    Bm_target: float           # Target air gap flux density [T]
    
    # Slot geometry (stator)
    slot_height_ratio: float   # h1 / D_bore (0.10 - 0.15)
    slot_width_ratio: float    # a1 / slot_pitch (0.40 - 0.55)
    tooth_width_ratio: float   # b_tooth / slot_pitch (0.45 - 0.60)
    yoke_height_ratio: float   # h_yoke / h_slot (0.70 - 0.90)
    
    # Rotor bar geometry
    bar_area_ratio: float      # A_bar / (π*D*h_slot/N_bars) (0.50 - 0.70)
    
    # Winding configuration
    short_pitch_slots: int     # Coil shortening [slots] (0 - 2)
    
    def clone(self) -> 'DesignGenes':
        """Create a deep copy."""
        return copy.deepcopy(self)
    
    def validate(self) -> bool:
        """Check if genes represent a valid design."""
        # Bore diameter: 100mm - 500mm
        if not (0.1 <= self.D_bore <= 0.5):
            return False
        
        # Stack length: 80mm - 400mm
        if not (0.08 <= self.L_stack <= 0.4):
            return False
        
        # Flux density
        if not (DesignRanges.BM_MIN <= self.Bm_target <= DesignRanges.BM_MAX):
            return False
        
        # Slots must be positive and reasonable
        if self.N_slots < 12 or self.N_slots > 144:
            return False
        
        if self.N_bars < 10 or self.N_bars > 120:
            return False
        
        if self.conductors_per_slot < 2 or self.conductors_per_slot > 50:
            return False
        
        # Ratios must be in valid ranges
        if not (0.10 <= self.slot_height_ratio <= 0.15):
            return False
        
        if not (0.40 <= self.slot_width_ratio <= 0.55):
            return False
        
        if not (0.45 <= self.tooth_width_ratio <= 0.60):
            return False
        
        if not (0.70 <= self.yoke_height_ratio <= 0.90):
            return False
        
        if not (0.50 <= self.bar_area_ratio <= 0.70):
            return False
        
        if not (0 <= self.short_pitch_slots <= 2):
            return False
        
        return True


@dataclass
class DesignFitness:
    """
    Fitness evaluation of a motor design.
    
    Lower is better (penalty-based fitness).
    """
    # Target deviations (normalized)
    power_error: float         # |P_out - P_rated| / P_rated
    torque_error: float        # |T - T_rated| / T_rated
    efficiency_error: float    # |η - η_target| / η_target
    power_factor_error: float  # |cosφ - cosφ_target| / cosφ_target
    speed_error: float         # |n - n_rated| / n_rated
    slip_error: float          # |s - s_rated| / s_rated
    
    # Constraint violations
    flux_density_penalty: float    # Penalties for B exceeding limits
    current_density_penalty: float # Penalty for J exceeding limits
    geometry_penalty: float        # Penalty for unrealistic geometry
    convergence_penalty: float     # Penalty if internal loop didn't converge
    
    # Specification shortfalls (only penalize when below target)
    efficiency_shortfall_penalty: float
    power_factor_shortfall_penalty: float
    
    # Additional metrics
    total_losses: float        # Total losses [W]
    cost_estimate: float       # Rough cost metric (copper + iron mass)
    
    @property
    def total_fitness(self) -> float:
        """
        Total fitness (lower is better).
        
        Weighted sum of errors and penalties.
        """
        # Errors on nameplate targets (high weight)
        power_term = 10.0 * self.power_error + 80.0 * (self.power_error ** 2)
        target_error = (
            power_term +
            11.0 * self.torque_error +
            14.0 * self.efficiency_error +
            12.0 * self.power_factor_error +
            8.0 * self.speed_error +
            10.0 * self.slip_error
        )
        
        spec_shortfalls = (
            24.0 * self.efficiency_shortfall_penalty +
            24.0 * self.power_factor_shortfall_penalty
        )
        
        # Constraint penalties (very high weight)
        constraints = (
            5.0 * self.flux_density_penalty +
            5.0 * self.current_density_penalty +
            0.01 * self.geometry_penalty +
            100.0 * self.convergence_penalty
        )
        
        # Soft objectives (low weight)
        soft = 0.001 * self.total_losses + 0.0001 * self.cost_estimate
        
        return target_error + spec_shortfalls + constraints + soft
    
    @property
    def is_valid(self) -> bool:
        """Check if design satisfies all hard constraints."""
        return (self.convergence_penalty == 0 and
                self.flux_density_penalty < 0.1 and
                self.current_density_penalty < 0.1 and
                self.geometry_penalty < 0.1 and 
                self.power_error < 0.1 and
                self.torque_error < 0.1 and
                self.efficiency_error < 0.1 and
                self.power_factor_error < 0.1 and
                self.speed_error < 0.1)
    
    @property
    def max_error(self) -> float:
        """Maximum error among target parameters."""
        return max(
            self.power_error,
            self.torque_error,
            self.efficiency_error,
            self.power_factor_error,
            self.speed_error,
            self.slip_error
        )


@dataclass
class Individual:
    """
    Individual in the GA population (one candidate design).
    """
    genes: DesignGenes
    fitness: Optional[DesignFitness] = None
    lamination: Optional[LaminationAssembly] = None
    winding: Optional[WindingDesign] = None
    circuit: Optional[CircuitParameters] = None
    converged: bool = False
    final_state: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other: 'Individual') -> bool:
        """For sorting by fitness."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness.total_fitness < other.fitness.total_fitness


class GeneticOptimizer:
    """
    Genetic Algorithm for induction motor design optimization.
    """
    
    def __init__(
        self,
        nameplate: NameplateData,
        steel: ElectricalSteel,
        stator_conductor: ConductorMaterial = COPPER,
        rotor_conductor: ConductorMaterial = ALUMINUM,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elitism_count: int = 5,
        random_injection_rate: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the genetic optimizer.
        
        Args:
            nameplate: Motor specifications (targets)
            steel: Electrical steel for laminations
            stator_conductor: Stator conductor material
            rotor_conductor: Rotor conductor material
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            elitism_count: Number of best individuals to preserve
            random_injection_rate: Fraction of population replaced by random
                individuals each generation (improves exploration)
            verbose: Print progress
        """
        self.nameplate = nameplate
        self.steel = steel
        self.stator_conductor = stator_conductor
        self.rotor_conductor = rotor_conductor
        
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.random_injection_rate = max(0.0, min(0.5, random_injection_rate))
        self.verbose = verbose
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []
        self.preliminary = None
    
    def _log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)
    
    def initialize_population(self):
        """Create initial random population."""
        self._log(f"\nInitializing population of {self.population_size} individuals...")
        
        # Use preliminary sizing as guidance for ranges
        from ..calculations.preliminary import calculate_preliminary_dimensions
        
        prelim = calculate_preliminary_dimensions(self.nameplate)
        self.preliminary = prelim
        
        for i in range(self.population_size):
            genes = self._create_random_genes(prelim)
            self.population.append(Individual(genes=genes))
        
        self._log(f"Population initialized with {len(self.population)} individuals")
    
    def _create_random_genes(self, prelim) -> DesignGenes:
        """Create random valid genes based on preliminary sizing."""
        p = self.nameplate.pole_pairs
        
        # Bore diameter: ±30% of preliminary
        D_min = prelim.D_bore * 0.7
        D_max = prelim.D_bore * 1.3
        D_bore = random.uniform(D_min, D_max)
        
        # Stack length: ±30% of preliminary
        L_min = prelim.L_stack * 0.7
        L_max = prelim.L_stack * 1.3
        L_stack = random.uniform(L_min, L_max)
        
        # Slots: 3-5 slots per pole per phase, must be divisible by 6 for 3-phase
        q_min, q_max = 3, 5
        q = random.uniform(q_min, q_max)
        N_slots = int(6 * p * q)
        # Round to nearest multiple of 6
        N_slots = max(12, 6 * round(N_slots / 6))
        
        # Rotor bars: typically N_slots ± (2-4)
        delta_bars = random.randint(-4, -2)
        N_bars = N_slots + delta_bars
        
        # Flux density
        Bm_target = random.uniform(
            DesignRanges.BM_MIN + 0.05,
            DesignRanges.BM_MAX - 0.05
        )
        
        # Conductors per slot: estimate from voltage
        n_estimate = prelim.D_bore * prelim.L_stack * Bm_target * 100  # Rough estimate
        n_min = max(2, int(n_estimate * 0.5))
        n_max = int(n_estimate * 2.0)
        conductors_per_slot = 2 * random.randint(n_min // 2, n_max // 2)  # Keep even
        
        # Geometry ratios with realistic ranges
        slot_height_ratio = random.uniform(0.10, 0.15)
        slot_width_ratio = random.uniform(0.40, 0.55)
        tooth_width_ratio = random.uniform(0.45, 0.60)
        yoke_height_ratio = random.uniform(0.70, 0.90)
        bar_area_ratio = random.uniform(0.50, 0.70)
        
        # Short pitching
        short_pitch_slots = random.randint(0, 2)
        
        genes = DesignGenes(
            D_bore=D_bore,
            L_stack=L_stack,
            N_slots=N_slots,
            N_bars=N_bars,
            conductors_per_slot=conductors_per_slot,
            Bm_target=Bm_target,
            slot_height_ratio=slot_height_ratio,
            slot_width_ratio=slot_width_ratio,
            tooth_width_ratio=tooth_width_ratio,
            yoke_height_ratio=yoke_height_ratio,
            bar_area_ratio=bar_area_ratio,
            short_pitch_slots=short_pitch_slots
        )
        
        # Ensure validity
        if not genes.validate():
            # Retry with safer parameters
            return self._create_random_genes(prelim)
        
        return genes
    
    def _create_random_individual(self) -> Individual:
        """Helper to instantiate a new random individual."""
        if self.preliminary is None:
            raise RuntimeError("Preliminary sizing not initialized.")
        genes = self._create_random_genes(self.preliminary)
        return Individual(genes=genes)
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals in population."""
        self._log("\nEvaluating population fitness...")
        
        for i, individual in enumerate(self.population):
            if individual.fitness is None:  # Skip if already evaluated
                individual.fitness = self._evaluate_individual(individual)
                
                if (i + 1) % 10 == 0:
                    self._log(f"  Evaluated {i + 1}/{len(self.population)} individuals")
    
    def _evaluate_individual(self, individual: Individual) -> DesignFitness:
        """
        Evaluate fitness of one individual.
        
        This involves:
        1. Building lamination from genes
        2. Running internal convergence loop
        3. Calculating performance at rated point
        4. Computing fitness based on targets
        """
        genes = individual.genes
        individual.final_state = None
        
        try:
            # Build lamination from genes
            lamination = self._build_lamination_from_genes(genes)
            individual.lamination = lamination
            
            # Design winding
            winding = self._design_winding_from_genes(genes, lamination)
            individual.winding = winding
            
            # Run internal convergence loop
            circuit, converged, final_state = self._internal_convergence_loop(
                genes, lamination, winding
            )
            individual.circuit = circuit
            individual.converged = converged
            individual.final_state = final_state
            
            # Calculate performance at rated point
            solution = solve_circuit(
                circuit,
                self.nameplate.voltage_phase,
                final_state['slip']
            )
            
            perf = calculate_performance(
                circuit, solution,
                self.nameplate.frequency,
                self.nameplate.pole_pairs,
                final_state['P_mech']
            )
            
            # Compute fitness
            fitness = self._compute_fitness(
                perf, final_state, genes, lamination, converged
            )
            
            return fitness
            
        except Exception as e:
            # If evaluation fails, return very bad fitness
            self._log(f"  Warning: Evaluation failed with error: {e}")
            return DesignFitness(
                power_error=10.0,
                torque_error=10.0,
                efficiency_error=10.0,
                power_factor_error=10.0,
                speed_error=10.0,
                slip_error=10.0,
                flux_density_penalty=10.0,
                current_density_penalty=10.0,
                geometry_penalty=10.0,
                convergence_penalty=10.0,
                efficiency_shortfall_penalty=10.0,
                power_factor_shortfall_penalty=10.0,
                total_losses=1e6,
                cost_estimate=1e6
            )
    
    def _build_lamination_from_genes(self, genes: DesignGenes) -> LaminationAssembly:
        """Build lamination assembly from gene encoding."""
        D = genes.D_bore
        L = genes.L_stack
        N_slots = genes.N_slots
        N_bars = genes.N_bars
        p = self.nameplate.pole_pairs
        
        # Slot pitch
        slot_pitch = math.pi * D / N_slots
        
        # Slot dimensions
        h_slot = genes.slot_height_ratio * D
        a1 = genes.slot_width_ratio * slot_pitch
        b_tooth = genes.tooth_width_ratio * slot_pitch
        h_yoke = genes.yoke_height_ratio * h_slot
        
        # Outer diameter
        D_outer = D + 2 * (h_slot + h_yoke)
        
        # Air gap
        delta = AirGap.empirical_minimum(D)
        delta = max(delta, 0.3e-3)
        
        # Stator slot
        stator_slot = StatorSlot(
            h1=h_slot * 0.85,
            h2=0,
            h3=1e-3,
            h4=0.7e-3,
            a1=a1,
            a_opening=min(2.5e-3, a1 * 0.4)
        )
        
        # Stator lamination
        stator = StatorLamination(
            D_bore=D,
            D_outer=D_outer,
            L_stack=L,
            N_slots=N_slots,
            slot=stator_slot,
            tooth_width=b_tooth,
            yoke_height=h_yoke
        )
        stator.set_pole_pitch(p)
        
        # Rotor bar
        bar_pitch = math.pi * (D - 2*delta) / N_bars
        bar_area = genes.bar_area_ratio * bar_pitch * h_slot
        
        rotor_bar = RotorBar(
            area=bar_area,
            height=h_slot
        )
        
        # End ring
        end_ring = EndRing(
            area=bar_area * 1.5,
            mean_diameter=D - 2*delta - h_slot
        )
        
        # Rotor lamination
        D_rotor = D - 2 * delta
        D_shaft = 0.25 * D
        
        rotor = RotorLamination(
            D_outer=D_rotor,
            D_inner=D_shaft,
            L_stack=L,
            N_bars=N_bars,
            bar=rotor_bar,
            end_ring=end_ring,
            tooth_width=b_tooth * 1.1,
            skew_slots=1
        )
        
        return LaminationAssembly(
            stator=stator,
            rotor=rotor,
            air_gap=AirGap(delta)
        )
    
    def _design_winding_from_genes(
        self, genes: DesignGenes, lamination: LaminationAssembly
    ) -> WindingDesign:
        """Design winding from genes."""
        p = self.nameplate.pole_pairs
        N_slots = genes.N_slots
        n = genes.conductors_per_slot
        
        # Winding configuration
        config = WindingConfiguration(
            N_slots=N_slots,
            pole_pairs=p,
            layers=2,
            coil_pitch_slots=int(N_slots/(2*p)) - genes.short_pitch_slots,
            conductors_per_slot=n
        )
        
        # Conductor sizing
        slot_area = lamination.stator.slot_area
        fill_factor = 0.42
        conductor_area = fill_factor * slot_area / n
        
        return WindingDesign(
            config=config,
            conductor_area=conductor_area,
            fill_factor=fill_factor
        )
    
    def _internal_convergence_loop(
        self,
        genes: DesignGenes,
        lamination: LaminationAssembly,
        winding: WindingDesign,
        max_iterations: int = 30,
        tolerance: float = 0.01
    ) -> Tuple[CircuitParameters, bool, Dict]:
        """
        Internal convergence loop for given geometry.
        
        Iterates until slip, currents, and losses stabilize.
        
        Returns:
            (circuit_params, converged, final_state)
        """
        from ..calculations.magnetic import (
            calculate_total_mmf, calculate_magnetizing_current,
            calculate_magnetizing_reactance_rigorous, calculate_Bm_from_voltage
        )
        from ..calculations.losses import (
            calculate_iron_losses, calculate_stator_copper_loss,
            calculate_rotor_bar_current, calculate_end_ring_current,
            calculate_rotor_losses, calculate_mechanical_losses,
            calculate_slip_from_losses
        )
        from ..models.winding import (
            calculate_stator_leakage_permeances,
            calculate_stator_leakage_reactance
        )
        from ..utils.constants import EMF_FACTOR_TYPICAL
        
        p = self.nameplate.pole_pairs
        f = self.nameplate.frequency
        
        # Initial guesses
        eta = self.nameplate.efficiency
        cosfi = self.nameplate.power_factor
        slip = self.nameplate.slip_rated
        
        prev_slip = slip
        prev_eta = eta
        prev_cosfi = cosfi
        
        for iteration in range(max_iterations):
            # Clamp values
            eta = max(0.5, min(0.98, eta))
            cosfi = max(0.5, min(0.95, cosfi))
            slip = max(0.005, min(0.15, slip))
            
            # Current from power balance
            eta_cosfi = eta * cosfi
            I_phase = self.nameplate.power_w / (
                math.sqrt(3) * self.nameplate.voltage_line * eta_cosfi
            )
            
            # Flux density from voltage
            Bm = calculate_Bm_from_voltage(
                voltage_phase=self.nameplate.voltage_phase,
                frequency=f,
                winding_factor=winding.config.winding_factor,
                slots_per_pole_per_phase=winding.config.q,
                conductors_per_slot=winding.config.conductors_per_slot,
                L_stack=lamination.L_stack,
                D_bore=lamination.D_bore,
                emf_factor=EMF_FACTOR_TYPICAL
            )
            
            # Stator resistance
            Rs = winding.phase_resistance(
                stack_length=lamination.L_stack,
                resistivity=self.stator_conductor.resistivity,
                D_bore=lamination.D_bore
            )
            
            # Iron losses
            iron_losses = calculate_iron_losses(
                stator=lamination.stator,
                rotor=lamination.rotor,
                Bm=Bm,
                frequency=f,
                steel=self.steel,
                pole_pairs=p
            )
            P_iron = iron_losses.total_stator
            
            # Copper losses
            P_cu_s = calculate_stator_copper_loss(I_phase, Rs)
            
            # Rotor losses
            K_skew = lamination.rotor.skew_factor(p)
            I_bar = calculate_rotor_bar_current(
                current_phase=I_phase,
                power_factor=cosfi,
                winding_factor=winding.config.winding_factor,
                total_conductors=int(winding.config.total_conductors_per_phase),
                N_bars=lamination.rotor.N_bars,
                skew_factor=K_skew
            )
            
            I_ring = calculate_end_ring_current(
                bar_current=I_bar,
                pole_pairs=p,
                N_bars=lamination.rotor.N_bars
            )
            
            P_bars, P_rings = calculate_rotor_losses(
                bar_current=I_bar,
                ring_current=I_ring,
                bar=lamination.rotor.bar,
                end_ring=lamination.rotor.end_ring,
                L_stack=lamination.L_stack,
                N_bars=lamination.rotor.N_bars,
                conductor=self.rotor_conductor
            )
            P_cu_r = P_bars + P_rings
            
            # Mechanical losses
            P_mech = calculate_mechanical_losses(
                self.nameplate.power_w,
                self.nameplate.rpm_rated
            )
            
            # Update slip
            new_slip = calculate_slip_from_losses(
                P_joule_rotor=P_cu_r,
                P_output=self.nameplate.power_w,
                P_mechanical=P_mech
            )
            new_slip = max(0.005, min(0.15, new_slip))
            
            # Update efficiency
            P_total_loss = P_cu_s + P_cu_r + P_iron + P_mech
            new_eta = self.nameplate.power_w / (self.nameplate.power_w + P_total_loss)
            
            # Magnetizing circuit
            mmf = calculate_total_mmf(
                Bm=Bm,
                stator=lamination.stator,
                rotor=lamination.rotor,
                air_gap=lamination.air_gap,
                steel=self.steel,
                pole_pairs=p
            )
            
            I_mu = calculate_magnetizing_current(
                mmf=mmf,
                winding_factor=winding.config.winding_factor,
                conductors_per_slot=winding.config.conductors_per_slot,
                slots_per_pole_per_phase=winding.config.q
            )
            
            Xm = calculate_magnetizing_reactance_rigorous(
                self.nameplate.voltage_phase,
                I_mu
            )
            
            # Leakage reactances
            permeances = calculate_stator_leakage_permeances(
                winding=winding.config,
                slot_permeance=lamination.stator.slot.slot_permeance(),
                pole_pitch=lamination.stator.pole_pitch,
                air_gap=lamination.air_gap.length
            )
            
            Xs = calculate_stator_leakage_reactance(
                winding=winding,
                permeances=permeances,
                frequency=f,
                stack_length=lamination.L_stack,
                pole_pitch=lamination.stator.pole_pitch
            )
            
            # Rotor resistance and reactance
            R_bar = self.rotor_conductor.resistance(
                lamination.L_stack,
                lamination.rotor.bar.area
            )
            
            R_ring_per_bar = P_rings / (lamination.rotor.N_bars * I_bar**2) if I_bar > 0 else 0
            R_bar_total = R_bar + R_ring_per_bar
            
            omega = 2 * math.pi * f
            lambda_slot_r = lamination.stator.slot.slot_permeance()
            X_bar = omega * lamination.L_stack * lambda_slot_r
            
            lambda_ring = MU_0 * 0.3
            X_ring = omega * math.pi * lamination.rotor.D_outer / lamination.rotor.N_bars * lambda_ring
            X_bar_total = X_bar + X_ring
            
            from ..calculations.equivalent_circuit import refer_rotor_to_stator
            Rr, Xr = refer_rotor_to_stator(
                R_rotor_actual=R_bar_total,
                X_rotor_actual=X_bar_total,
                winding_factor_stator=winding.config.winding_factor,
                conductors_per_phase_stator=int(winding.config.total_conductors_per_phase),
                N_bars=lamination.rotor.N_bars,
                skew_factor=K_skew
            )
            
            # Power factor from circuit
            circuit = CircuitParameters(Rs=Rs, Xs=Xs, Rr=Rr, Xr=Xr, Xm=Xm)
            solution = solve_circuit(circuit, self.nameplate.voltage_phase, new_slip)
            new_cosfi = solution.power_factor
            
            # Check convergence
            delta_slip = abs(new_slip - prev_slip) / prev_slip if prev_slip > 0 else 1.0
            delta_eta = abs(new_eta - prev_eta) / prev_eta if prev_eta > 0 else 1.0
            delta_cosfi = abs(new_cosfi - prev_cosfi) / prev_cosfi if prev_cosfi > 0 else 1.0
            
            if delta_slip < tolerance and delta_eta < tolerance and delta_cosfi < tolerance:
                # Converged
                final_state = {
                    'slip': new_slip,
                    'efficiency': new_eta,
                    'power_factor': new_cosfi,
                    'Bm': Bm,
                    'I_phase': I_phase,
                    'P_cu_s': P_cu_s,
                    'P_cu_r': P_cu_r,
                    'P_iron': P_iron,
                    'P_mech': P_mech
                }
                return circuit, True, final_state
            
            # Update for next iteration
            prev_slip = slip
            prev_eta = eta
            prev_cosfi = cosfi
            
            slip = new_slip
            eta = new_eta
            cosfi = new_cosfi
        
        # Did not converge
        final_state = {
            'slip': slip,
            'efficiency': eta,
            'power_factor': cosfi,
            'Bm': Bm,
            'I_phase': I_phase,
            'P_cu_s': P_cu_s,
            'P_cu_r': P_cu_r,
            'P_iron': P_iron,
            'P_mech': P_mech
        }
        return circuit, False, final_state
    
    def _compute_fitness(
        self,
        perf,
        final_state: Dict,
        genes: DesignGenes,
        lamination: LaminationAssembly,
        converged: bool
    ) -> DesignFitness:
        """Compute fitness based on performance vs targets."""
        
        # Target errors (normalized)
        power_error = abs(perf.P_output - self.nameplate.power_w) / self.nameplate.power_w
        torque_error = abs(perf.torque - self.nameplate.torque_rated) / self.nameplate.torque_rated
        efficiency_error = abs(perf.efficiency - self.nameplate.efficiency) / self.nameplate.efficiency
        power_factor_error = abs(perf.power_factor - self.nameplate.power_factor) / self.nameplate.power_factor
        speed_error = abs(perf.speed_rpm - self.nameplate.rpm_rated) / self.nameplate.rpm_rated
        slip_target = max(1e-4, self.nameplate.slip_rated)
        slip_error = abs(final_state['slip'] - slip_target) / slip_target
        
        # Shortfall penalties (only when under target)
        efficiency_shortfall_penalty = max(
            0.0,
            (self.nameplate.efficiency - perf.efficiency) / self.nameplate.efficiency
        )
        power_factor_shortfall_penalty = max(
            0.0,
            (self.nameplate.power_factor - perf.power_factor) / self.nameplate.power_factor
        )
        
        # Flux density penalties
        flux_density_penalty = 0.0
        
        B_tooth_s = lamination.stator.tooth_flux_density(final_state['Bm'])
        if B_tooth_s > DesignRanges.BD_STATOR_MAX:
            flux_density_penalty += (B_tooth_s - DesignRanges.BD_STATOR_MAX) / DesignRanges.BD_STATOR_MAX
        
        B_yoke_s = lamination.stator.yoke_flux_density(final_state['Bm'], self.nameplate.pole_pairs)
        if B_yoke_s > DesignRanges.BC_STATOR_MAX:
            flux_density_penalty += (B_yoke_s - DesignRanges.BC_STATOR_MAX) / DesignRanges.BC_STATOR_MAX
        elif B_yoke_s < DesignRanges.BC_STATOR_MIN:
            flux_density_penalty += (DesignRanges.BC_STATOR_MIN - B_yoke_s) / DesignRanges.BC_STATOR_MIN
        
        # Current density penalty
        current_density_penalty = 0.0
        J = final_state['I_phase'] / lamination.stator.slot_area * genes.conductors_per_slot
        if J > DesignRanges.J_MAX:
            current_density_penalty = (J - DesignRanges.J_MAX) / DesignRanges.J_MAX
        elif J < DesignRanges.J_MIN:
            current_density_penalty = (DesignRanges.J_MIN - J) / DesignRanges.J_MIN
        
        # Geometry penalty
        geometry_penalty = 0.0
        aspect_ratio = lamination.L_stack / lamination.stator.pole_pitch
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            geometry_penalty += 0.5
        
        # Convergence penalty
        convergence_penalty = 0.0 if converged else 1.0
        
        # Total losses
        total_losses = final_state['P_cu_s'] + final_state['P_cu_r'] + final_state['P_iron'] + final_state['P_mech']
        
        # Cost estimate (copper + iron mass in kg)
        copper_volume = genes.conductors_per_slot * lamination.stator.N_slots * lamination.stator.slot_area * lamination.L_stack
        copper_mass = copper_volume * self.stator_conductor.density
        
        iron_volume = lamination.stator.tooth_volume + lamination.stator.yoke_volume
        iron_mass = iron_volume * self.steel.density
        
        cost_estimate = copper_mass + iron_mass
        
        return DesignFitness(
            power_error=power_error,
            torque_error=torque_error,
            efficiency_error=efficiency_error,
            power_factor_error=power_factor_error,
            speed_error=speed_error,
            slip_error=slip_error,
            flux_density_penalty=flux_density_penalty,
            current_density_penalty=current_density_penalty,
            geometry_penalty=geometry_penalty,
            convergence_penalty=convergence_penalty,
            efficiency_shortfall_penalty=efficiency_shortfall_penalty,
            power_factor_shortfall_penalty=power_factor_shortfall_penalty,
            total_losses=total_losses,
            cost_estimate=cost_estimate
        )
    
    def selection(self) -> List[Individual]:
        """
        Select parents for reproduction using tournament selection.
        
        Returns:
            List of parent individuals
        """
        tournament_size = 3
        n_parents = self.population_size
        parents = []
        
        for _ in range(n_parents):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda ind: ind.fitness.total_fitness)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        
        Uses uniform crossover: each gene has 50% chance of coming from either parent.
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return Individual(genes=parent1.genes.clone()), Individual(genes=parent2.genes.clone())
        
        # Create offspring genes
        g1 = parent1.genes
        g2 = parent2.genes
        
        # Uniform crossover for continuous values
        child1_genes = DesignGenes(
            D_bore=g1.D_bore if random.random() < 0.5 else g2.D_bore,
            L_stack=g1.L_stack if random.random() < 0.5 else g2.L_stack,
            N_slots=g1.N_slots if random.random() < 0.5 else g2.N_slots,
            N_bars=g1.N_bars if random.random() < 0.5 else g2.N_bars,
            conductors_per_slot=g1.conductors_per_slot if random.random() < 0.5 else g2.conductors_per_slot,
            Bm_target=g1.Bm_target if random.random() < 0.5 else g2.Bm_target,
            slot_height_ratio=g1.slot_height_ratio if random.random() < 0.5 else g2.slot_height_ratio,
            slot_width_ratio=g1.slot_width_ratio if random.random() < 0.5 else g2.slot_width_ratio,
            tooth_width_ratio=g1.tooth_width_ratio if random.random() < 0.5 else g2.tooth_width_ratio,
            yoke_height_ratio=g1.yoke_height_ratio if random.random() < 0.5 else g2.yoke_height_ratio,
            bar_area_ratio=g1.bar_area_ratio if random.random() < 0.5 else g2.bar_area_ratio,
            short_pitch_slots=g1.short_pitch_slots if random.random() < 0.5 else g2.short_pitch_slots
        )
        
        child2_genes = DesignGenes(
            D_bore=g2.D_bore if random.random() < 0.5 else g1.D_bore,
            L_stack=g2.L_stack if random.random() < 0.5 else g1.L_stack,
            N_slots=g2.N_slots if random.random() < 0.5 else g1.N_slots,
            N_bars=g2.N_bars if random.random() < 0.5 else g1.N_bars,
            conductors_per_slot=g2.conductors_per_slot if random.random() < 0.5 else g1.conductors_per_slot,
            Bm_target=g2.Bm_target if random.random() < 0.5 else g1.Bm_target,
            slot_height_ratio=g2.slot_height_ratio if random.random() < 0.5 else g1.slot_height_ratio,
            slot_width_ratio=g2.slot_width_ratio if random.random() < 0.5 else g1.slot_width_ratio,
            tooth_width_ratio=g2.tooth_width_ratio if random.random() < 0.5 else g1.tooth_width_ratio,
            yoke_height_ratio=g2.yoke_height_ratio if random.random() < 0.5 else g1.yoke_height_ratio,
            bar_area_ratio=g2.bar_area_ratio if random.random() < 0.5 else g1.bar_area_ratio,
            short_pitch_slots=g2.short_pitch_slots if random.random() < 0.5 else g1.short_pitch_slots
        )
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual):
        """
        Mutate an individual's genes.
        
        Each gene has mutation_rate probability of being mutated.
        """
        genes = individual.genes
        
        # Mutate continuous values with Gaussian noise
        if random.random() < self.mutation_rate:
            genes.D_bore *= random.gauss(1.0, 0.1)
            genes.D_bore = max(0.1, min(0.5, genes.D_bore))
        
        if random.random() < self.mutation_rate:
            genes.L_stack *= random.gauss(1.0, 0.1)
            genes.L_stack = max(0.08, min(0.4, genes.L_stack))
        
        if random.random() < self.mutation_rate:
            genes.Bm_target += random.gauss(0, 0.05)
            genes.Bm_target = max(DesignRanges.BM_MIN, min(DesignRanges.BM_MAX, genes.Bm_target))
        
        # Mutate integer values
        if random.random() < self.mutation_rate:
            delta = random.choice([-6, 6])  # Change by one q (6 slots)
            genes.N_slots = max(12, genes.N_slots + delta)
        
        if random.random() < self.mutation_rate:
            genes.N_bars += random.choice([-2, -1, 1, 2])
            genes.N_bars = max(10, min(120, genes.N_bars))
        
        if random.random() < self.mutation_rate:
            genes.conductors_per_slot += random.choice([-2, 2])
            genes.conductors_per_slot = max(2, min(50, genes.conductors_per_slot))
        
        if random.random() < self.mutation_rate:
            genes.short_pitch_slots = random.randint(0, 2)
        
        # Mutate ratios
        if random.random() < self.mutation_rate:
            genes.slot_height_ratio += random.gauss(0, 0.01)
            genes.slot_height_ratio = max(0.10, min(0.15, genes.slot_height_ratio))
        
        if random.random() < self.mutation_rate:
            genes.slot_width_ratio += random.gauss(0, 0.02)
            genes.slot_width_ratio = max(0.40, min(0.55, genes.slot_width_ratio))
        
        if random.random() < self.mutation_rate:
            genes.tooth_width_ratio += random.gauss(0, 0.02)
            genes.tooth_width_ratio = max(0.45, min(0.60, genes.tooth_width_ratio))
        
        if random.random() < self.mutation_rate:
            genes.yoke_height_ratio += random.gauss(0, 0.02)
            genes.yoke_height_ratio = max(0.70, min(0.90, genes.yoke_height_ratio))
        
        if random.random() < self.mutation_rate:
            genes.bar_area_ratio += random.gauss(0, 0.02)
            genes.bar_area_ratio = max(0.50, min(0.70, genes.bar_area_ratio))
    
    def _inject_random_individuals(self, population: List[Individual]) -> List[Individual]:
        """Inject random individuals to improve exploration."""
        if self.random_injection_rate <= 0 or self.preliminary is None:
            return population
        
        n_inject = max(1, int(self.population_size * self.random_injection_rate))
        removable = max(0, len(population) - self.elitism_count)
        remove_count = min(removable, n_inject)
        
        for _ in range(remove_count):
            population.pop()
        
        for _ in range(n_inject):
            population.append(self._create_random_individual())
        
        random.shuffle(population)
        return population
    
    def evolve(self):
        """Run the genetic algorithm for n_generations."""
        self._log(f"\n{'='*70}")
        self._log(f"GENETIC ALGORITHM OPTIMIZATION")
        self._log(f"{'='*70}")
        self._log(f"Population size: {self.population_size}")
        self._log(f"Generations: {self.n_generations}")
        self._log(f"Mutation rate: {self.mutation_rate}")
        self._log(f"Crossover rate: {self.crossover_rate}")
        
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            self._log(f"\n--- Generation {generation + 1}/{self.n_generations} ---")
            
            # Sort population by fitness
            self.population.sort()
            
            # Track best
            best = self.population[0]
            if self.best_individual is None or best.fitness.total_fitness < self.best_individual.fitness.total_fitness:
                self.best_individual = best
            
            # Log statistics
            avg_fitness = sum(ind.fitness.total_fitness for ind in self.population) / len(self.population)
            best_fitness = best.fitness.total_fitness
            
            self._log(f"Best fitness: {best_fitness:.4f}")
            self._log(f"Avg fitness: {avg_fitness:.4f}")
            self._log(f"Best max error: {best.fitness.max_error:.2%}")
            self._log(f"Best valid: {best.fitness.is_valid}")
            
            # Store history
            self.history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_max_error': best.fitness.max_error,
                'best_valid': best.fitness.is_valid
            })
            
            # Check for good solution
            if best.fitness.is_valid and best.fitness.max_error < 0.05:
                self._log(f"\n✓ Found excellent solution at generation {generation + 1}!")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            new_population.extend(self.population[:self.elitism_count])
            
            # Generate offspring
            parents = self.selection()
            
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                self.mutate(child1)
                self.mutate(child2)
                
                # Add to new population if valid
                if child1.genes.validate():
                    new_population.append(child1)
                if len(new_population) < self.population_size and child2.genes.validate():
                    new_population.append(child2)
            
            # Inject new random individuals for exploration
            new_population = self._inject_random_individuals(new_population)
            
            # Replace population
            self.population = new_population[:self.population_size]
            
            # Evaluate new population
            self.evaluate_population()
        
        # Final summary
        self._log(f"\n{'='*70}")
        self._log(f"OPTIMIZATION COMPLETE")
        self._log(f"{'='*70}")
        self._log(f"\nBest solution found:")
        self._log(f"  Total fitness: {self.best_individual.fitness.total_fitness:.4f}")
        self._log(f"  Power error: {self.best_individual.fitness.power_error:.2%}")
        self._log(f"  Torque error: {self.best_individual.fitness.torque_error:.2%}")
        self._log(f"  Efficiency error: {self.best_individual.fitness.efficiency_error:.2%}")
        self._log(f"  Power factor error: {self.best_individual.fitness.power_factor_error:.2%}")
        self._log(f"  Speed error: {self.best_individual.fitness.speed_error:.2%}")
        self._log(f"  Valid design: {self.best_individual.fitness.is_valid}")
        
        return self.best_individual


class GradientDescentOptimizer:
    """
    Gradient-descent optimizer that minimizes the same fitness used by the GA.
    
    The gradients are estimated numerically (finite differences) directly on the
    gene vector, while all evaluations reuse the GA evaluation pipeline to keep
    consistency with the existing cost function.
    """
    
    PARAM_SPECS = [
        ('D_bore', 0.1, 0.5, 'float'),
        ('L_stack', 0.08, 0.4, 'float'),
        ('N_slots', 12, 144, 'slots'),
        ('N_bars', 10, 120, 'int'),
        ('conductors_per_slot', 2, 50, 'even_int'),
        ('Bm_target', DesignRanges.BM_MIN, DesignRanges.BM_MAX, 'float'),
        ('slot_height_ratio', 0.10, 0.15, 'float'),
        ('slot_width_ratio', 0.40, 0.55, 'float'),
        ('tooth_width_ratio', 0.45, 0.60, 'float'),
        ('yoke_height_ratio', 0.70, 0.90, 'float'),
        ('bar_area_ratio', 0.50, 0.70, 'float'),
        ('short_pitch_slots', 0, 2, 'int')
    ]
    
    def __init__(
        self,
        nameplate: NameplateData,
        steel: ElectricalSteel,
        stator_conductor: ConductorMaterial = COPPER,
        rotor_conductor: ConductorMaterial = ALUMINUM,
        step_size: float = 0.05,
        gradient_epsilon: float = 1e-3,
        max_iterations: int = 40,
        tolerance: float = 1e-4,
        verbose: bool = True
    ):
        self.nameplate = nameplate
        self.steel = steel
        self.stator_conductor = stator_conductor
        self.rotor_conductor = rotor_conductor
        self.step_size = step_size
        self.gradient_epsilon = gradient_epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        from ..calculations.preliminary import calculate_preliminary_dimensions
        self.preliminary = calculate_preliminary_dimensions(self.nameplate)
        
        # Reuse the GA evaluation pipeline for consistency
        self.evaluator = GeneticOptimizer(
            nameplate=self.nameplate,
            steel=self.steel,
            stator_conductor=self.stator_conductor,
            rotor_conductor=self.rotor_conductor,
            population_size=1,
            n_generations=1,
            verbose=False
        )
        self.evaluator.preliminary = self.preliminary
        
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _vector_from_genes(self, genes: DesignGenes) -> List[float]:
        return [
            genes.D_bore,
            genes.L_stack,
            float(genes.N_slots),
            float(genes.N_bars),
            float(genes.conductors_per_slot),
            genes.Bm_target,
            genes.slot_height_ratio,
            genes.slot_width_ratio,
            genes.tooth_width_ratio,
            genes.yoke_height_ratio,
            genes.bar_area_ratio,
            float(genes.short_pitch_slots)
        ]
    
    def _project_value(self, value: float, min_v: float, max_v: float, kind: str) -> float:
        value = max(min_v, min(max_v, value))
        if kind == 'slots':
            value = max(12, 6 * round(value / 6))
        elif kind == 'int':
            value = round(value)
        elif kind == 'even_int':
            value = int(round(value / 2) * 2)
        return value
    
    def _genes_from_vector(self, vector: List[float]) -> DesignGenes:
        values = []
        for (i, spec) in enumerate(self.PARAM_SPECS):
            name, min_v, max_v, kind = spec
            values.append(self._project_value(vector[i], min_v, max_v, kind))
        
        return DesignGenes(
            D_bore=values[0],
            L_stack=values[1],
            N_slots=int(values[2]),
            N_bars=int(values[3]),
            conductors_per_slot=int(values[4]),
            Bm_target=values[5],
            slot_height_ratio=values[6],
            slot_width_ratio=values[7],
            tooth_width_ratio=values[8],
            yoke_height_ratio=values[9],
            bar_area_ratio=values[10],
            short_pitch_slots=int(values[11])
        )
    
    def _evaluate_vector(self, vector: List[float]) -> Tuple[float, Individual, List[float]]:
        genes = self._genes_from_vector(vector)
        individual = Individual(genes=genes.clone())
        fitness = self.evaluator._evaluate_individual(individual)
        individual.fitness = fitness
        actual_vector = self._vector_from_genes(individual.genes)
        return fitness.total_fitness, individual, actual_vector
    
    def _evaluate_fitness(self, vector: List[float]) -> float:
        genes = self._genes_from_vector(vector)
        individual = Individual(genes=genes)
        fitness = self.evaluator._evaluate_individual(individual)
        return fitness.total_fitness
    
    def _gradient(self, vector: List[float], base_value: float) -> List[float]:
        grad = []
        for i, value in enumerate(vector):
            delta = self.gradient_epsilon * max(1.0, abs(value))
            pos_vec = vector.copy()
            pos_vec[i] = value + delta
            neg_vec = vector.copy()
            neg_vec[i] = value - delta
            f_pos = self._evaluate_fitness(pos_vec)
            f_neg = self._evaluate_fitness(neg_vec)
            grad.append((f_pos - f_neg) / (2 * delta))
        return grad
    
    def _project_vector(self, vector: List[float]) -> List[float]:
        projected = []
        for (value, spec) in zip(vector, self.PARAM_SPECS):
            _, min_v, max_v, kind = spec
            projected.append(self._project_value(value, min_v, max_v, kind))
        return projected
    
    def _initial_vector(self) -> List[float]:
        genes = self.evaluator._create_random_genes(self.preliminary)
        return self._vector_from_genes(genes)
    
    def optimize(self, initial_genes: Optional[DesignGenes] = None) -> Individual:
        """Run gradient descent and return the best individual found."""
        if initial_genes is None:
            vector = self._initial_vector()
        else:
            vector = self._vector_from_genes(initial_genes)
        
        best_individual: Optional[Individual] = None
        best_value = float('inf')
        step = self.step_size
        
        for iteration in range(1, self.max_iterations + 1):
            value, individual, vector = self._evaluate_vector(vector)
            
            if best_individual is None or value < best_value:
                best_individual = individual
                best_value = value
            
            grad = self._gradient(vector, value)
            grad_norm = math.sqrt(sum(g * g for g in grad))
            
            self.history.append({
                'iteration': iteration,
                'fitness': value,
                'grad_norm': grad_norm,
                'step_size': step
            })
            
            self._log(
                f"Iteration {iteration:02d}: fitness={value:.4f}, "
                f"grad_norm={grad_norm:.4e}"
            )
            
            if grad_norm < self.tolerance:
                self._log("Converged (gradient norm below tolerance).")
                break
            
            # Gradient descent update with simple backtracking
            updated = False
            step_local = step
            for _ in range(10):
                candidate = [
                    vector[i] - step_local * grad[i]
                    for i in range(len(vector))
                ]
                candidate = self._project_vector(candidate)
                cand_value = self._evaluate_fitness(candidate)
                if cand_value < value:
                    vector = candidate
                    updated = True
                    break
                step_local *= 0.5
            
            if not updated:
                self._log("Step size too small or no improvement; stopping.")
                break
        
        if best_individual is None:
            raise RuntimeError("Gradient descent failed to evaluate any individual.")
        
        self.best_individual = best_individual
        return best_individual


def optimize_motor_genetic(
    power_kw: float,
    voltage: float,
    frequency: float,
    pole_pairs: int,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    *,
    steel: Optional[ElectricalSteel] = None,
    stator_conductor: ConductorMaterial = COPPER,
    rotor_conductor: ConductorMaterial = ALUMINUM,
    verbose: bool = True,
    **optimizer_kwargs: Any
) -> Tuple[Individual, GeneticOptimizer]:
    """
    Convenience wrapper that runs the genetic optimizer end-to-end.
    
    Args:
        power_kw: Rated power [kW]
        voltage: Line voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        rpm_rated: Rated speed [rpm]
        efficiency: Target efficiency
        power_factor: Target power factor
        steel: Electrical steel (defaults to M400-50A)
        stator_conductor: Material for stator winding
        rotor_conductor: Material for the cage
        verbose: Print progress information
        **optimizer_kwargs: Extra options forwarded to ``GeneticOptimizer``
            (e.g. population_size, n_generations, mutation_rate, ...)
    
    Returns:
        Tuple with the best individual found and the configured optimizer.
    """
    from ..models.nameplate import create_nameplate_from_rpm
    
    steel = steel or create_M400_50A()
    nameplate = create_nameplate_from_rpm(
        power_kw=power_kw,
        voltage_line=voltage,
        frequency=frequency,
        pole_pairs=pole_pairs,
        rpm_rated=rpm_rated,
        efficiency=efficiency,
        power_factor=power_factor
    )
    
    optimizer = GeneticOptimizer(
        nameplate=nameplate,
        steel=steel,
        stator_conductor=stator_conductor,
        rotor_conductor=rotor_conductor,
        verbose=verbose,
        **optimizer_kwargs
    )
    best = optimizer.evolve()
    
    return best, optimizer


def optimize_motor_gradient_descent(
    power_kw: float,
    voltage: float,
    frequency: float,
    pole_pairs: int,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    *,
    steel: Optional[ElectricalSteel] = None,
    stator_conductor: ConductorMaterial = COPPER,
    rotor_conductor: ConductorMaterial = ALUMINUM,
    verbose: bool = True,
    initial_genes: Optional[DesignGenes] = None,
    **optimizer_kwargs: Any
) -> Tuple[Individual, GradientDescentOptimizer]:
    """
    Convenience wrapper for the gradient-descent optimizer.
    """
    from ..models.nameplate import create_nameplate_from_rpm
    
    steel = steel or create_M400_50A()
    nameplate = create_nameplate_from_rpm(
        power_kw=power_kw,
        voltage_line=voltage,
        frequency=frequency,
        pole_pairs=pole_pairs,
        rpm_rated=rpm_rated,
        efficiency=efficiency,
        power_factor=power_factor
    )
    
    optimizer = GradientDescentOptimizer(
        nameplate=nameplate,
        steel=steel,
        stator_conductor=stator_conductor,
        rotor_conductor=rotor_conductor,
        verbose=verbose,
        **optimizer_kwargs
    )
    best = optimizer.optimize(initial_genes=initial_genes)
    return best, optimizer


def hybrid_optimize_motor(
    power_kw: float,
    voltage: float,
    frequency: float,
    pole_pairs: int,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    *,
    steel: Optional[ElectricalSteel] = None,
    stator_conductor: ConductorMaterial = COPPER,
    rotor_conductor: ConductorMaterial = ALUMINUM,
    top_candidates: int = 5,
    ga_kwargs: Optional[Dict[str, Any]] = None,
    gd_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[Individual, Dict[str, Any]]:
    """
    Run GA followed by gradient descent on the top-N GA individuals.
    
    Returns:
        (best_individual, info_dict) where info contains GA/GD histories.
    """
    ga_kwargs = dict(ga_kwargs or {})
    gd_kwargs = dict(gd_kwargs or {})
    steel = steel or create_M400_50A()
    ga_verbose = ga_kwargs.pop('verbose', verbose)
    
    best_ga, ga_optimizer = optimize_motor_genetic(
        power_kw=power_kw,
        voltage=voltage,
        frequency=frequency,
        pole_pairs=pole_pairs,
        rpm_rated=rpm_rated,
        efficiency=efficiency,
        power_factor=power_factor,
        steel=steel,
        stator_conductor=stator_conductor,
        rotor_conductor=rotor_conductor,
        verbose=ga_verbose,
        **ga_kwargs
    )
    
    # Collect elites
    individuals = sorted(
        ga_optimizer.population,
        key=lambda ind: ind.fitness.total_fitness if ind.fitness else float('inf')
    )[:max(1, top_candidates)]
    
    gd_verbose = gd_kwargs.pop('verbose', verbose)
    gd_results = []
    for idx, ind in enumerate(individuals, 1):
        if verbose:
            print(f"\n>>> Refining GA candidate #{idx} with gradient descent")
        gd_best, gd_opt = optimize_motor_gradient_descent(
            power_kw=power_kw,
            voltage=voltage,
            frequency=frequency,
            pole_pairs=pole_pairs,
            rpm_rated=rpm_rated,
            efficiency=efficiency,
            power_factor=power_factor,
            steel=steel,
            stator_conductor=stator_conductor,
            rotor_conductor=rotor_conductor,
            verbose=gd_verbose,
            initial_genes=ind.genes.clone(),
            **gd_kwargs
        )
        gd_results.append((gd_best, gd_opt))
    
    # Choose best overall
    best_pair = min(
        [(best_ga, ga_optimizer)] + gd_results,
        key=lambda item: item[0].fitness.total_fitness if item[0].fitness else float('inf')
    )
    best_overall, best_optimizer = best_pair
    
    info = {
        'ga_optimizer': ga_optimizer,
        'gd_results': gd_results,
        'best_optimizer': best_optimizer
    }
    return best_overall, info
