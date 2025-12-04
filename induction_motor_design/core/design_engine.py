"""
Main design engine for induction motor iterative design.

This module orchestrates the complete design process:
1. Preliminary sizing from nameplate data
2. Iterative refinement until convergence
3. Final verification and output
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math

from ..models.nameplate import NameplateData
from ..models.materials import (
    ElectricalSteel, ConductorMaterial, 
    COPPER, ALUMINUM, create_M400_50A
)
from ..models.lamination import (
    StatorLamination, RotorLamination, AirGap, LaminationAssembly,
    StatorSlot, RotorBar, EndRing
)
from ..models.winding import (
    WindingConfiguration, WindingDesign,
    calculate_conductors_per_slot, round_conductors,
    calculate_stator_leakage_permeances, calculate_stator_leakage_reactance
)
from ..calculations.preliminary import (
    calculate_preliminary_dimensions, PreliminarySizing
)
from ..calculations.magnetic import (
    calculate_total_mmf, calculate_magnetizing_current,
    calculate_magnetizing_reactance_rigorous, calculate_Bm_from_voltage,
    verify_flux_densities
)
from ..calculations.losses import (
    calculate_iron_losses, calculate_stator_copper_loss,
    calculate_rotor_bar_current, calculate_end_ring_current,
    calculate_rotor_losses, calculate_mechanical_losses,
    calculate_slip_from_losses, calculate_efficiency,
    TotalLosses, IronLosses, CopperLosses, ResistanceFe
)
from ..calculations.equivalent_circuit import (
    CircuitParameters, solve_circuit, calculate_performance,
    calculate_torque_speed_curve, PerformancePoint, refer_rotor_to_stator
)
from .convergence import (
    ConvergenceTracker, create_standard_tracker,
    DesignIteration, DesignHistory
)
from ..utils.constants import DesignRanges, EMF_FACTOR_TYPICAL, MU_0
from .thermal_model import (
    MotorLosses,
    evaluate_design_thermal_performance,
    ThermalModelResult,
    InsulationClass
)


@dataclass
class DesignInputs:
    """
    All inputs needed for motor design.
    
    Attributes:
        nameplate: Motor specifications
        steel: Electrical steel for laminations
        stator_conductor: Conductor material for stator
        rotor_conductor: Conductor material for rotor (cage)
        lamination: Pre-defined lamination assembly (optional)
    """
    nameplate: NameplateData
    steel: ElectricalSteel = field(default_factory=create_M400_50A)
    stator_conductor: ConductorMaterial = field(default_factory=lambda: COPPER)
    rotor_conductor: ConductorMaterial = field(default_factory=lambda: ALUMINUM)
    lamination: Optional[LaminationAssembly] = None
    
    # Design choices (can be overridden)
    Bm_target: float = DesignRanges.BM_TYPICAL
    J_target: float = DesignRanges.J_TYPICAL
    fill_factor: float = DesignRanges.FILL_FACTOR_TYPICAL
    short_pitch_slots: int = 1  # Slots of coil shortening
    
    # Thermal model options
    enable_thermal_model: bool = False
    insulation_class: InsulationClass = InsulationClass.F
    thermal_ambient: float = 40.0
    cooling_type: str = "TEFC"


@dataclass
class DesignOutputs:
    """
    Complete design outputs.
    """
    # Geometry
    lamination: LaminationAssembly
    winding: WindingDesign
    
    # Electromagnetic
    Bm: float                  # Final air gap flux density [T]
    flux_densities: dict       # All flux densities
    
    # Circuit parameters
    circuit: CircuitParameters
    
    # Performance at rated point
    rated_performance: PerformancePoint
    
    # Losses breakdown
    losses: TotalLosses
    
    # Convergence info
    iterations: int
    converged: bool
    history: DesignHistory
    
    # Thermal analysis (optional)
    thermal_result: Optional[ThermalModelResult] = None


class InductionMotorDesigner:
    """
    Main design engine for induction motors.
    
    Usage:
        designer = InductionMotorDesigner(inputs)
        outputs = designer.run()
    """
    
    def __init__(
        self,
        inputs: DesignInputs,
        max_iterations: int = 50,
        tolerance: float = 0.005,
        verbose: bool = True
    ):
        """
        Initialize the designer.
        
        Args:
            inputs: Design inputs
            max_iterations: Maximum design iterations
            tolerance: Convergence tolerance
            verbose: Print progress messages
        """
        self.inputs = inputs
        self.nameplate = inputs.nameplate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Will be populated during design
        self.preliminary: Optional[PreliminarySizing] = None
        self.lamination: Optional[LaminationAssembly] = None
        self.winding: Optional[WindingDesign] = None
        self.history = DesignHistory()
        
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def run(self) -> DesignOutputs:
        """
        Run the complete design process.
        
        Returns:
            DesignOutputs with final design
        """
        self._log("=" * 60)
        self._log("INDUCTION MOTOR DESIGN")
        self._log("=" * 60)
        self._log(f"\nTarget specifications:")
        self._log(f"  Power: {self.nameplate.power_kw} kW")
        self._log(f"  Voltage: {self.nameplate.voltage_line} V")
        self._log(f"  Speed: {self.nameplate.rpm_rated} rpm")
        self._log(f"  Efficiency target: {self.nameplate.efficiency:.1%}")
        self._log(f"  Power factor target: {self.nameplate.power_factor:.2f}")
        
        # Step 1: Preliminary sizing
        self._log("\n--- Step 1: Preliminary Sizing ---")
        self._preliminary_sizing()
        
        # Step 2: Create or use lamination
        self._log("\n--- Step 2: Lamination Setup ---")
        self._setup_lamination()
        
        # Step 3: Design winding
        self._log("\n--- Step 3: Winding Design ---")
        self._design_winding()
        
        # Step 4: Iterative design loop
        self._log("\n--- Step 4: Iterative Design ---")
        converged = self._design_loop()
        
        # Step 5: Final calculations and output
        self._log("\n--- Step 5: Final Results ---")
        outputs = self._compile_outputs(converged)
        
        return outputs
    
    def _preliminary_sizing(self):
        """Calculate preliminary dimensions."""
        self.preliminary = calculate_preliminary_dimensions(
            self.nameplate,
            Ka=0.92,  # Initial winding factor estimate
            Bm=self.inputs.Bm_target,
            delta=DesignRanges.DELTA_TYPICAL
        )
        
        self._log(f"  Preliminary bore diameter: {self.preliminary.D_bore*1000:.1f} mm")
        self._log(f"  Preliminary stack length: {self.preliminary.L_stack*1000:.1f} mm")
        self._log(f"  Pole pitch: {self.preliminary.pole_pitch*1000:.1f} mm")
    
    def _setup_lamination(self):
        """Set up lamination geometry."""
        if self.inputs.lamination is not None:
            self.lamination = self.inputs.lamination
            self._log("  Using provided lamination geometry")
        else:
            # Create default lamination based on preliminary sizing
            self.lamination = self._create_default_lamination()
            self._log("  Created default lamination geometry")
        
        self._log(f"  Bore diameter: {self.lamination.D_bore*1000:.1f} mm")
        self._log(f"  Stack length: {self.lamination.L_stack*1000:.1f} mm")
        self._log(f"  Stator slots: {self.lamination.stator.N_slots}")
        self._log(f"  Rotor bars: {self.lamination.rotor.N_bars}")
    
    def _create_default_lamination(self) -> LaminationAssembly:
        """Create default lamination from preliminary sizing."""
        p = self.nameplate.pole_pairs
        D = self.preliminary.D_bore
        L = self.preliminary.L_stack
        
        # Typical slot numbers
        N_slots = 6 * p * 3  # 3 slots per pole per phase, integer
        N_bars = N_slots - 2  # Typical: slightly fewer bars than slots
        
        # Estimate dimensions
        slot_pitch = math.pi * D / N_slots
        tooth_width = 0.5 * slot_pitch  # ~50% tooth
        slot_width = 0.4 * slot_pitch
        slot_height = 0.12 * D  # ~12% of bore diameter
        yoke_height = slot_height * 0.8
        D_outer = D + 2 * (slot_height + yoke_height)
        
        # Air gap (empirical formula)
        delta = AirGap.empirical_minimum(D)
        delta = max(delta, 0.3e-3)  # Minimum 0.3mm
        
        # Stator slot
        stator_slot = StatorSlot(
            h1=slot_height * 0.85,
            h2=0,
            h3=1e-3,
            h4=0.7e-3,
            a1=slot_width,
            a_opening=2.5e-3
        )
        
        # Rotor bar
        bar_area = (math.pi * D / N_bars) * slot_height * 0.6  # Approximate
        rotor_bar = RotorBar(
            area=bar_area,
            height=slot_height
        )
        
        # End ring
        end_ring = EndRing(
            area=bar_area * 1.5,
            mean_diameter=D - 2 * slot_height
        )
        
        # Stator lamination
        stator = StatorLamination(
            D_bore=D,
            D_outer=D_outer,
            L_stack=L,
            N_slots=N_slots,
            slot=stator_slot,
            tooth_width=tooth_width,
            yoke_height=yoke_height
        )
        stator.set_pole_pitch(p)
        
        # Rotor lamination
        D_rotor = D - 2 * delta
        D_shaft = 0.25 * D  # Approximate shaft diameter
        rotor = RotorLamination(
            D_outer=D_rotor,
            D_inner=D_shaft,
            L_stack=L,
            N_bars=N_bars,
            bar=rotor_bar,
            end_ring=end_ring,
            tooth_width=tooth_width * 1.1,  # Rotor teeth slightly wider
            skew_slots=1  # Typical skew
        )
        
        return LaminationAssembly(
            stator=stator,
            rotor=rotor,
            air_gap=AirGap(delta)
        )
    
    def _design_winding(self):
        """Design the stator winding."""
        p = self.nameplate.pole_pairs
        N_slots = self.lamination.stator.N_slots
        
        # Winding configuration
        config = WindingConfiguration(
            N_slots=N_slots,
            pole_pairs=p,
            layers=2,  # Double layer
            coil_pitch_slots=int(N_slots/(2*p)) - self.inputs.short_pitch_slots
        )
        
        # Calculate conductors per slot
        n_calc = calculate_conductors_per_slot(
            voltage_phase=self.nameplate.voltage_phase,
            frequency=self.nameplate.frequency,
            winding_factor=config.winding_factor,
            q=config.q,
            Bm=self.inputs.Bm_target,
            stack_length=self.lamination.L_stack,
            D_bore=self.lamination.D_bore,
            emf_factor=EMF_FACTOR_TYPICAL
        )
        
        n = round_conductors(n_calc, double_layer=True)
        config.conductors_per_slot = n
        
        self._log(f"  Slots per pole per phase q: {config.q}")
        self._log(f"  Winding factor: {config.winding_factor:.4f}")
        self._log(f"  Conductors per slot: {n} (calculated: {n_calc:.1f})")
        
        # Conductor sizing
        slot_area = self.lamination.stator.slot_area
        conductor_area = self.inputs.fill_factor * slot_area / n
        
        self.winding = WindingDesign(
            config=config,
            conductor_area=conductor_area,
            fill_factor=self.inputs.fill_factor
        )
        
        self._log(f"  Conductor area: {conductor_area*1e6:.2f} mm²")
    
    def _design_loop(self) -> bool:
        """
        Main iterative design loop.
        
        Returns:
            True if converged, False otherwise
        """
        # Initialize convergence tracker
        tracker = create_standard_tracker(
            initial_eta=self.nameplate.efficiency,
            initial_cosfi=self.nameplate.power_factor,
            target_eta=self.nameplate.efficiency,
            target_cosfi=self.nameplate.power_factor,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance
        )
        
        # Initial values
        eta = self.nameplate.efficiency
        cosfi = self.nameplate.power_factor
        slip = self.nameplate.slip_rated
        
        iteration = 0
        while tracker.should_continue:
            iteration += 1
            
            # Calculate at this iteration
            result = self._calculate_iteration(eta, cosfi, slip)
            
            # Update tracker
            tracker.update(
                efficiency=result['efficiency'],
                power_factor=result['power_factor'],
                slip=result['slip']
            )
            
            # Store iteration
            self.history.add(DesignIteration(
                iteration=iteration,
                Bm=result['Bm'],
                J=result['J'],
                delta=result['delta'],
                I_phase=result['I_phase'],
                I_magnetizing=result['I_magnetizing'],
                P_cu_stator=result['P_cu_stator'],
                P_cu_rotor=result['P_cu_rotor'],
                P_iron=result['P_iron'],
                P_mech=result['P_mech'],
                efficiency=result['efficiency'],
                power_factor=result['power_factor'],
                slip=result['slip'],
                Rs=result['Rs'],
                Rr=result['Rr'],
                Xs=result['Xs'],
                Xr=result['Xr'],
                Xm=result['Xm']
            ))
            
            # Update for next iteration
            eta = result['efficiency']
            cosfi = result['power_factor']
            slip = result['slip']
            
            if self.verbose and iteration % 5 == 0:
                self._log(f"  Iteration {iteration}: η={eta:.4f}, cosφ={cosfi:.4f}, s={slip:.4f}")
        
        self._log(f"\n  Converged: {tracker.is_converged} after {iteration} iterations")
        return tracker.is_converged
    
    def _calculate_iteration(
        self,
        eta: float,
        cosfi: float,
        slip: float
    ) -> dict:
        """
        Perform one design iteration.
        
        Args:
            eta: Current efficiency estimate
            cosfi: Current power factor estimate
            slip: Current slip estimate
        
        Returns:
            Dictionary with all calculated values
        """
        p = self.nameplate.pole_pairs
        f = self.nameplate.frequency
        
        # Clamp values to reasonable ranges
        eta = max(0.5, min(0.98, eta))
        cosfi = max(0.5, min(0.95, cosfi))
        slip = max(0.005, min(0.15, slip))
        
        # Update current estimate
        eta_cosfi = eta * cosfi
        I_line = self.nameplate.power_w / (math.sqrt(3) * self.nameplate.voltage_line * eta_cosfi)
        I_phase = I_line  # Y connection
        
        # Recalculate Bm from voltage equation
        Bm = calculate_Bm_from_voltage(
            voltage_phase=self.nameplate.voltage_phase,
            frequency=f,
            winding_factor=self.winding.config.winding_factor,
            slots_per_pole_per_phase=self.winding.config.q,
            conductors_per_slot=self.winding.config.conductors_per_slot,
            L_stack=self.lamination.L_stack,
            D_bore=self.lamination.D_bore,
            emf_factor=EMF_FACTOR_TYPICAL
        )
        
        # Current density
        J = I_phase / self.winding.total_conductor_area
        
        # Linear current density
        N = self.winding.config.total_conductors_per_phase
        delta = 3 * N * I_phase / (math.pi * self.lamination.D_bore)
        
        # Stator resistance
        Rs = self.winding.phase_resistance(
            stack_length=self.lamination.L_stack,
            resistivity=self.inputs.stator_conductor.resistivity,
            D_bore=self.lamination.D_bore
        )
        
        # Iron losses
        iron_losses = calculate_iron_losses(
            stator=self.lamination.stator,
            rotor=self.lamination.rotor,
            Bm=Bm,
            frequency=f,
            steel=self.inputs.steel,
            pole_pairs=p
        )
        P_iron = iron_losses.total_stator
        
        # Stator copper losses
        P_cu_s = calculate_stator_copper_loss(I_phase, Rs)
        
        # Rotor currents and losses
        K_skew = self.lamination.rotor.skew_factor(p)
        I_bar = calculate_rotor_bar_current(
            current_phase=I_phase,
            power_factor=cosfi,
            winding_factor=self.winding.config.winding_factor,
            total_conductors=int(N),
            N_bars=self.lamination.rotor.N_bars,
            skew_factor=K_skew
        )
        
        I_ring = calculate_end_ring_current(
            bar_current=I_bar,
            pole_pairs=p,
            N_bars=self.lamination.rotor.N_bars
        )
        
        P_bars, P_rings = calculate_rotor_losses(
            bar_current=I_bar,
            ring_current=I_ring,
            bar=self.lamination.rotor.bar,
            end_ring=self.lamination.rotor.end_ring,
            L_stack=self.lamination.L_stack,
            N_bars=self.lamination.rotor.N_bars,
            conductor=self.inputs.rotor_conductor
        )
        P_cu_r = P_bars + P_rings
        
        # Mechanical losses
        P_mech = calculate_mechanical_losses(
            self.nameplate.power_w,
            self.nameplate.rpm_rated
        )
        
        # New slip from rotor losses
        new_slip = calculate_slip_from_losses(
            P_joule_rotor=P_cu_r,
            P_output=self.nameplate.power_w,
            P_mechanical=P_mech
        )
        # Clamp slip to reasonable range
        new_slip = max(0.005, min(0.15, new_slip))
        
        # New efficiency
        P_total_loss = P_cu_s + P_cu_r + P_iron + P_mech
        new_eta = self.nameplate.power_w / (self.nameplate.power_w + P_total_loss)
        
        # Magnetizing circuit
        mmf = calculate_total_mmf(
            Bm=Bm,
            stator=self.lamination.stator,
            rotor=self.lamination.rotor,
            air_gap=self.lamination.air_gap,
            steel=self.inputs.steel,
            pole_pairs=p
        )
        
        I_mu = calculate_magnetizing_current(
            mmf=mmf,
            winding_factor=self.winding.config.winding_factor,
            conductors_per_slot=self.winding.config.conductors_per_slot,
            slots_per_pole_per_phase=self.winding.config.q
        )
        
        Xm = calculate_magnetizing_reactance_rigorous(
            self.nameplate.voltage_phase,
            I_mu
        )
        
        # Leakage reactances
        permeances = calculate_stator_leakage_permeances(
            winding=self.winding.config,
            slot_permeance=self.lamination.stator.slot.slot_permeance(),
            pole_pitch=self.lamination.stator.pole_pitch,
            air_gap=self.lamination.air_gap.length
        )
        
        Xs = calculate_stator_leakage_reactance(
            winding=self.winding,
            permeances=permeances,
            frequency=f,
            stack_length=self.lamination.L_stack,
            pole_pitch=self.lamination.stator.pole_pitch
        )
        
        # Rotor resistance and reactance (referred)
        R_bar = self.inputs.rotor_conductor.resistance(
            self.lamination.L_stack,
            self.lamination.rotor.bar.area
        )
        
        # Add end ring contribution to bar resistance
        R_ring_per_bar = P_rings / (self.lamination.rotor.N_bars * I_bar**2) if I_bar > 0 else 0
        R_bar_total = R_bar + R_ring_per_bar
        
        # Rotor leakage reactance (slot + end ring)
        # Simplified estimate based on rotor slot geometry
        omega = 2 * math.pi * f
        lambda_slot_r = self.lamination.stator.slot.slot_permeance()  # Similar to stator
        X_bar = omega * self.lamination.L_stack * lambda_slot_r
        
        # End ring reactance contribution
        lambda_ring = MU_0 * 0.3  # Empirical
        X_ring = omega * math.pi * self.lamination.rotor.D_outer / self.lamination.rotor.N_bars * lambda_ring
        
        X_bar_total = X_bar + X_ring
        
        Rr, Xr = refer_rotor_to_stator(
            R_rotor_actual=R_bar_total,
            X_rotor_actual=X_bar_total,
            winding_factor_stator=self.winding.config.winding_factor,
            conductors_per_phase_stator=int(N),
            N_bars=self.lamination.rotor.N_bars,
            skew_factor=K_skew
        )
        
        # Power factor from circuit
        circuit = CircuitParameters(Rs=Rs, Xs=Xs, Rr=Rr, Xr=Xr, Xm=Xm)
        solution = solve_circuit(circuit, self.nameplate.voltage_phase, new_slip)
        new_cosfi = solution.power_factor
        
        return {
            'Bm': Bm,
            'J': J,
            'delta': delta,
            'I_phase': I_phase,
            'I_magnetizing': I_mu,
            'P_cu_stator': P_cu_s,
            'P_cu_rotor': P_cu_r,
            'P_iron': P_iron,
            'P_mech': P_mech,
            'efficiency': new_eta,
            'power_factor': new_cosfi,
            'slip': new_slip,
            'Rs': Rs,
            'Rr': Rr,
            'Xs': Xs,
            'Xr': Xr,
            'Xm': Xm
        }
    
    def _compile_outputs(self, converged: bool) -> DesignOutputs:
        """Compile final design outputs."""
        final = self.history.current
        
        # Circuit parameters
        circuit = CircuitParameters(
            Rs=final.Rs,
            Xs=final.Xs,
            Rr=final.Rr,
            Xr=final.Xr,
            Xm=final.Xm,
            Rfe=self.nameplate.voltage_phase**2 / final.P_iron if final.P_iron > 0 else float('inf')
        )
        
        # Performance at rated point
        solution = solve_circuit(circuit, self.nameplate.voltage_phase, final.slip)
        rated_perf = calculate_performance(
            circuit, solution,
            self.nameplate.frequency,
            self.nameplate.pole_pairs,
            final.P_mech
        )
        
        # Losses
        iron = IronLosses(
            P_teeth_stator=final.P_iron * 0.6,  # Approximate split
            P_yoke_stator=final.P_iron * 0.4,
            P_teeth_rotor=0,
            P_yoke_rotor=0
        )
        copper = CopperLosses(
            P_stator=final.P_cu_stator,
            P_rotor_bars=final.P_cu_rotor * 0.8,
            P_rotor_rings=final.P_cu_rotor * 0.2
        )
        losses = TotalLosses(iron=iron, copper=copper, mechanical=final.P_mech)
        
        # Flux densities
        flux_densities = verify_flux_densities(
            Bm=final.Bm,
            stator=self.lamination.stator,
            rotor=self.lamination.rotor,
            pole_pairs=self.nameplate.pole_pairs
        )
        
        # Print summary
        self._log(f"\n{'='*60}")
        self._log("DESIGN RESULTS")
        self._log(f"{'='*60}")
        self._log(f"\nGeometry:")
        self._log(f"  Bore diameter: {self.lamination.D_bore*1000:.1f} mm")
        self._log(f"  Stack length: {self.lamination.L_stack*1000:.1f} mm")
        self._log(f"  Air gap: {self.lamination.air_gap.length*1000:.2f} mm")
        
        self._log(f"\nElectromagnetic:")
        self._log(f"  Air gap flux density Bm: {final.Bm:.3f} T")
        self._log(f"  Stator tooth Bd: {flux_densities['B_tooth_stator']:.3f} T")
        self._log(f"  Stator yoke Bc: {flux_densities['B_yoke_stator']:.3f} T")
        
        self._log(f"\nCircuit Parameters:")
        self._log(f"  Rs: {final.Rs:.4f} Ω")
        self._log(f"  Xs: {final.Xs:.4f} Ω")
        self._log(f"  Rr: {final.Rr:.4f} Ω")
        self._log(f"  Xr: {final.Xr:.4f} Ω")
        self._log(f"  Xm: {final.Xm:.2f} Ω")
        
        self._log(f"\nPerformance:")
        self._log(f"  Efficiency: {final.efficiency:.1%}")
        self._log(f"  Power factor: {final.power_factor:.3f}")
        self._log(f"  Slip: {final.slip:.4f} ({final.slip*100:.2f}%)")
        self._log(f"  Speed: {self.nameplate.rpm_sync*(1-final.slip):.0f} rpm")
        
        self._log(f"\nLosses:")
        self._log(f"  Stator copper: {final.P_cu_stator:.1f} W")
        self._log(f"  Rotor copper: {final.P_cu_rotor:.1f} W")
        self._log(f"  Iron: {final.P_iron:.1f} W")
        self._log(f"  Mechanical: {final.P_mech:.1f} W")
        self._log(f"  Total: {losses.total:.1f} W")
        
        thermal_result = None
        if self.inputs.enable_thermal_model:
            motor_losses = MotorLosses(
                P_cu_stator=final.P_cu_stator,
                P_cu_rotor=final.P_cu_rotor,
                P_iron=final.P_iron,
                P_mechanical=final.P_mech
            )
            rpm = self.nameplate.rpm_sync * (1 - final.slip)
            thermal_result = evaluate_design_thermal_performance(
                lamination=self.lamination,
                winding=self.winding,
                stator_conductor=self.inputs.stator_conductor,
                rotor_conductor=self.inputs.rotor_conductor,
                steel=self.inputs.steel,
                losses=motor_losses,
                rpm=rpm,
                insulation_class=self.inputs.insulation_class,
                ambient_temp=self.inputs.thermal_ambient,
                cooling_type=self.inputs.cooling_type
            )
            self._log(f"\nThermal:")
            self._log(
                f"  Winding temperature: {thermal_result.T_winding:.1f} °C "
                f"(limit {thermal_result.insulation_class.max_temp:.0f} °C)"
            )
            self._log(
                f"  Frame temperature: {thermal_result.T_frame:.1f} °C, "
                f"margin: {thermal_result.temperature_margin:.1f} K"
            )
        
        return DesignOutputs(
            lamination=self.lamination,
            winding=self.winding,
            Bm=final.Bm,
            flux_densities=flux_densities,
            circuit=circuit,
            rated_performance=rated_perf,
            losses=losses,
            iterations=self.history.count,
            converged=converged,
            history=self.history,
            thermal_result=thermal_result
        )


def design_motor(
    power_kw: float,
    voltage: float,
    frequency: float,
    pole_pairs: int,
    rpm_rated: float,
    efficiency: float,
    power_factor: float,
    verbose: bool = True,
    **kwargs
) -> DesignOutputs:
    """
    Convenience function for quick motor design.
    
    Args:
        power_kw: Rated power [kW]
        voltage: Line voltage [V]
        frequency: Supply frequency [Hz]
        pole_pairs: Number of pole pairs
        rpm_rated: Rated speed [rpm]
        efficiency: Target efficiency
        power_factor: Target power factor
        verbose: Print progress messages
        **kwargs: Additional options passed to DesignInputs
    
    Returns:
        DesignOutputs
    """
    from ..models.nameplate import create_nameplate_from_rpm
    
    nameplate = create_nameplate_from_rpm(
        power_kw=power_kw,
        voltage_line=voltage,
        frequency=frequency,
        pole_pairs=pole_pairs,
        rpm_rated=rpm_rated,
        efficiency=efficiency,
        power_factor=power_factor
    )
    
    inputs = DesignInputs(nameplate=nameplate, **kwargs)
    designer = InductionMotorDesigner(inputs, verbose=verbose)
    
    return designer.run()
