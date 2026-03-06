#!/usr/bin/env python3
"""Example: Injection optimization using acados

This script demonstrates how to use the ECO package for injection optimization
with a fully symbolic CasADi model solved via acados.
"""

import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

import numpy as np
from eco.model_casadi.model_parameters import ModelParameters
from eco.simulation.par_op_def import OperatingPoint
from eco.simulation.complete_simulation import complete_simulation
from eco.formulation.scale_unscale import scale_unscale
from eco.formulation.create_acados_ocp import create_acados_functions_inj_opt
from eco.formulation.run_sqp_acados import run_sqp_acados_inj_opt


class OptimizationParameters:
    """Container for optimization parameters"""
    def __init__(self):
        # Optimization range
        self.optimization_range = [-16, 26]  # [deg CA]
        self.delta_phi = 0.5  # Step size [deg CA]
        self.ca = np.arange(self.optimization_range[0],
                           self.optimization_range[1] + self.delta_phi,
                           self.delta_phi)

        # Enable NOx calculation
        self.en_nox = True

        # Initial guess for control variables [SOE1, SOE2, DOE1, DOE2]
        self.u0 = np.array([-10, -5, 180, 400])

        # Time between injections [s]
        self.dt_inj = 1e-3

        # Bounds on states
        self.x_min = np.array([0, 0, -2e6, 273, 0])  # [Pa, J, J, K, mol/mol]
        self.x_max = np.array([2e7, 5e3, 5e3, 1e4, 0.1])

        # Bounds on inputs
        self.u_min = np.array([-20, -20, 80, 80])  # [deg CA]
        self.u_max = np.array([10, 10, 800, 800])

        # Reference values for cost function
        self.reference = {
            'imep': 1e6,  # Target IMEP [J]
            't_min': 1200,  # Min temperature for NOx [K]
            'p_max': 1.8e7,  # Max cylinder pressure [Pa]
            'dp_max': 8e5,  # Max pressure derivative [Pa/deg]
            'c_nox': 5e-6,  # Target NOx concentration [mol/mol]
            'phi_max': 60,  # Latest start of combustion [deg CA]
            'coc_max': 20  # Latest center of combustion [deg CA]
        }

        # Cost function weights
        self.weights = {
            'imep': 1e6,
            't_min': 1e3,
            'p_max': 1e5,
            'dp_max': 1e4,
            'nox': 1e8,
            'phi_max': 1e2
        }

        # Scaling configuration for scale_unscale
        # Variable names, scale factors, and offsets
        self.scale_offs_vars = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO',
                                'NOppm', 'SOE', 'DOE']
        self.scale_vals = {
            'pCyl': 1e7,
            'QComb': 1e3,
            'IMEP': 1e5,
            'Theta': 1e3,
            'NO': 1e-6,
            'NOppm': 1e3,
            'SOE': 70,
            'DOE': 700
        }
        # scale_unscale expects .scale (list) and .offs (list) aligned with scale_offs_vars
        self.scale = [self.scale_vals[v] for v in self.scale_offs_vars]
        self.offs = [0.0] * len(self.scale_offs_vars)

        # SQP solver settings (used by create_acados_ocp)
        self.sqp = {
            'n_sqp_max': 50,
            'step_size': 1.0,
        }


def main():
    """Main example script"""
    print("=" * 70)
    print("ECO - Engine Combustion Optimization Example")
    print("=" * 70)
    print()

    # Initialize model parameters
    print("Initializing model parameters...")
    par_model = ModelParameters()
    par_model.n_inputs = 4   # 2 injections, SOE + DOE each
    par_model.n_states = 5   # with NOx: [pCyl, QComb, IMEP, ThetaUZ, NO]
    par_model.n_outputs = 12  # complete_model returns 12 outputs with NOx

    # Define operating point via measurement dict
    print("Setting up operating point...")
    meas = {
        'ne': 1500 / 60,        # Engine speed [1/s]  (1500 rpm)
        'p_im': 1.5e5,          # Intake manifold pressure [Pa]
        't_im': 313,            # Intake temperature [K]
        'x_bg': 0.03,           # Burnt gas fraction at intake [-]
        'p_rail': 1600e5,       # Rail pressure [Pa]
    }
    par_op = OperatingPoint(par_model, meas=meas)

    # Simulation parameters (keys must match complete_simulation expectations)
    class SimulationParameters:
        def __init__(self, op):
            self.op = op
            self.opts = {
                'delta_phi': 0.5,
                'int': 'RK4',   # integration method
                'n_int': 1      # sub-steps per crank angle step
            }

    par_sim = SimulationParameters(par_op)

    # Optimization parameters
    print("Setting up optimization parameters...")
    par_opt = OptimizationParameters()

    # Run initial simulation with u0
    print("\nRunning initial simulation...")
    ca_range = np.arange(par_opt.optimization_range[0],
                        par_opt.optimization_range[1] + par_opt.delta_phi,
                        par_opt.delta_phi)

    x0 = np.array([par_op.p_int, 0, 0, par_op.theta_ivc, 0])
    u0_sim = np.tile(par_opt.u0, (len(ca_range), 1)).T

    try:
        sim_init = complete_simulation(ca_range, x0, u0_sim, par_sim, par_model)

        print(f"  Initial IMEP: {sim_init['x'][2, -1]:.2f} J")
        print(f"  Max pressure: {np.max(sim_init['x'][0, :]) / 1e5:.2f} bar")
        if par_opt.en_nox:
            print(f"  Final NOx: {sim_init['x'][4, -1] * 1e6:.2f} ppm")
    except Exception as e:
        print(f"  Warning: Initial simulation failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing with optimization setup...")

    # Demonstrate scaling/unscaling
    print("\nDemonstrating variable scaling...")
    u_names = ['SOE', 'SOE', 'DOE', 'DOE']
    u_scaled, u_scale, u_offs = scale_unscale(par_opt.u0, u_names, par_opt, 'scale', False)
    u_unscaled, _, _ = scale_unscale(u_scaled, u_names, par_opt, 'unscale', False)

    print(f"  Original:  {par_opt.u0}")
    print(f"  Scaled:    {u_scaled}")
    print(f"  Unscaled:  {u_unscaled}")
    print(f"  Match: {np.allclose(par_opt.u0, u_unscaled)}")

    # ---- Create acados OCP
    print("\nCreating acados OCP (code generation)...")
    ocp_solver, fcn = create_acados_functions_inj_opt(par_opt, par_model, par_op)

    if ocp_solver is None:
        print("OCP solver creation failed. Exiting.")
        return

    # ---- Run optimization
    print("\nRunning optimization...")
    opt_vars, status, result = run_sqp_acados_inj_opt(
        ocp_solver, fcn, par_model, par_sim, par_opt)

    print(f"\nOptimal controls (physical units):")
    n_inj = par_model.n_inputs // 2
    for i in range(n_inj):
        print(f"  Injection {i+1}: SOE = {opt_vars[i]:.2f} degCA, "
              f"DOE = {opt_vars[n_inj + i]:.1f} \u00b5s")

    if result:
        print(f"\nFinal states:")
        print(f"  IMEP:         {result['x'][2, -1]:.2f} Pa")
        print(f"  Max pressure: {np.max(result['x'][0, :]) / 1e5:.2f} bar")
        if par_opt.en_nox:
            print(f"  Final NOx:    {result['x'][4, -1] * 1e6:.2f} ppm")

    print("\nExample completed!")


if __name__ == "__main__":
    main()
