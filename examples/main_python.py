#!/usr/bin/env python3
"""main_python.py – Pareto-optimal injection inputs.

Python equivalent of ECO/05_example/MATLAB/mainMATLAB.m

Workflow:
  1. Build and solve unconstrained OCP (T_min=273K, cNOx=1e4 → no active constraint)
  2. Pareto sweep over T_min (exhaust gas temperature)
  3. Pareto sweep over cNOx (NOx concentration)
  4. Plot results
"""

import sys
import os
import ctypes
import numpy as np

# ---- paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_DIR)

# ---- load acados shared libraries ----
ACADOS_PATH = '/home/morettog/projects/phd/acados'
os.environ.setdefault('ACADOS_SOURCE_DIR', ACADOS_PATH)
_acados_lib_dir = os.path.join(ACADOS_PATH, 'lib')
for _lib in ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so', 'libacados.so']:
    _p = os.path.join(_acados_lib_dir, _lib)
    if os.path.isfile(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

from eco.model_casadi.model_parameters import ModelParameters
from eco.simulation.par_op_def import OperatingPoint
from eco.formulation.create_acados_ocp import create_acados_functions_inj_opt
from eco.formulation.run_sqp_acados import run_sqp_acados_inj_opt


# ---------------------------------------------------------------------------
#  Parameter definitions (matching MATLAB parOpt / parModel / parSim)
# ---------------------------------------------------------------------------
class OptimizationParameters:
    """Optimization parameters matching MATLAB init_ocp.m"""
    def __init__(self):
        # References
        self.reference = {
            'imep': 6e5,         # Target IMEP [Pa]
            'p_max': 150e5,      # Max cylinder pressure [Pa]
            'dp_max': 4e5,       # Max pressure rise rate [Pa/degCA]
            't_min': 0 + 273,    # Min exhaust gas temperature [K]
            'c_nox': 1e4,        # Max NOx concentration [ppm]
            'coc_max': 20,       # Latest center of combustion [degCA]
            'phi_max': 1 / 1.3,  # Max fuel-air equivalence ratio [-]
        }

        # Discretization
        self.delta_phi = 0.5
        self.delta_phi_acados = 3.0
        self.optimization_range = [-16, 26]

        # Solver
        self.sqp = {'n_sqp_max': 100, 'step_size': 1.0}
        self.en_nox = True

        # Initial guess [SOE1, SOE2, DOE1, DOE2]
        self.u0 = np.array([-10.3, -5.3, 167.0, 372.0])
        n_inj = len(self.u0) // 2

        # Control bounds
        self.u_min = np.array([self.optimization_range[0]] * n_inj +
                              [80.0] * n_inj)
        self.u_max = np.array([10.0] * n_inj + [800.0] * n_inj)

        # State bounds [pCyl, QComb, IMEP]
        self.x_min = np.array([0, 0, -2e6])
        self.x_max = np.array([150e5, 2e3, 2e6])

        # Injection distancing
        self.dt_inj = 400  # [µs]

        # Variable scaling
        self.scale_offs_vars = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO',
                                'NOppm', 'SOE', 'DOE']
        self.offs = [0, 0, 0, 0, 0, 0, -20, 100]
        self.scale = [100e5, 1000, 10e5, 1e3, 1e-6, 1e3, 70, 700]

        # Crank angle vector (built in extend_to_evo)
        self.ca = np.arange(self.optimization_range[0],
                            self.optimization_range[1] + self.delta_phi,
                            self.delta_phi)



class SimulationParameters:
    """Simulation parameters matching MATLAB parSim"""
    def __init__(self, par_op):
        self.op = par_op
        self.opts = {
            'delta_phi': 0.5,
            'int': 'RK4',
            'n_int': 1,
        }
        # Full crank angle vector for post-simulation
        self.ca_full = np.arange(par_op.ca_ivc, par_op.ca_evo + 0.5, 0.5)


# ---------------------------------------------------------------------------
#  Pareto sweep
# ---------------------------------------------------------------------------
def pareto_sweep(ocp_solver, fcn, par_opt, par_model, par_sim,
                 ref_key, ref_values, name='', unit=''):
    """Run Pareto sweep: vary one reference, re-solve OCP for each value."""
    from eco.simulation.complete_simulation import complete_simulation

    n_ref = len(ref_values)
    u_opt_all = np.zeros((par_model.n_inputs, n_ref))
    eta = np.zeros(n_ref)
    y_out = np.zeros(n_ref)
    status_all = np.zeros(n_ref, dtype=int)

    # Full-cycle simulation grid (IVC -> EVO) at fine resolution, matching
    # parSim.Opts.ca / x0Sim used by MATLAB pareto_Tmin.m and pareto_cNOx.m.
    ca_sim = np.arange(par_sim.op.ca_ivc,
                       par_sim.op.ca_evo + par_opt.delta_phi,
                       par_opt.delta_phi)
    x0_sim = np.array([par_sim.op.p_int, 0.0, 0.0])
    if par_opt.en_nox:
        x0_sim = np.append(x0_sim, [par_sim.op.theta_ivc, 0.0])

    # Snapshot the swept reference so it can be restored afterwards. In MATLAB
    # parOpt is passed by value, so each sweep varies ONE reference and leaves
    # the others at their defaults; in Python par_opt is shared by reference, so
    # without this the last swept value would leak into the next sweep.
    orig_ref = par_opt.reference[ref_key]

    for k, ref_val in enumerate(ref_values):
        print(f"\n  [{name}] Solving for ref = {ref_val:.1f} ...")
        par_opt.reference[ref_key] = ref_val

        u_opt, status, _result = run_sqp_acados_inj_opt(
            ocp_solver, fcn, par_model, par_sim, par_opt)
        u_opt_all[:, k] = u_opt
        status_all[k] = status

        # Evaluate the optimal inputs with a fresh full-cycle simulation
        # (IVC -> EVO) at fine resolution, exactly like MATLAB pareto_*.m.
        # (The OCP already integrates to EVO, but the benchmark re-simulates
        # to read off IMEP, Tevo and NOx.)
        u_sim = np.tile(u_opt, (len(ca_sim), 1)).T
        sim_full = complete_simulation(ca_sim, x0_sim, u_sim, par_sim, par_model)

        imep = sim_full['x'][2, -1]     # IMEP at EVO
        m_fuel = sim_full['y'][4, -1]   # mFuelPrep at EVO
        vol_dis = par_model.eng['vol_dis']
        eta[k] = 100 * imep * vol_dis / (m_fuel * 42.6e6) if m_fuel > 0 else 0

        if ref_key == 't_min':
            y_out[k] = sim_full['y'][0, -1] - 273  # T_evo in °C
        else:
            y_out[k] = sim_full['y'][8, -1]        # NOppm

        print(f"    u_opt = {u_opt}")
        print(f"    eta = {eta[k]:.2f} %, y = {y_out[k]:.1f}")

    # Restore the swept reference to its default so it does not leak into the
    # next sweep (matches MATLAB's pass-by-value parOpt semantics).
    par_opt.reference[ref_key] = orig_ref

    return {
        'name': name,
        'unit': unit,
        'u_opt': u_opt_all,
        'y': y_out,
        'eta': eta,
        'status': status_all,
        'ref_values': ref_values,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_results(data_list):
    """Plot Pareto front results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(1, len(data_list), figsize=(12, 5))
    if len(data_list) == 1:
        axes = [axes]
    for ax, data in zip(axes, data_list):
        ax.plot(data['y'], data['eta'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(f"{data['name']} {data['unit']}")
        ax.set_ylabel(r'$\eta_{\rm ind}$ [%]')
        ax.set_title(f'Pareto: {data["name"]}')
        ax.grid(True)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'pareto_results.png')
    plt.savefig(out_path, dpi=150)
    print(f"  Saved plot to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  ECO – Pareto-Optimal Injection Inputs (Python)")
    print("=" * 70)

    # ---- Model parameters ----
    par_model = ModelParameters()
    par_model.n_inputs = 4   # 2 injections × (SOE + DOE)
    par_model.n_states = 5   # with NOx: pCyl, QComb, IMEP, ThetaUZ, NO
    par_model.n_outputs = 12

    # ---- Operating point (default) ----
    par_op = OperatingPoint(par_model)

    # ---- Simulation & optimization parameters ----
    par_sim = SimulationParameters(par_op)
    par_opt = OptimizationParameters()

    # Include NOx states in bounds
    if par_opt.en_nox:
        par_opt.x_min = np.append(par_opt.x_min, [0, 0])
        par_opt.x_max = np.append(par_opt.x_max, [1e4, 0.1])

    # ---- Extend optimization grid to EVO (matching init_ocp.m:76-81) ----
    # The OCP horizon runs at fine resolution (delta_phi) over the
    # optimization range, then continues with coarse (delta_phi_acados) steps
    # up to exhaust-valve opening (EVO). This places the terminal node — where
    # IMEP, Tevo and NOx are constrained — at EVO, exactly like the MATLAB
    # benchmark, instead of at the end of the optimization range (26°).
    ca_acados = np.arange(
        par_opt.ca[-1] + par_opt.delta_phi_acados,
        par_op.ca_evo + par_opt.delta_phi_acados * 0.5,
        par_opt.delta_phi_acados)
    par_opt.ca = np.concatenate([par_opt.ca, ca_acados])
    print(f"    OCP horizon: {par_opt.ca[0]:.1f}° -> {par_opt.ca[-1]:.1f}° "
          f"({len(par_opt.ca)} nodes; terminal at EVO)")

    # ===================================================================
    #  Phase 1: Pre-integration (IVC → OCP start) for initial conditions
    # ===================================================================
    print("\n[1] Pre-integration: IVC → OCP start")
    ca_ocp_start = par_opt.optimization_range[0]
    ca_pre = np.arange(par_op.ca_ivc, ca_ocp_start + par_opt.delta_phi, par_opt.delta_phi)
    x0_ivc = np.array([par_op.p_int, 0.0, 0.0])
    if par_opt.en_nox:
        x0_ivc = np.append(x0_ivc, [par_op.theta_ivc, 0.0])
    u_pre = np.tile(par_opt.u0, (len(ca_pre), 1)).T

    from eco.simulation.acados_simulation import acados_simulation
    sim_pre = acados_simulation(ca_pre, x0_ivc, u_pre, par_sim, par_model)
    x0_ocp = sim_pre['x'][:, -1]
    print(f"    x0 at {ca_ocp_start}°: p={x0_ocp[0] / 1e5:.1f} bar")

    # Store pre-simulation for later use
    par_sim.sim_pre = sim_pre
    par_sim.x0_ocp = x0_ocp

    # ---- Build OCP ----
    print("\n[2] Creating acados OCP (code generation)...")
    ocp_solver, fcn = create_acados_functions_inj_opt(par_opt, par_model, par_op)
    if ocp_solver is None:
        print("FATAL: OCP solver creation failed.")
        return

    # ---- Solve unconstrained (baseline) ----
    print("\n[3] Solving unconstrained OCP...")
    u_opt_unc, status_unc, result_unc = run_sqp_acados_inj_opt(
        ocp_solver, fcn, par_model, par_sim, par_opt)
    print(f"    Unconstrained u_opt = {u_opt_unc}")

    # ---- Pareto: T_min ----
    print("\n[4] Pareto sweep: minimum exhaust temperature (T_min)")
    t_min_values = np.array([0, 480, 510, 540]) + 273.0
    data_tmin = pareto_sweep(ocp_solver, fcn, par_opt, par_model, par_sim,
                             ref_key='t_min', ref_values=t_min_values,
                             name=r'$T_{\rm evo,min}$', unit='[°C]')

    # ---- Pareto: cNOx ----
    print("\n[5] Pareto sweep: maximum NOx concentration (cNOx)")
    c_nox_values = np.array([1e4, 1550, 1200, 900])
    data_cnox = pareto_sweep(ocp_solver, fcn, par_opt, par_model, par_sim,
                             ref_key='c_nox', ref_values=c_nox_values,
                             name=r'$X_{\rm NO,max}$', unit='[ppm]')

    # ---- Plot ----
    print("\n[6] Plotting results...")
    plot_results([data_tmin, data_cnox])

    print("\nDone.")


if __name__ == '__main__':
    main()
