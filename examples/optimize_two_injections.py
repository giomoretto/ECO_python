#!/usr/bin/env python3
"""Optimize two-injection strategy using acados OCP (economic formulation).

Find injection timing and duration (SOE, DOE) for two injections that
minimise injected fuel subject to MATLAB-standard constraints:
  - IMEP ≥ 6 bar (terminal box constraint at EVO)
  - dp/dφ ≤ 4 bar/degCA (slacked stage constraint)
  - pMax ≤ 150 bar (slacked stage constraint)
  - CoC ≥ -20 degCA (terminal constraint)
  - Phi ≤ 1/1.3 (stage constraint)
  - NOx ≤ 1e4 ppm (slacked terminal constraint)

Workflow:
  1. Simulate IVC → OCP start (pure integration, no optimisation)
  2. Optimise over [-10, EVO≈155] degCA (acados SQP with linking constraint)

Uses the shared eco/formulation module for OCP creation/initialization/solving.
"""

import sys
import os
import ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---- load acados shared libraries ----
acados_path = '/home/morettog/projects/phd/acados'
os.environ.setdefault('ACADOS_SOURCE_DIR', acados_path)
_acados_lib_dir = os.path.join(acados_path, 'lib')
for _lib in ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so', 'libacados.so']:
    _p = os.path.join(_acados_lib_dir, _lib)
    if os.path.isfile(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

from eco.model_casadi.model_parameters import ModelParameters
from eco.simulation.par_op_def import OperatingPoint
from eco.simulation.acados_simulation import acados_simulation
from eco.formulation.create_acados_ocp import create_acados_functions_inj_opt
from eco.formulation.init_acados_ocp import create_init_acados_ocp_inj_opt
from eco.formulation.run_sqp_acados import run_sqp_acados_inj_opt


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
N_INJ = 2                # number of injections
EN_NOX = True             # include NOx states (theta_uz, NO)
DELTA_PHI = 0.5           # crank-angle step [degCA]

CA_OCP_START = -10.0      # optimisation window start [degCA aTDC]
CA_OCP_END = 26.0         # optimisation window end [degCA aTDC] (NOT EVO)

IMEP_REF = 6e5            # target IMEP [Pa]  (6 bar)
DP_MAX = 4e5              # max pressure rise rate [Pa/degCA]  (4 bar/degCA, MATLAB default)

# Initial guess [SOE1, SOE2, DOE1, DOE2]
U0 = np.array([-10.0, -2.0, 200.0, 400.0])

# Control bounds  [SOE1, SOE2, DOE1, DOE2]
U_LB = np.array([-20.0, -20.0, 80.0, 80.0])
U_UB = np.array([10.0, 10.0, 800.0, 800.0])


class OptimizationParameters:
    """Lightweight container for optimization parameters matching MATLAB parOpt."""
    def __init__(self):
        self.optimization_range = [CA_OCP_START, CA_OCP_END]
        self.delta_phi = DELTA_PHI
        self.en_nox = EN_NOX

        # Build uniform crank-angle vector (no MATLAB-style tail coarsening)
        self.ca = np.arange(CA_OCP_START, CA_OCP_END + DELTA_PHI, DELTA_PHI)

        # Injection timing and duration
        self.u0 = U0
        self.u_min = U_LB
        self.u_max = U_UB

        # MATLAB-standard constraint references
        self.reference = {
            'imep': 6e5,        # target IMEP [Pa]
            'dp_max': 4e5,      # max pressure rise rate [Pa/degCA]
            'p_max': 150e5,     # max cylinder pressure [Pa]
            't_min': 500 + 273,   # min exhaust temperature [K]
            'c_nox': 1e4,       # max NOx [ppm]
            'coc_max': 20,      # latest center of combustion [degCA aTDC]
            'phi_max': 1 / 1.3, # max equivalence ratio
        }

        # Injection distancing
        self.dt_inj = 400  # [µs]

        # Variable scaling (MATLAB parOpt values)
        self.scale_offs_vars = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO',
                                'NOppm', 'SOE', 'DOE']
        self.offs = [0, 0, 0, 0, 0, 0, -20, 100]
        self.scale = [100e5, 1000, 10e5, 1e3, 1e-6, 1e3, 70, 700]

        # SQP solver settings
        self.sqp = {'n_sqp_max': 200, 'step_size': 1.0}


# ---------------------------------------------------------------------------
#  Main routine
# ---------------------------------------------------------------------------
def _hdr(txt):
    print(f'\n{"─" * 60}\n  {txt}\n{"─" * 60}')


def main():
    _hdr('ECO – Two-Injection Optimisation (economic)')
    print(f'  Minimise fuel (algebraic Qtot from injector model)')
    print(f'  IMEP constraint:  ≥ {IMEP_REF / 1e5:.1f} bar (at EVO)')
    print(f'  dp/dφ constraint: ≤ {DP_MAX / 1e5:.1f} bar/degCA (slacked)')
    print(f'  OCP window:       [{CA_OCP_START}, {CA_OCP_END}] degCA aTDC')
    print(f'  Injections:       {N_INJ} (with linking constraint u_k ≡ u0)')

    # ---- Model & operating point ----
    par_model = ModelParameters()
    par_model.n_states = 5 if EN_NOX else 3
    par_model.n_outputs = 12 if EN_NOX else 8
    par_model.n_inputs = 2 * N_INJ

    par_op = OperatingPoint(par_model)

    # ---- Optimization parameters ----
    par_opt = OptimizationParameters()

    # ---- Simulation helper ----
    class SimParams:
        pass
    par_sim = SimParams()
    par_sim.op = par_op
    par_sim.opts = {'delta_phi': DELTA_PHI, 'int': 'RK4', 'n_int': 1}

    # ===================================================================
    #  Phase 1: Simulate IVC → OCP start (pure integration)
    # ===================================================================
    ca_pre = np.arange(par_op.ca_ivc, CA_OCP_START + DELTA_PHI, DELTA_PHI)
    x0_ivc = np.array([par_op.p_int, 0.0, 0.0, par_op.theta_ivc, 0.0]) \
        if EN_NOX else np.array([par_op.p_int, 0.0, 0.0])
    u_pre = np.tile(par_opt.u0, (len(ca_pre), 1)).T

    _hdr('Phase 1: IVC → OCP start')
    sim_pre = acados_simulation(ca_pre, x0_ivc, u_pre, par_sim, par_model)
    x0_ocp = sim_pre['x'][:, -1]
    print(f'  x0 at {CA_OCP_START}°: p={x0_ocp[0] / 1e5:.1f} bar, '
          f'Qcomb={x0_ocp[1]:.1f} J, IMEP={x0_ocp[2] / 1e5:.3f} bar')

    # ===================================================================
    #  Baseline simulation over OCP range (for diagnostics)
    # ===================================================================
    ca_ocp_full = par_opt.ca
    u_base = np.tile(par_opt.u0, (len(ca_ocp_full), 1)).T
    sim_base = acados_simulation(ca_ocp_full, x0_ocp, u_base, par_sim, par_model)
    dp_base = np.diff(sim_base['x'][0, :]) / DELTA_PHI

    _hdr('Baseline (u0) over OCP window')
    print(f'  IMEP:      {sim_base["x"][2, -1] / 1e5:.2f} bar')
    print(f'  Q_comb:    {sim_base["x"][1, -1]:.1f} J')
    print(f'  Peak p:    {np.max(sim_base["x"][0, :]) / 1e5:.1f} bar')
    print(f'  Max dp/dφ: {np.max(dp_base) / 1e5:.2f} bar/degCA')

    # ===================================================================
    #  Phase 2: Optimise over [CA_OCP_START, EVO]
    # ===================================================================
    _hdr('Phase 2: Creating and solving acados OCP')

    # Create OCP with shadow-state linking constraint
    ocp_solver, fcn = create_acados_functions_inj_opt(
        par_opt, par_model, par_op,
        model_name='ECO_opt_2inj',
        code_export_subdir='c_generated_code_opt_2inj'
    )

    if ocp_solver is None:
        print('FATAL: OCP solver creation failed.')
        return None, None, None, None

    # Initialize with simulation baseline
    ocp_solver = create_init_acados_ocp_inj_opt(
        ocp_solver, fcn, par_model, par_sim, par_opt
    )

    # Solve
    u_opt, status, result = run_sqp_acados_inj_opt(
        ocp_solver, fcn, par_model, par_sim, par_opt
    )

    _status_map = {
        0: 'success', 1: 'failure', 2: 'max iter',
        3: 'min step', 4: 'QP fail',
    }
    print(f'  Status: {status} – {_status_map.get(status, "unknown")}')

    if status != 0:
        print('Optimization failed; aborting.')
        return None, None, None, None

    # Extract trajectory from result
    x_ocp = result['x']
    u_ocp = result['u']
    ca_ocp_sol = result['ca']

    # ===================================================================
    #  Phase 3: Post-integration from OCP end to EVO
    # ===================================================================
    _hdr('Phase 3: OCP end → EVO')
    ca_post = np.arange(CA_OCP_END + DELTA_PHI, par_op.ca_evo + DELTA_PHI, DELTA_PHI)
    x0_post = x_ocp[:, -1]  # state at OCP_END
    u_post = np.tile(u_opt, (len(ca_post), 1)).T  # use optimal controls

    sim_post = acados_simulation(ca_post, x0_post, u_post, par_sim, par_model)
    print(f'  Integrated {CA_OCP_END}° → {par_op.ca_evo}° (EVO)')
    print(f'  x at EVO: p={sim_post["x"][0, -1] / 1e5:.1f} bar, '
          f'Qcomb={sim_post["x"][1, -1]:.1f} J, IMEP={sim_post["x"][2, -1] / 1e5:.3f} bar')

    # ---- Combine full trajectory (Phase 1 + OCP + Phase 3) ----
    ca_full = np.concatenate([sim_pre['ca'], ca_ocp_sol[1:], sim_post['ca']])
    x_full = np.concatenate([sim_pre['x'], x_ocp[:, 1:], sim_post['x']], axis=1)
    dp_ocp = np.diff(x_ocp[0, :]) / DELTA_PHI

    # ---- Results ----
    _hdr('Results (full cycle IVC → EVO)')
    for i in range(N_INJ):
        print(f'  Inj {i + 1}:  SOE = {u_opt[i]:+7.2f} degCA,  '
              f'DOE = {u_opt[N_INJ + i]:7.1f} µs')
    print()
    print(f'  Q_comb (EVO):     {x_full[1, -1]:.1f} J  (fuel objective)')
    print(f'  IMEP (EVO):       {x_full[2, -1] / 1e5:.2f} bar  '
          f'(constraint ≥ {IMEP_REF / 1e5:.1f})')
    print(f'  Peak p:           {np.max(x_full[0, :]) / 1e5:.1f} bar  '
          f'(limit {par_opt.reference["p_max"] / 1e5:.1f})')
    print(f'  Max dp/dφ:        {np.max(dp_ocp) / 1e5:.2f} bar/degCA  '
          f'(limit {DP_MAX / 1e5:.1f})')
    if EN_NOX:
        print(f'  NOx (EVO):        {x_full[4, -1] * 1e6:.2f} ppm  '
              f'(limit {par_opt.reference["c_nox"]:.0f})')

    # ---- Check control consistency across OCP stages ----
    # With the linking constraint u_k = u_shadow_k, all stages should
    # agree to solver tolerance (< 1e-4), not just within 1.0 unit warning.
    u_spread = np.max(u_ocp, axis=1) - np.min(u_ocp, axis=1)
    if np.any(u_spread > 0.01):
        print('\n  ⚠ Controls vary across stages (spread per channel):')
        labels = [f'SOE_{i+1}' for i in range(N_INJ)] + \
                 [f'DOE_{i+1}' for i in range(N_INJ)]
        for i, (lbl, sp) in enumerate(zip(labels, u_spread)):
            print(f'    {lbl}: {sp:.4f}')
    else:
        print('\n  ✓ Linking constraint held: all stages agree on u (spread < 0.01)')

    return u_opt, x_full, ca_full, u_ocp


if __name__ == '__main__':
    main()
