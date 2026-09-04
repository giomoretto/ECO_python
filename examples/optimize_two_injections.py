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
import numpy as np

# ---- make the eco package importable no matter where this is run from ----
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# ---- load acados (location from ACADOS_SOURCE_DIR, see eco/acados_env.py) ----
from eco.acados_env import load_acados
load_acados()

from eco.model_casadi.model_parameters import ModelParameters
from eco.simulation.par_op_def import OperatingPoint
from eco.simulation.acados_simulation import acados_simulation
from eco.formulation.create_acados_ocp import create_acados_functions_inj_opt
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
            'imep': 8e5,        # target IMEP [Pa]
            'dp_max': 4e5,      # max pressure rise rate [Pa/degCA]
            'p_max': 150e5,     # max cylinder pressure [Pa]
            't_min': 700 + 273,   # min exhaust temperature [K]
            'c_nox': 2e3,       # max NOx [ppm]
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
    print(f'  Injections:       {N_INJ} (inputs constant over the horizon)')

    # ---- Model & operating point ----
    par_model = ModelParameters()
    par_model.n_states = 5 if EN_NOX else 3
    par_model.n_outputs = 12 if EN_NOX else 8
    par_model.n_inputs = 2 * N_INJ

    par_op = OperatingPoint(par_model)

    # ---- Optimization parameters ----
    par_opt = OptimizationParameters()

    # Extend the OCP grid past the optimisation window to EVO with coarse
    # steps (matching init_ocp.m), so the terminal constraints (IMEP, Tevo,
    # NOx) are enforced at exhaust-valve opening. Without this, IMEP >= 6 bar
    # is imposed mid-expansion (at 26° IMEP is still negative) and the QP is
    # infeasible.
    delta_phi_acados = 3.0
    ca_acados = np.arange(par_opt.ca[-1] + delta_phi_acados,
                          par_op.ca_evo + delta_phi_acados * 0.5,
                          delta_phi_acados)
    par_opt.ca = np.concatenate([par_opt.ca, ca_acados])
    print(f'  OCP horizon: {par_opt.ca[0]:.1f}° -> {par_opt.ca[-1]:.1f}° '
          f'({len(par_opt.ca)} nodes; terminal at EVO)')

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

    _status_map = {
        0: 'success', 1: 'failure', 2: 'max iter',
        3: 'min step', 4: 'QP fail',
    }

    # Warm-start homotopy: aggressive constraints can be infeasible from a cold
    # start, so ramp the tightened references from loose (inactive) values to
    # their targets, warm-starting each solve from the previous one. The init
    # sets the initial trajectory only once and never resets, so acados keeps
    # the previous solution as its guess (mirrors main_python's Pareto sweeps).
    targets = {k: par_opt.reference[k] for k in ('imep', 't_min', 'c_nox')}
    loose = {
        'imep':  min(6e5, targets['imep']),       # IMEP >= : start <= 6 bar
        't_min': min(0 + 273, targets['t_min']),  # Tevo >= : start inactive
        'c_nox': max(1e4, targets['c_nox']),      # NOx  <= : start inactive
    }
    n_homotopy = 6

    _hdr('Phase 2: homotopy solve (loose -> target constraints)')
    last_good = None
    for step in range(n_homotopy + 1):
        a = step / n_homotopy
        for key in targets:
            par_opt.reference[key] = loose[key] + a * (targets[key] - loose[key])
        u_opt, status, result = run_sqp_acados_inj_opt(
            ocp_solver, fcn, par_model, par_sim, par_opt)
        print(f'  {a * 100:5.1f}%  IMEP>={par_opt.reference["imep"] / 1e5:.1f}bar  '
              f'Tevo>={par_opt.reference["t_min"] - 273:.0f}C  '
              f'NOx<={par_opt.reference["c_nox"]:.0f}ppm  '
              f'-> status {status} ({_status_map.get(status, "?")})')
        if status == 0:
            last_good = (u_opt, result, dict(par_opt.reference))
        else:
            print(f'\n  Homotopy stalled at {a * 100:.0f}% toward the target; the '
                  f'remaining tightening looks infeasible for this operating '
                  f'point. Reporting the tightest feasible solution reached.')
            break

    if last_good is None:
        print('Optimization failed even at the loosest constraints; aborting.')
        return None, None, None, None
    u_opt, result, achieved = last_good
    par_opt.reference.update(achieved)   # report the constraints actually met

    # Optimal injection inputs at every OCP node (constant: zero-dynamics states)
    u_ocp = result['u']

    # ===================================================================
    #  Phase 3: fresh full-cycle simulation (IVC → EVO) with optimal u
    # ===================================================================
    # The OCP already integrates to EVO, so re-simulate the whole cycle at
    # fine resolution with the optimal injection inputs to read off the
    # constrained quantities (matching MATLAB pareto_*.m).
    _hdr('Phase 3: full-cycle simulation (IVC → EVO)')
    ca_full = np.arange(par_op.ca_ivc, par_op.ca_evo + DELTA_PHI, DELTA_PHI)
    x0_full = np.array([par_op.p_int, 0.0, 0.0, par_op.theta_ivc, 0.0]) \
        if EN_NOX else np.array([par_op.p_int, 0.0, 0.0])
    u_full = np.tile(u_opt, (len(ca_full), 1)).T
    sim_full = acados_simulation(ca_full, x0_full, u_full, par_sim, par_model)
    x_full = sim_full['x']
    y_full = sim_full['y']
    dp_full = np.diff(x_full[0, :]) / DELTA_PHI
    print(f'  Simulated {par_op.ca_ivc}° → {par_op.ca_evo}° (EVO)')
    print(f'  x at EVO: p={x_full[0, -1] / 1e5:.1f} bar, '
          f'Qcomb={x_full[1, -1]:.1f} J, IMEP={x_full[2, -1] / 1e5:.3f} bar')

    # ---- Results ----
    _hdr('Results (full cycle IVC → EVO)')
    for i in range(N_INJ):
        print(f'  Inj {i + 1}:  SOE = {u_opt[i]:+7.2f} degCA,  '
              f'DOE = {u_opt[N_INJ + i]:7.1f} µs')
    print()
    print(f'  Q_comb (EVO):     {x_full[1, -1]:.1f} J  (fuel objective)')
    print(f'  IMEP (EVO):       {x_full[2, -1] / 1e5:.2f} bar  '
          f'(constraint ≥ {par_opt.reference["imep"] / 1e5:.1f})')
    print(f'  Peak p:           {np.max(x_full[0, :]) / 1e5:.1f} bar  '
          f'(limit {par_opt.reference["p_max"] / 1e5:.1f})')
    print(f'  Max dp/dφ:        {np.max(dp_full) / 1e5:.2f} bar/degCA  '
          f'(limit {DP_MAX / 1e5:.1f})')
    print(f'  Tevo (EVO):       {y_full[0, -1] - 273:.1f} °C  '
          f'(constraint ≥ {par_opt.reference["t_min"] - 273:.0f})')
    if EN_NOX:
        print(f'  NOx (EVO):        {y_full[8, -1]:.1f} ppm  '
              f'(limit {par_opt.reference["c_nox"]:.0f})')

    # ---- Check the injection inputs are constant across the horizon ----
    # They are zero-dynamics states, so every node must agree to solver tol.
    u_spread = np.max(u_ocp, axis=1) - np.min(u_ocp, axis=1)
    if np.any(u_spread > 0.01):
        print('\n  ⚠ Inputs vary across stages (spread per channel):')
        labels = [f'SOE_{i+1}' for i in range(N_INJ)] + \
                 [f'DOE_{i+1}' for i in range(N_INJ)]
        for lbl, sp in zip(labels, u_spread):
            print(f'    {lbl}: {sp:.4f}')
    else:
        print('\n  ✓ Injection inputs constant across the horizon (spread < 0.01)')

    return u_opt, x_full, ca_full, u_ocp


if __name__ == '__main__':
    main()
