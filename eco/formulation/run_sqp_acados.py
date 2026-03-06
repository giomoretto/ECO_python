"""Run SQP solver with acados

This module handles running the SQP optimization and extracting results.
"""

import numpy as np
from typing import Tuple, Any, Dict
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')
from eco.formulation.init_acados_ocp import create_init_acados_ocp_inj_opt
from eco.formulation.scale_unscale import scale_unscale


def run_sqp_acados_inj_opt(ocp: Any, fcn: Dict,
                           par_model: Any, par_sim: Any,
                           par_opt: Any) -> Tuple[np.ndarray, int, Dict]:
    """Run SQP optimization with acados

    Args:
        ocp: Acados OCP solver object
        fcn: Dictionary of helper functions / scaling info
        par_model: Model parameters
        par_sim: Simulation parameters
        par_opt: Optimization parameters

    Returns:
        opt_vars: Optimal decision variables (unscaled, physical units)
        status: Solver status code (0 = success)
        result: Dictionary with full trajectories (unscaled)
    """
    if ocp is None:
        print("Error: OCP solver not available")
        return par_opt.u0, -1, {}

    # ---- initialise
    ocp = create_init_acados_ocp_inj_opt(ocp, fcn, par_model, par_sim, par_opt)

    # ---- solve
    try:
        status = ocp.solve()
    except Exception as e:
        print(f"Error during optimisation: {e}")
        import traceback
        traceback.print_exc()
        return par_opt.u0, -1, {}

    N = len(par_opt.ca) - 1
    n_states = par_model.n_states
    nx = fcn['nx']
    nu = fcn['nu']

    # ---- extract scaled solution
    x_traj_s = np.zeros((nx, N + 1))
    u_traj_s = np.zeros((nu, N))
    for i in range(N + 1):
        x_traj_s[:, i] = ocp.get(i, 'x')
    for i in range(N):
        u_traj_s[:, i] = ocp.get(i, 'u')

    # ---- print status
    status_text = {
        0: "0 - 'success'",
        1: "1 - 'failure'",
        2: "2 - 'maximum number of SQP iterations reached'",
        3: "3 - 'minimum step size in QP solver reached'",
        4: "4 - 'qp solver failed'",
    }
    print(f"Status: {status_text.get(status, f'{status} - unknown')}")

    try:
        sqp_iter = ocp.get_stats('sqp_iter')
        time_tot = ocp.get_stats('time_tot')
        time_lin = ocp.get_stats('time_lin')
        time_sim = ocp.get_stats('time_sim')
        time_qp = ocp.get_stats('time_qp_sol')
        print(f"Total CPU Time: {time_tot*1000:.2f} ms for {sqp_iter} SQP Steps")
        print(f"  linearization: {time_lin*1000:.2f} ms, "
              f"integrator: {time_sim*1000:.2f} ms, "
              f"QP solution: {time_qp*1000:.2f} ms")
    except Exception:
        print("Timing information not available")

    # ---- un-scale
    x_scale = fcn['x_scale']
    x_offs = fcn['x_offs']
    u_scale = fcn['u_scale']
    u_offs = fcn['u_offs']
    ca_scale = fcn['ca_scale']
    ca_offs = fcn['ca_offs']

    x_phys = x_traj_s[:n_states, :] * x_scale.reshape(-1, 1) + x_offs.reshape(-1, 1)
    ca_phys = x_traj_s[n_states, :] * ca_scale[0] + ca_offs[0]
    u_phys = u_traj_s * u_scale.reshape(-1, 1) + u_offs.reshape(-1, 1)

    # Optimal controls: take from first node (should be ~constant)
    opt_vars = u_phys[:, 0]

    result = {
        'u': u_phys,
        'x': x_phys,
        'ca': ca_phys,
        'x_scaled': x_traj_s,
        'u_scaled': u_traj_s,
    }

    return opt_vars, status, result
