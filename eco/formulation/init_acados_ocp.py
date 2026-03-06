"""Initialize acados OCP with initial trajectory

This module handles OCP initialization with simulation-based trajectories.
"""

import numpy as np
from typing import Any, Dict
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')
from eco.simulation.complete_simulation import complete_simulation
from eco.formulation.scale_unscale import scale_unscale


def create_init_acados_ocp_inj_opt(ocp: Any, fcn: Dict,
                                   par_model: Any,
                                   par_sim: Any,
                                   par_opt: Any) -> Any:
    """Initialize acados OCP with trajectories from simulation

    The OCP uses:
      x = [pCyl, QComb, IMEP, ThetaUZ, NO, ca]   (n_states + 1)
      u = [SOE1, SOE2, ..., DOE1, DOE2, ...]      (n_inputs)

    Args:
        ocp: Acados OCP solver object
        fcn: Dictionary returned by create_acados_functions_inj_opt
        par_model: Model parameters
        par_sim: Simulation parameters
        par_opt: Optimization parameters

    Returns:
        ocp: Updated OCP solver object
    """
    if ocp is None:
        return ocp

    n_states = par_model.n_states
    n_inputs = par_model.n_inputs
    n_inj = n_inputs // 2
    en_nox = par_opt.en_nox
    N = len(par_opt.ca) - 1
    nx = fcn['nx']
    nu = fcn['nu']

    # ---- variable name lists
    if en_nox:
        x_names = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO']
    else:
        x_names = ['pCyl', 'QComb', 'IMEP']
    soe_names = ['SOE'] * n_inj
    doe_names = ['DOE'] * n_inj
    u_names = soe_names + doe_names

    # ---- run simulation from IVC to start of optimisation range
    ca_pre_opt = np.arange(par_sim.op.ca_ivc,
                           par_opt.optimization_range[0],
                           par_sim.opts['delta_phi'])

    u0_pre_sim = np.tile(par_opt.u0, (len(ca_pre_opt), 1)).T

    x0_pre_opt = np.array([par_sim.op.p_int, 0, 0])
    if en_nox:
        x0_pre_opt = np.append(x0_pre_opt, [par_sim.op.theta_ivc, 0])

    sim_pre_opt = complete_simulation(ca_pre_opt, x0_pre_opt, u0_pre_sim,
                                      par_sim, par_model)
    x0_opt = sim_pre_opt['x'][:, -1]

    # ---- simulate over optimisation range with u0
    u0_sim = np.tile(par_opt.u0, (len(par_opt.ca), 1)).T
    sim_opt0 = complete_simulation(par_opt.ca, x0_opt, u0_sim,
                                   par_sim, par_model)

    # ---- scale trajectories
    x_scale = fcn['x_scale']
    x_offs = fcn['x_offs']
    u_scale = fcn['u_scale']
    u_offs = fcn['u_offs']
    ca_scale = fcn['ca_scale']
    ca_offs = fcn['ca_offs']

    # Scale physical states
    x_traj_init = (sim_opt0['x'] - x_offs.reshape(-1, 1)) / x_scale.reshape(-1, 1)
    # Scale crank angle
    ca_traj_init = (par_opt.ca - ca_offs[0]) / ca_scale[0]
    # Build acados state: [states_scaled; ca_scaled]
    x_acados_init = np.vstack([x_traj_init, ca_traj_init.reshape(1, -1)])

    # Scale controls
    u0_scaled = (par_opt.u0 - u_offs) / u_scale

    # ---- set initial guess on all shooting nodes
    for i in range(N + 1):
        ocp.set(i, 'x', x_acados_init[:, i])
    for i in range(N):
        ocp.set(i, 'u', u0_scaled)

    # ---- fix initial physical-state values at node 0
    x0_phys_scaled = (x0_opt - x_offs) / x_scale
    ca0_scaled = (par_opt.ca[0] - ca_offs[0]) / ca_scale[0]
    lbx_0 = np.concatenate([x0_phys_scaled, [ca0_scaled]])
    ubx_0 = np.concatenate([x0_phys_scaled, [ca0_scaled]])

    ocp.constraints_set(0, 'lbx', lbx_0)
    ocp.constraints_set(0, 'ubx', ubx_0)

    return ocp
