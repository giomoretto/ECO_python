"""Initialize acados OCP with simulation-based trajectories and runtime bounds.

State layout (see create_acados_ocp.py):
  x = [u(nInputs); x_phys(nStates); ca(1)]   (all scaled), no controls (nu=0).

The injection inputs u are the injection part of the initial state, free within
[u_min, u_max] at node 0 and propagated unchanged by their zero dynamics.

Warm-starting (matching MATLAB createInitAcadosOCP_InjOpt): the simulation-based
initial trajectory is set only on the very first solve. Subsequent solves in a
Pareto sweep are NOT reset and keep the previous solution as their initial guess,
so tightening a reference (e.g. NOx) is approached gradually. Only the
constraint bounds (which carry the changing references) are refreshed each solve.
"""

import numpy as np

from eco.formulation.scale_unscale import scale_unscale
from eco.simulation.complete_simulation import complete_simulation


def create_init_acados_ocp_inj_opt(ocp, fcn, par_model, par_sim, par_opt):
    """Initialize the acados OCP for one solve (bounds, references, guess)."""
    if ocp is None:
        return ocp

    n_inj = fcn['n_inj']
    en_nox = fcn['en_nox']
    N = len(par_opt.ca) - 1

    soe_names = ['SOE'] * n_inj
    doe_names = ['DOE'] * n_inj
    u_names = soe_names + doe_names

    x_scale = fcn['x_scale']
    x_offs = fcn['x_offs']
    u_scale = fcn['u_scale']
    u_offs = fcn['u_offs']
    ca_scale = fcn['ca_scale']
    ca_offs = fcn['ca_offs']

    # ---- first solve only: build simulation-based initial trajectory --------
    if not fcn.get('_initialized', False):
        # simulate IVC -> start of optimisation range for the initial state
        ca_pre_opt = np.arange(par_sim.op.ca_ivc,
                               par_opt.optimization_range[0],
                               par_sim.opts['delta_phi'])
        u0_pre_sim = np.tile(par_opt.u0, (len(ca_pre_opt), 1)).T
        x0_pre_opt = np.array([par_sim.op.p_int, 0, 0])
        if en_nox:
            x0_pre_opt = np.append(x0_pre_opt, [par_sim.op.theta_ivc, 0])
        sim_pre_opt = complete_simulation(ca_pre_opt, x0_pre_opt, u0_pre_sim,
                                          par_sim, par_model)
        x0_opt = sim_pre_opt['x'][:, -1]     # physical states at OCP start

        # simulate over the optimisation range with u0 for the state guess
        u0_sim = np.tile(par_opt.u0, (len(par_opt.ca), 1)).T
        sim_opt0 = complete_simulation(par_opt.ca, x0_opt, u0_sim,
                                       par_sim, par_model)

        x_traj_s = (sim_opt0['x'] - x_offs.reshape(-1, 1)) / x_scale.reshape(-1, 1)
        ca_traj_s = (par_opt.ca - ca_offs[0]) / ca_scale[0]
        u0_scaled = (par_opt.u0 - u_offs) / u_scale
        u_traj_s = np.tile(u0_scaled.reshape(-1, 1), (1, len(par_opt.ca)))
        ux_init = np.vstack([u_traj_s, x_traj_s, ca_traj_s.reshape(1, -1)])
        for i in range(N + 1):
            ocp.set(i, 'x', ux_init[:, i])

        fcn['x0_opt'] = x0_opt
        fcn['_initialized'] = True

    x0_opt = fcn['x0_opt']

    # ---- fix initial state at node 0 (constant across the sweep) ------------
    #   u free within [u_min, u_max], physical states pinned to x0_opt, ca pinned
    lbu_s, _, _ = scale_unscale(par_opt.u_min, u_names, par_opt, 'scale', False)
    ubu_s, _, _ = scale_unscale(par_opt.u_max, u_names, par_opt, 'scale', False)
    x0_phys_s = (x0_opt - x_offs) / x_scale
    ca0_s = (par_opt.ca[0] - ca_offs[0]) / ca_scale[0]
    lbx0 = np.concatenate([lbu_s, x0_phys_s, [ca0_s]])
    ubx0 = np.concatenate([ubu_s, x0_phys_s, [ca0_s]])
    ocp.constraints_set(0, 'lbx', lbx0)
    ocp.constraints_set(0, 'ubx', ubx0)

    # ---- parameter values (dt_inj) on every node ---------------------------
    p_val = np.array([getattr(par_opt, 'dt_inj', 400.0)])
    for i in range(N + 1):
        ocp.set(i, 'p', p_val)

    # ---- stage nonlinear constraint bounds: h = [pCyl, dpCyl, h_coc] -------
    inf_rep = 1e2
    pmax_s, _, _ = scale_unscale(np.array([par_opt.reference['p_max']]),
                                 ['pCyl'], par_opt, 'scale', False)
    dpmax_s, _, _ = scale_unscale(np.array([par_opt.reference['dp_max']]),
                                  ['pCyl'], par_opt, 'scale', True)

    coc_max = par_opt.reference.get('coc_max', 20.0)
    coc_node = N
    for k in range(N + 1):
        if par_opt.ca[k] >= coc_max:
            coc_node = k
            break

    # Path nodes 1..N-1 (node 0 is at the start of the range: constraints inert)
    for i in range(1, N):
        uh = np.array([float(pmax_s[0]), float(dpmax_s[0]), inf_rep])
        lh = np.array([0.0, -inf_rep, 0.0 if i >= coc_node else -inf_rep])
        ocp.constraints_set(i, 'uh', uh)
        ocp.constraints_set(i, 'lh', lh)

    # ---- terminal nonlinear constraint bounds ------------------------------
    #   h_e = [IMEP, Tevo, (NOppm,) Phi, bInj...]
    imep_s, _, _ = scale_unscale(np.array([par_opt.reference['imep']]),
                                 ['IMEP'], par_opt, 'scale', False)
    tevo_s, _, _ = scale_unscale(np.array([par_opt.reference['t_min']]),
                                 ['Theta'], par_opt, 'scale', False)
    nh_e = fcn['nh_e']
    lh_e = -inf_rep * np.ones(nh_e)
    uh_e = inf_rep * np.ones(nh_e)
    lh_e[0] = float(imep_s[0])      # IMEP >= ref
    lh_e[1] = float(tevo_s[0])      # Tevo >= Tmin
    idx = 2
    if en_nox:
        cnox_s, _, _ = scale_unscale(np.array([par_opt.reference['c_nox']]),
                                     ['NOppm'], par_opt, 'scale', False)
        uh_e[idx] = float(cnox_s[0])   # NOx <= cNOx
        lh_e[idx] = 0.0
        idx += 1
    uh_e[idx] = float(par_opt.reference['phi_max'])   # Phi <= PhiMax
    lh_e[idx] = 0.0
    idx += 1
    for _ in range(n_inj - 1):        # injection distancing >= 0
        lh_e[idx] = 0.0
        idx += 1
    ocp.constraints_set(N, 'lh', lh_e)
    ocp.constraints_set(N, 'uh', uh_e)

    return ocp
