"""Create acados OCP for injection optimization

This module creates the acados optimal control problem formulation using a
fully symbolic CasADi model.
"""

import numpy as np
import os
import sys
from typing import Tuple, Any, Dict
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca

sys.path.insert(0, '/home/morettog/projects/phd/python_code')
from eco.formulation.casadi_model import complete_model_sym
from eco.formulation.scale_unscale import scale_unscale


def create_acados_functions_inj_opt(par_opt: Any, par_model: Any,
                                    par_op: Any) -> Tuple[AcadosOcpSolver, Dict]:
    """Create acados OCP formulation for injection optimization.

    Formulation
    -----------
    States:   x = [pCyl, QComb, IMEP, ThetaUZ, NO, ca]   (n_states + 1)
    Controls: u = [SOE1, SOE2, ..., DOE1, DOE2, ...]      (n_inputs)

    All variables are in scaled (≈ O(1)) units.
    The controls appear in the ODE (injection timing/duration affect
    combustion) but their time derivative is not modeled – they are
    piece-wise constant over each shooting interval.  We add a small
    regularization ‖u‖² in the stage cost so that the QP Hessian in the
    u-block is positive-definite.

    Args:
        par_opt: Optimization parameters
        par_model: Model parameters
        par_op: OperatingPoint instance (needed for symbolic model build)

    Returns:
        ocp_solver: AcadosOcpSolver (or None if code-gen fails)
        fcn: dict with CasADi helper functions and scaling vectors
    """
    # ------------------------------------------------------------------ dims
    n_states = par_model.n_states
    n_inputs = par_model.n_inputs
    n_inj = n_inputs // 2
    ca_num = par_opt.ca
    N = len(ca_num) - 1
    T = float(np.sum(np.diff(ca_num)))
    en_nox = par_opt.en_nox

    # -------------------------------------------------------- scaling helpers
    if en_nox:
        x_names = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO']
    else:
        x_names = ['pCyl', 'QComb', 'IMEP']
    soe_names = ['SOE'] * n_inj
    doe_names = ['DOE'] * n_inj
    u_names = soe_names + doe_names

    _, x_scale, x_offs = scale_unscale(np.zeros(n_states), x_names, par_opt,
                                       'scale', False)
    _, u_scale, u_offs = scale_unscale(np.zeros(n_inputs), u_names, par_opt,
                                       'scale', False)
    _, ca_scale, ca_offs = scale_unscale(np.array([0.0]), ['SOE'], par_opt,
                                         'scale', False)

    nx = n_states + 1          # physical states + crank angle
    nu = n_inputs              # injection parameters

    # -------------------------------------------- CasADi symbolic variables
    x_sym = ca.SX.sym('x', nx)          # [pCyl_s, QComb_s, IMEP_s, Theta_s, NO_s, ca_s]
    u_sym = ca.SX.sym('u', nu)          # [SOE1_s, SOE2_s, DOE1_s, DOE2_s]

    x_phys_states = x_sym[:n_states] * ca.vertcat(*x_scale.tolist()) \
                    + ca.vertcat(*x_offs.tolist())
    ca_phys = x_sym[n_states] * ca_scale[0] + ca_offs[0]
    u_phys = u_sym * ca.vertcat(*u_scale.tolist()) + ca.vertcat(*u_offs.tolist())

    # -------------------------------------------- build symbolic ODE
    xdot_phys, y_phys = complete_model_sym(x_phys_states, u_phys, ca_phys,
                                           par_model, par_op, en_nox)

    xdot_scaled = xdot_phys / ca.vertcat(*x_scale.tolist())

    f_expl = ca.vertcat(
        xdot_scaled,           # physical state derivatives (scaled)
        1.0 / ca_scale[0],    # d(ca_scaled)/dphi
    )

    # -------------------------------------------------- acados Model
    model = AcadosModel()
    model.name = 'ECO_inj_opt'
    model.x = x_sym
    model.u = u_sym
    model.p = ca.SX.sym('p', 0)
    model.f_expl_expr = f_expl
    model.xdot = ca.SX.sym('xdot', nx)
    model.f_impl_expr = model.xdot - f_expl

    # ------------------------------------------------ cost expressions
    idx_imep = 2               # IMEP is 3rd state
    idx_pcyl = 0               # pCyl is 1st state

    imep_ref_s, _, _ = scale_unscale(np.array([par_opt.reference['imep']]),
                                     ['IMEP'], par_opt, 'scale', False)
    imep_ref_s = float(imep_ref_s[0])

    p_max_ref_s, _, _ = scale_unscale(np.array([par_opt.reference['p_max']]),
                                      ['pCyl'], par_opt, 'scale', False)
    p_max_ref_s = float(p_max_ref_s[0])

    # Stage cost: small regularisation on u so QP Hessian is positive-definite
    w_reg = 1e-4
    cost_path = w_reg * ca.dot(u_sym, u_sym)

    # Terminal cost ----------------------------------------------------------
    w_imep = 1e1
    w_pmax = 1e1

    cost_e = w_imep * (imep_ref_s - x_sym[idx_imep])**2

    # Soft penalty on max pressure (smooth max(pCyl - p_max_ref, 0))
    eps_p = 1e-6
    pmax_viol = 0.5 * ((x_sym[idx_pcyl] - p_max_ref_s) +
                        ca.sqrt((x_sym[idx_pcyl] - p_max_ref_s)**2 + eps_p))
    cost_e += w_pmax * pmax_viol**2

    if en_nox:
        idx_no = 4
        c_nox_ref_s, _, _ = scale_unscale(np.array([par_opt.reference['c_nox']]),
                                          ['NO'], par_opt, 'scale', False)
        c_nox_ref_s = float(c_nox_ref_s[0])
        w_nox = 1e1
        nox_viol = 0.5 * ((x_sym[idx_no] - c_nox_ref_s) +
                           ca.sqrt((x_sym[idx_no] - c_nox_ref_s)**2 + eps_p))
        cost_e += w_nox * nox_viol**2

    model.cost_expr_ext_cost = cost_path
    model.cost_expr_ext_cost_e = cost_e

    # -------------------------------------------------- helper functions
    ux_sym_full = ca.vertcat(u_sym, x_sym)
    f_sim = ca.Function('f_sim', [x_sym, u_sym], [f_expl],
                        ['x', 'u'], ['xdot'])
    f_cost_e = ca.Function('f_cost_e', [x_sym], [cost_e], ['x'], ['cost'])

    # -------------------------------------------------- acados OCP
    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # ---------------------------------- constraints (scaled)
    lbu_s, _, _ = scale_unscale(np.array(par_opt.u_min), u_names, par_opt,
                                'scale', False)
    ubu_s, _, _ = scale_unscale(np.array(par_opt.u_max), u_names, par_opt,
                                'scale', False)
    lbx_phys_s, _, _ = scale_unscale(np.array(par_opt.x_min[:n_states]),
                                     x_names, par_opt, 'scale', False)
    ubx_phys_s, _, _ = scale_unscale(np.array(par_opt.x_max[:n_states]),
                                     x_names, par_opt, 'scale', False)
    calb_s, _, _ = scale_unscale(np.array([par_opt.ca[0]]), ['SOE'], par_opt,
                                 'scale', False)
    caub_s, _, _ = scale_unscale(np.array([par_opt.ca[-1]]), ['SOE'], par_opt,
                                 'scale', False)

    # Control bounds (same at every stage)
    ocp.constraints.lbu = lbu_s
    ocp.constraints.ubu = ubu_s
    ocp.constraints.idxbu = np.arange(nu)

    # State bounds – path (stages 1..N-1)
    lbx_path = np.concatenate([lbx_phys_s, calb_s])
    ubx_path = np.concatenate([ubx_phys_s, caub_s])
    ocp.constraints.lbx = lbx_path
    ocp.constraints.ubx = ubx_path
    ocp.constraints.idxbx = np.arange(nx)

    # State bounds – initial (fix physical states + ca at node 0)
    # The actual values are set via init_acados_ocp at runtime
    ocp.constraints.lbx_0 = np.concatenate([lbx_phys_s, calb_s])
    ocp.constraints.ubx_0 = np.concatenate([ubx_phys_s, calb_s])
    ocp.constraints.idxbx_0 = np.arange(nx)

    # State bounds – terminal
    ocp.constraints.lbx_e = lbx_path
    ocp.constraints.ubx_e = ubx_path
    ocp.constraints.idxbx_e = np.arange(nx)

    # ---------------------------------- solver options
    if par_opt.sqp['n_sqp_max'] == 1:
        nlp_solver = 'SQP_RTI'
    else:
        nlp_solver = 'SQP'

    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hpipm_mode = 'ROBUST'
    ocp.solver_options.nlp_solver_type = nlp_solver
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_max_iter = par_opt.sqp['n_sqp_max']
    ocp.solver_options.nlp_solver_step_length = par_opt.sqp['step_size']
    ocp.solver_options.levenberg_marquardt = 1e0
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-6
    ocp.solver_options.tf = T
    ocp.solver_options.hessian_approx = 'EXACT'

    code_gen_dir = os.path.join(os.path.dirname(__file__), 'c_generated_code')
    os.makedirs(code_gen_dir, exist_ok=True)
    ocp.code_export_directory = code_gen_dir

    # ---------------------------------- create solver
    try:
        ocp_solver = AcadosOcpSolver(ocp, json_file=os.path.join(
            code_gen_dir, 'acados_ocp.json'))
    except Exception as e:
        print(f"Warning: Could not create acados solver: {e}")
        import traceback
        traceback.print_exc()
        ocp_solver = None

    fcn = {
        'f_sim': f_sim,
        'f_cost_e': f_cost_e,
        'x_scale': x_scale,
        'x_offs': x_offs,
        'u_scale': u_scale,
        'u_offs': u_offs,
        'ca_scale': ca_scale,
        'ca_offs': ca_offs,
        'nx': nx,
        'nu': nu,
    }

    return ocp_solver, fcn
