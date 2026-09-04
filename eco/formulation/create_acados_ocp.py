"""Create acados OCP for injection optimization (economic formulation).

Mirrors the MATLAB ``createAcadosFunctions_InjOpt.m`` transcription exactly:

  * The injection inputs u = [SOE1, SOE2, DOE1, DOE2] are NOT acados controls.
    They are carried as augmented STATES with zero dynamics, so whatever value
    they take at node 0 is propagated unchanged over the whole horizon. The
    problem therefore has ``nu = 0`` controls; the only free decision is the
    injection part of the initial state, box-constrained to [u_min, u_max]
    at node 0.

  * acados state:   x = [u(nInputs); x_phys(nStates); ca(1)]   (all scaled)
  * acados control: none (nu = 0)
  * parameters:     p = [dt_inj]  (injection distancing time [µs])

  Stage constraints    h   = [pCyl, dpCyl, h_coc]                    (scaled)
  Terminal constraints h_e = [IMEP, Tevo, (NOppm,) Phi, bInj...]     (scaled)
  Cost: stage = 0, terminal = total injected fuel energy (scaled QComb).
"""

import numpy as np
import os
from typing import Tuple, Any, Dict
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca

from eco.formulation.casadi_model import complete_model_sym
from eco.formulation.scale_unscale import scale_unscale
from eco.model_casadi.algebraic_injector_model import algebraic_injector_model


def create_acados_functions_inj_opt(par_opt: Any, par_model: Any, par_op: Any,
                                    model_name: str = 'ECO',
                                    code_export_subdir: str = 'c_generated_code') -> Tuple[AcadosOcpSolver, Dict]:
    """Create acados OCP with economic cost + nonlinear constraints.

    See module docstring for the state/control layout. Returns the compiled
    ``AcadosOcpSolver`` (or ``None`` on failure) and a ``fcn`` dict with scaling
    info and dimensions used by the init / run helpers.
    """
    n_states = par_model.n_states
    n_inputs = par_model.n_inputs
    n_inj = n_inputs // 2
    en_nox = par_opt.en_nox
    ca_num = np.asarray(par_opt.ca, dtype=float)
    N = len(ca_num) - 1
    T = float(np.sum(np.diff(ca_num)))

    # ---- Scaling ----
    if en_nox:
        x_names = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO']
    else:
        x_names = ['pCyl', 'QComb', 'IMEP']
    soe_names = ['SOE'] * n_inj
    doe_names = ['DOE'] * n_inj
    u_names = soe_names + doe_names

    _, x_scale, x_offs = scale_unscale(np.zeros(n_states), x_names,
                                       par_opt, 'scale', False)
    _, u_scale, u_offs = scale_unscale(np.zeros(n_inputs), u_names,
                                       par_opt, 'scale', False)
    _, ca_scale, ca_offs = scale_unscale(np.array([0.0]), ['SOE'],
                                         par_opt, 'scale', False)

    nx_full = n_inputs + n_states + 1   # [u; x_phys; ca]
    nu = 0                              # no acados controls (u are states)

    # ---- CasADi symbolics: augmented state ux = [u; x_phys; ca] (scaled) ----
    ux_sym = ca.SX.sym('ux', nx_full)
    u_sym = ux_sym[:n_inputs]                        # scaled injection inputs
    x_sym = ux_sym[n_inputs:n_inputs + n_states]     # scaled physical states
    ca_sym = ux_sym[n_inputs + n_states]             # scaled crank angle

    p_sym = ca.SX.sym('p', 1)      # [dt_inj]
    dt_inj = p_sym[0]

    # scaling vectors
    x_sc_vec = ca.vertcat(*x_scale.tolist())
    x_of_vec = ca.vertcat(*x_offs.tolist())
    u_sc_vec = ca.vertcat(*u_scale.tolist())
    u_of_vec = ca.vertcat(*u_offs.tolist())

    # unscale to physical
    x_phys = x_sym * x_sc_vec + x_of_vec
    u_phys = u_sym * u_sc_vec + u_of_vec
    ca_phys = ca_sym * ca_scale[0] + ca_offs[0]

    soe_phys = u_phys[:n_inj]      # [degCA aTDC]
    doe_phys = u_phys[n_inj:]      # [µs]
    soe_mus = soe_phys / (par_op.eng_spd * 360) * 1e6

    # ---- ODE ----
    xdot_phys, _y_phys = complete_model_sym(x_phys, u_phys, ca_phys,
                                            par_model, par_op, en_nox)
    xdot_scaled = xdot_phys / x_sc_vec

    # Augmented dynamics: u has zero dynamics (held constant across horizon),
    # ca advances at 1/ca_scale, physical states follow the model ODE.
    f_expl = ca.vertcat(
        ca.SX.zeros(n_inputs, 1),   # du/dphi = 0  (constant injection inputs)
        xdot_scaled,                # dx_phys/dphi (scaled)
        1.0 / ca_scale[0],          # dca_s/dphi
    )

    # ---- Algebraic injector model (total mass + per-injection DOI) ----
    _, m_tot = algebraic_injector_model(0, par_op.p_rail, soe_mus,
                                        doe_phys, n_inj)
    doe_min = 66.73   # µs
    eps_lim = 1.0
    doi_vec = []
    for i in range(n_inj):
        doe_i = doe_phys[i]
        doe_lim_i = (0.5 * ca.sqrt((doe_i - doe_min)**2 + eps_lim)
                     + 0.5 * (doe_i - doe_min) + doe_min)
        doi_vec.append((doe_lim_i - doe_min) * 2.093)

    # ---- AcadosModel ----
    model = AcadosModel()
    model.name = model_name
    model.x = ux_sym
    model.u = ca.SX.sym('u', 0)     # nu = 0
    model.xdot = ca.SX.sym('xdot', nx_full)
    model.p = p_sym
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl

    # ---- Stage constraints: h = [pCyl_s, dpCyl_s, h_coc] ----
    # (matching MATLAB constr_expr_h = [x{1}; xdot{1}; h_coc])
    h_pcyl = x_sym[0]           # pCyl scaled
    h_dpcyl = xdot_scaled[0]    # dpCyl/dphi scaled

    qcomb_scale = float(x_scale[1])
    q_tot = m_tot * par_model.thermo['fuel']['low_heat_val']
    h_coc = x_sym[1] - q_tot / (2.0 * qcomb_scale)   # QComb_s - Qtot/(2*scale)

    model.con_h_expr = ca.vertcat(h_pcyl, h_dpcyl, h_coc)
    nh = 3

    # ---- Terminal constraints h_e = [IMEP, Tevo, (NOppm,) Phi, bInj...] ----
    # (matching MATLAB constr_expr_h_e). Tevo and NOppm come from the model
    # outputs y{1}/y{9} evaluated at the terminal node (state-only functions);
    # Phi and injection distancing depend only on the constant u state.
    _, y_e = complete_model_sym(x_phys, u_phys, ca_phys,
                                par_model, par_op, en_nox)
    theta_cyl_e = y_e[0]        # MATLAB y{1}: mean cylinder temperature -> Tevo

    theta_scale = par_opt.scale[par_opt.scale_offs_vars.index('Theta')]
    theta_offs = par_opt.offs[par_opt.scale_offs_vars.index('Theta')]

    # Phi (equivalence ratio from total injected fuel, matching MATLAB)
    phi_e = (m_tot * par_model.thermo['fuel']['air_fuel_ratio_st']
             * par_model.thermo['xi']['air']['O2']
             / par_op.m_cyl_tot / par_op.xi_o2)

    # Injection distancing (DOI-based), scaled like MATLAB u(i)-u(i-1)-dCA
    soe_scale = float(u_scale[0])
    b_inj = []
    for i in range(1, n_inj):
        dca_inj_phys = (doi_vec[i - 1] + dt_inj) * par_op.eng_spd * 360 * 1e-6
        dca_inj_scaled = dca_inj_phys / soe_scale
        b_inj.append(u_sym[i] - u_sym[i - 1] - dca_inj_scaled)

    h_e_list = [x_sym[2]]                                  # IMEP (scaled)
    h_e_list.append((theta_cyl_e - theta_offs) / theta_scale)   # Tevo
    if en_nox:
        noppm_scale = par_opt.scale[par_opt.scale_offs_vars.index('NOppm')]
        noppm_offs = par_opt.offs[par_opt.scale_offs_vars.index('NOppm')]
        no_ppm_e = y_e[8]                                 # MATLAB y{9}
        h_e_list.append((no_ppm_e - noppm_offs) / noppm_scale)
        idx_nox = 2          # NOx slot in h_e
    h_e_list.append(phi_e)                                # Phi
    h_e_list.extend(b_inj)                                # injection distancing

    model.con_h_expr_e = ca.vertcat(*h_e_list)
    nh_e = len(h_e_list)

    # ---- Cost: economic (minimize total injected fuel energy) ----
    model.cost_expr_ext_cost = ca.SX(0)                   # stage cost = 0
    model.cost_expr_ext_cost_e = q_tot / qcomb_scale      # terminal cost

    # ---- OCP assembly ----
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.parameter_values = np.array([getattr(par_opt, 'dt_inj', 400.0)])

    # Non-uniform time steps (fine over opt range, coarse to EVO)
    dt_vec = np.diff(ca_num).astype(float)
    if not np.allclose(dt_vec, dt_vec[0]):
        ocp.solver_options.time_steps = dt_vec
    ocp.solver_options.tf = T

    # ---- Initial-state box (all indices; tightened at runtime) ----
    inf_rep = 1e2
    ocp.constraints.idxbx_0 = np.arange(nx_full)
    ocp.constraints.lbx_0 = -inf_rep * np.ones(nx_full)
    ocp.constraints.ubx_0 = inf_rep * np.ones(nx_full)

    # ---- Stage nonlinear constraints (path nodes 1..N-1) ----
    ocp.constraints.lh = -inf_rep * np.ones(nh)
    ocp.constraints.uh = inf_rep * np.ones(nh)
    # Slack on dpMax (index 1), matching MATLAB constr_Jsh = [0;1;0]
    ocp.constraints.idxsh = np.array([1])
    ocp.cost.Zl = np.array([1e3])
    ocp.cost.Zu = np.array([20.0])
    ocp.cost.zl = np.array([1e3])
    ocp.cost.zu = np.array([1.0])

    # ---- Terminal nonlinear constraints ----
    ocp.constraints.lh_e = -inf_rep * np.ones(nh_e)
    ocp.constraints.uh_e = inf_rep * np.ones(nh_e)
    if en_nox:
        # Slack on NOx (index 2), matching MATLAB constr_Jsh_e
        ocp.constraints.idxsh_e = np.array([idx_nox])
        ocp.cost.Zl_e = np.array([1e3])
        ocp.cost.Zu_e = np.array([10.0])
        ocp.cost.zl_e = np.array([1e3])
        ocp.cost.zu_e = np.array([10.0])

    # ---- Solver options (matching MATLAB createAcadosFunctions_InjOpt) ----
    nlp_solver = 'SQP' if par_opt.sqp['n_sqp_max'] > 1 else 'SQP_RTI'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.nlp_solver_type = nlp_solver
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = getattr(par_opt, 'n_int',
                                                      par_opt.sqp.get('n_int', 1))
    ocp.solver_options.nlp_solver_max_iter = par_opt.sqp['n_sqp_max']
    ocp.solver_options.nlp_solver_step_length = par_opt.sqp['step_size']
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-6
    # Hessian handling matched to the MATLAB benchmark: MATLAB's default
    # Gauss-Newton Hessian keeps only the (exact) cost Hessian and drops the
    # indefinite second-order terms of the dynamics and nonlinear constraints.
    # EXTERNAL cost forces hessian_approx='EXACT', so reproduce that behaviour
    # by switching off the exact dynamics/constraint Hessians; the remaining
    # cost Hessian plus Levenberg-Marquardt keeps HPIPM well-posed.
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'

    # Code generation
    code_gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                code_export_subdir)
    os.makedirs(code_gen_dir, exist_ok=True)
    ocp.code_export_directory = code_gen_dir

    # ---- Create solver ----
    try:
        ocp_solver = AcadosOcpSolver(ocp, json_file=os.path.join(
            code_gen_dir, f'acados_ocp_{model_name}.json'))
    except Exception as e:
        print(f"ERROR: Could not create acados solver: {e}")
        import traceback
        traceback.print_exc()
        ocp_solver = None

    fcn = {
        'x_scale': x_scale,
        'x_offs': x_offs,
        'u_scale': u_scale,
        'u_offs': u_offs,
        'ca_scale': ca_scale,
        'ca_offs': ca_offs,
        'nx_full': nx_full,   # full acados state dim [u; x_phys; ca]
        'nu': nu,             # 0
        'n_states': n_states,
        'n_inputs': n_inputs,
        'n_inj': n_inj,
        'nh': nh,
        'nh_e': nh_e,
        'en_nox': en_nox,
    }

    return ocp_solver, fcn
