"""Complete engine model including combustion and NOx (CasADi version)"""

import casadi as ca
from .subfunctions.in_cylinder import (
    cyl_vol,
    static_cylinder_conditions,
    dynamic_cylinder_conditions,
)
from .subfunctions.nox import two_zone_model, nox_model
from .algebraic_injector_model import algebraic_injector_model
from .ign_del_model import ign_del_model
from .combustion_model import combustion_model
from .utils import check_validity


def complete_model(x, u, ca_deg, par_model, par_op, en_nox=True):
    """Complete in-cylinder model with combustion and optional NOx (CasADi symbolic).

    Args:
        x: CasADi SX state vector [pCyl, QComb, IMEP, (ThetaUZ, NO)]
        u: CasADi SX input vector [SOE1, SOE2, ..., DOE1, DOE2, ...]
        ca_deg: CasADi SX scalar – crank angle [degCA aTDC]
        par_model: ModelParameters instance
        par_op: OperatingPoint instance
        en_nox: bool – include NOx states

    Returns:
        xdot: CasADi expression for dx/d(phi)
        y: CasADi expression for algebraic outputs
    """
    x = check_validity(x, ca_deg, par_model, par_op, en_nox)

    n_inj = u.shape[0] // 2
    soe = u[:n_inj]    # [degCA aTDC]
    doe = u[n_inj:]    # [µs]

    p_cyl = x[0]
    q_comb = x[1]

    v_cyl, dv_cyl, stroke = cyl_vol(ca_deg, par_model.eng)

    kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb = \
        static_cylinder_conditions(q_comb, p_cyl, v_cyl, par_model, par_op)

    tau_ign = ign_del_model(p_cyl, theta_cyl, xi_o2, par_model)

    # Time at (ca - tau_ign) in µs after TDC
    t_prep = (ca_deg / (par_op.eng_spd * 360) - tau_ign) * 1e6
    # SOE in µs after TDC
    t_soe = soe / (par_op.eng_spd * 360) * 1e6

    m_fuel_prep, m_tot = algebraic_injector_model(t_prep, par_op.p_rail,
                                                   t_soe, doe, n_inj)

    # Heat release rate
    dq_comb = combustion_model(q_comb, m_fuel_prep, tau_ign,
                               par_op.eng_spd, par_model)

    # Wall heat transfer and pressure derivative
    dp_cyl, dimep, dq_wall = dynamic_cylinder_conditions(
        ca_deg, p_cyl, dq_comb, v_cyl, dv_cyl, stroke,
        theta_cyl, kappa, spec_r, par_model, par_op)

    phi = (m_fuel_prep * par_model.thermo['fuel']['air_fuel_ratio_st'] *
           par_model.thermo['xi']['air']['O2'] /
           par_op.m_cyl_tot / par_op.xi_o2)

    if en_nox:
        theta_uz = x[3]
        no = x[4]

        d_theta_uz, theta_bz = two_zone_model(
            q_comb, theta_uz, dp_cyl, p_cyl, dv_cyl, v_cyl,
            theta_cyl, kappa, m_tot, par_op, par_model, x_bg, x_bz)

        d_no, no_ppm, no_eq = nox_model(
            q_comb, theta_uz, theta_bz, no, m_tot,
            par_op, par_model, x_bg,
            v_cyl, dv_cyl, p_cyl, dp_cyl, theta_cyl)

        xdot = ca.vertcat(dp_cyl, dq_comb, dimep, d_theta_uz, d_no)
        y = ca.vertcat(theta_cyl, kappa, dq_wall, spec_r, m_fuel_prep,
                       tau_ign, x_bg, theta_bz, no_ppm, phi, no_eq, x_bz)
    else:
        xdot = ca.vertcat(dp_cyl, dq_comb, dimep)
        y = ca.vertcat(theta_cyl, kappa, dq_wall, spec_r, m_fuel_prep,
                       tau_ign, phi, xi_o2)

    return xdot, y
