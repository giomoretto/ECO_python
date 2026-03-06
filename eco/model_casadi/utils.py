"""Utility functions for model validation (CasADi version)"""

import casadi as ca
from .subfunctions.in_cylinder import cyl_vol
from .subfunctions.ignition_delay import saturate_input


def check_validity(x, ca_deg, par_model, par_op, en_nox):
    """Check and limit states to physical boundaries (CasADi symbolic).

    Args:
        x: CasADi SX state vector
        ca_deg: Crank angle [degCA]
        par_model: Model parameters
        par_op: Operating point parameters
        en_nox: Whether NOx states are included

    Returns:
        x: Corrected state vector (CasADi SX)
    """
    p_cyl = x[0]
    q_comb = x[1]
    imep = x[2]

    # pCyl lower bound
    eps = 1e-3
    v_cyl, _, _ = cyl_vol(ca_deg, par_model.eng)
    lb = 0.8 * par_op.p_int * (par_op.v_int / v_cyl)**1.4
    p_cyl = 0.5 * (p_cyl - lb) + 0.5 * ca.sqrt(eps**2 + (p_cyl - lb)**2) + lb

    # QComb bounds
    x_bz_max = 0.95
    q_comb_max = ((-par_model.thermo['fuel']['low_heat_val'] * par_op.m_cyl_tot * x_bz_max) /
                  (x_bz_max - par_model.thermo['fuel']['air_fuel_ratio_st'] / (1 - par_op.xi_bg) - 1))
    q_comb = saturate_input(q_comb, 0, q_comb_max)

    if en_nox:
        theta_uz = x[3]
        nox = x[4]
        eps_nox = 1e-10
        lb_nox = 0
        nox = 0.5 * (nox - lb_nox) + 0.5 * ca.sqrt(eps_nox**2 + (nox - lb_nox)**2) + lb_nox
        return ca.vertcat(p_cyl, q_comb, imep, theta_uz, nox)
    else:
        return ca.vertcat(p_cyl, q_comb, imep)
