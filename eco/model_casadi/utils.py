"""Utility functions for model validation (CasADi version)"""

import numpy as np
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


def injection_pattern(soe: np.ndarray, doe: np.ndarray, eng_spd: float,
                      ca_min: float = -30, ca_max: float = 50):
    """Convert SOE/DOE inputs to an injection pulse pattern in crank angle.

    Args:
        soe: Start of energizing per injection [degCA aTDC], shape (n_inj,)
        doe: Duration of energizing per injection [µs], shape (n_inj,)
        eng_spd: Engine speed [1/s]
        ca_min: Crank angle lower bound for the pattern [degCA]
        ca_max: Crank angle upper bound for the pattern [degCA]

    Returns:
        ca_pattern: Crank angle vector for plotting (step-wise)
        u_pattern: Injection pulse signal (1 = injecting, 0 = off)
    """
    soe = np.atleast_1d(np.asarray(soe, dtype=float))
    doe = np.atleast_1d(np.asarray(doe, dtype=float))

    # DOE [µs] -> crank angle duration [degCA]:  1e-6 * rpm/60 * 360
    rpm = eng_spd * 60  # [1/min]
    t2ca = 1e-6 * rpm / 60 * 360  # [degCA/µs]

    # Build step-wise pattern: for each injection create
    #   ca_min -> SOE (off),  SOE -> SOE+DOE*t2ca (on),  then off until next
    ca_pts = [ca_min]
    u_pts = [0]
    for s, d in sorted(zip(soe, doe), key=lambda x: x[0]):
        eoe = s + d * t2ca  # end of energizing [degCA]
        ca_pts.extend([s, s, eoe, eoe])
        u_pts.extend([0, 1, 1, 0])
    ca_pts.append(ca_max)
    u_pts.append(0)

    return np.array(ca_pts), np.array(u_pts)
