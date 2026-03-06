"""Algebraic injector model (CasADi version)"""

import numpy as np
import casadi as ca


def algebraic_injector_model(t, p_rail, soe, doe, n_inj):
    """Calculate injected fuel mass trajectory (CasADi symbolic).

    Args:
        t: Scalar time [µs] (CasADi symbolic)
        p_rail: Rail pressure [Pa]
        soe: SX vector of SOE values [µs] (length n_inj)
        doe: SX vector of DOE values [µs] (length n_inj)
        n_inj: Number of injections

    Returns:
        m_act: Total injected fuel mass at time t [kg]
        m_tot: Total injected fuel mass over all injections [kg]
    """
    rho_diesel = 830
    d_noz = 120e-6
    n_noz = 8
    a_noz_tot = (d_noz / 2)**2 * np.pi * n_noz
    inj_del = 55  # µs
    doe_min = 66.73  # µs
    eps_lim = 1

    m_act = 0
    m_tot_acc = 0

    for i in range(n_inj):
        doe_i = doe[i]
        soe_i = soe[i]

        doe_lim = 0.5 * ca.sqrt((doe_i - doe_min)**2 + eps_lim) + 0.5 * (doe_i - doe_min) + doe_min

        p_cyl_inj = 60e5
        delta_p = p_rail - p_cyl_inj
        u_bernoulli = ca.sqrt(2 * delta_p / rho_diesel)

        corr_fct_s = 3.006 - 9090 * p_rail**(-0.4791)
        so = 100 * corr_fct_s * 1e-3
        sc = 200 * corr_fct_s * 1e-3

        doi = (doe_lim - doe_min) * 2.093

        corr_fct_cd = 9.8e-9 * p_rail + 0.425
        doi_lim = 0.5 * ca.sqrt((doi - 70)**2 + eps_lim) + 0.5 * (doi - 70) + 70
        coeff_d = (0.9301 - 0.02744 * (doi_lim * 1e-6)**(-0.3676)) * corr_fct_cd
        doi_lower = 70
        eps = 1e-2
        coeff_d_lim = (0.9301 - 0.02744 * (doi_lower * 1e-6)**(-0.3676)) * corr_fct_cd
        coeff_d = 0.5 * (coeff_d + coeff_d_lim + ca.sqrt(eps + (coeff_d - coeff_d_lim)**2))

        dm_fuel = coeff_d * u_bernoulli * a_noz_tot * rho_diesel * 1e3
        dm_fuel_max = doi * (so * sc) / (so + sc)
        eps2 = 1e-2
        dm_fuel = dm_fuel_max - 0.5 * ca.sqrt((dm_fuel - dm_fuel_max)**2 + eps2) + 0.5 * (dm_fuel - dm_fuel_max)

        m_tot_i = 1e-9 * (doi * dm_fuel - 0.5 * dm_fuel**2 * (1 / so + 1 / sc))
        m_tot_acc += m_tot_i

        dt_start = dm_fuel / so
        dt_stop = dm_fuel / sc

        t_soi = soe_i + inj_del
        t_eoi = t_soi + doi - dt_stop

        eps11 = 500
        eps12 = 500
        eps21 = 500
        eps22 = 500

        m_i = (1e-9 * (so / 8) *
               ((ca.sqrt((t - t_soi)**2 + eps11) + (t - t_soi))**2 -
                (ca.sqrt((t - t_soi - dt_start)**2 + eps12) +
                 (t - t_soi - dt_start))**2) -
               1e-9 * (sc / 8) *
               ((ca.sqrt((t - t_eoi)**2 + eps21) + (t - t_eoi))**2 -
                (ca.sqrt((t - t_eoi - dt_stop)**2 + eps22) +
                 (t - t_eoi - dt_stop))**2))
        m_act += m_i

    return m_act, m_tot_acc
