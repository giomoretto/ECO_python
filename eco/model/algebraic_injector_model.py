"""Algebraic injector model"""

import numpy as np
from typing import Tuple


def algebraic_injector_model(t: np.ndarray, p_rail: float, 
                             soe: np.ndarray, doe: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Calculate injected fuel mass trajectory
    
    Args:
        t: Time vector [µs]
        p_rail: Rail pressure [Pa]
        soe: Start of injector energizing [µs]
        doe: Duration of injector energizing [µs]
    
    Returns:
        m_act: Total injected fuel mass at time t [kg]
        m_tot: Total injected fuel mass over all injections [kg]
        doi: Duration of injection [µs]
    """
    # Convert to arrays
    t = np.atleast_1d(t)
    soe = np.atleast_1d(soe)
    doe = np.atleast_1d(doe)
    
    # Parameter Definition for injection rate model
    rho_diesel = 830
    d_noz = 120 * 10**-6
    n_noz = 8
    a_noz_tot = (d_noz / 2)**2 * np.pi * n_noz
    inj_del = 55  # [µs] SOE to SOI delay (hydraulic delay)
    doe_min = 66.73  # [µs]
    
    # Limit DOE and convert to s
    eps_lim = 1
    doe_lim = 0.5 * np.sqrt((doe - doe_min)**2 + eps_lim) + 0.5 * (doe - doe_min) + doe_min
    
    # Injection rate model
    # Bernoulli speed
    p_cyl = 60e5
    delta_p = p_rail - p_cyl
    u_bernoulli = np.sqrt(2 * delta_p / rho_diesel)
    
    # Injector opening & closing rate gradient [g/s / µs]
    # Rail pressure dependent correction factor
    corr_fct_s = 3.006 - 9090 * p_rail**-0.4791
    
    # Gradient at injector opening (g/s *1/µs)
    so = 100 * corr_fct_s * 1e-3
    
    # Gradient at injector closing (g/s *1/µs)
    sc = 200 * corr_fct_s * 1e-3
    
    # Real (physical) duration of injection [µs]
    doi = (doe_lim - doe_min) * 2.093
    
    # Discharge coefficient
    corr_fct_cd = 9.8e-9 * p_rail + 0.425
    doi_lim = 0.5 * np.sqrt((doi - 70)**2 + eps_lim) + 0.5 * (doi - 70) + 70
    coeff_d = (0.9301 - 0.02744 * (doi_lim * 1e-6)**-0.3676) * corr_fct_cd
    
    # Limit discharge coefficient around doe = 100 µs
    doi_lower = 70
    eps = 1e-2
    coeff_d_lim = (0.9301 - 0.02744 * (doi_lower * 1e-6)**-0.3676) * corr_fct_cd
    coeff_d = 0.5 * (coeff_d + coeff_d_lim + np.sqrt(eps + (coeff_d - coeff_d_lim)**2))
    
    # Injection rate at fully open injector [kg/s]
    dm_fuel = coeff_d * u_bernoulli * a_noz_tot * rho_diesel * 1e3
    
    # Limitation of flow rate for not completely open injectors
    dm_fuel_max = doi * (so * sc) / (so + sc)
    eps = 1e-2
    dm_fuel = dm_fuel_max - 0.5 * np.sqrt((dm_fuel - dm_fuel_max)**2 + eps) + 0.5 * (dm_fuel - dm_fuel_max)
    
    # Complete fuel injection value [kg]
    cost_type = 'exact'
    if cost_type == 'exact':
        m_tot = np.sum(1e-9 * (doi * dm_fuel - 0.5 * dm_fuel**2 * (1 / so + 1 / sc)))
    elif cost_type == 'quadratic':
        m_tot = 1e-6 * np.sum(0.02767 * (doe - 100) + 9.004e-5 * (doe - 100)**2)
    elif cost_type == 'linear':
        m_tot = 1e-6 * np.sum(0.04286 * (doe - 100))
    else:
        m_tot = np.sum(1e-9 * (doi * dm_fuel - 0.5 * dm_fuel**2 * (1 / so + 1 / sc)))
    
    # Duration of Opening and Closing
    dt_start = dm_fuel / so
    dt_stop = dm_fuel / sc
    
    # Derive SOI and EOI [µs]
    t_soi = soe + inj_del
    t_eoi = t_soi + doi - dt_stop
    
    # Derive smoothness of function
    eps11 = 500
    eps12 = 500
    eps21 = 500
    eps22 = 500
    
    # Fuel mass trajectory based on four "integrated" limit functions [kg]
    m = np.zeros_like(t, dtype=float)
    for i in range(len(soe)):
        m_i = (1e-9 * (so / 8) *
              ((np.sqrt((t - t_soi[i])**2 + eps11) + (t - t_soi[i]))**2 -
               (np.sqrt((t - t_soi[i] - dt_start[i])**2 + eps12) + 
                (t - t_soi[i] - dt_start[i]))**2) -
              1e-9 * (sc / 8) *
              ((np.sqrt((t - t_eoi[i])**2 + eps21) + (t - t_eoi[i]))**2 -
               (np.sqrt((t - t_eoi[i] - dt_stop[i])**2 + eps22) + 
                (t - t_eoi[i] - dt_stop[i]))**2))
        m += m_i
    
    # Sum up all masses of individual injections
    m_act = m
    
    # Return scalar if input was scalar
    if m_act.size == 1:
        m_act = float(m_act[0])
    
    return m_act, m_tot, doi
