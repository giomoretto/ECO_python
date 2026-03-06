"""Complete engine model including combustion and NOx"""

import numpy as np
from typing import Tuple, Any
from .subfunctions.in_cylinder import (
    cyl_vol,
    static_cylinder_conditions,
    dynamic_cylinder_conditions
)
from .subfunctions.nox import two_zone_model, nox_model
from .algebraic_injector_model import algebraic_injector_model
from .ign_del_model import ign_del_model
from .combustion_model import combustion_model
from .utils import check_validity


def complete_model(x: np.ndarray, u: np.ndarray, ca: float,
                   par_model: Any, par_op: Any,
                   num_sym: str = 'Num') -> Tuple[np.ndarray, np.ndarray]:
    """Complete in-cylinder model with combustion and optional NOx
    
    Args:
        x: States [1)p_cyl [Pa], 2)q_comb [J], 3)IMEP [Pa], 4)theta_uz [K], 5)NO [mole fraction in BG]]
        u: Input [SOE [degCA ATDC], DOE [µs]]
        ca: Crank angle [degCA]
        par_model: Model parameters
        par_op: Operating point parameters
        num_sym: 'Num' for numerical, 'Sym' for symbolic
    
    Returns:
        xdot: State derivatives [stateUnit/degCA]
        y: Outputs (see function docstring)
    """
    # Sanity checks - Limit states to physical boundaries
    x = check_validity(x, ca, par_model, par_op)
    
    # Define Inputs and States
    n_inj = len(u) // 2
    soe = u[:n_inj]
    doe = u[n_inj:2*n_inj]
    
    # States
    p_cyl = x[0]
    q_comb = x[1]
    # imep = x[2]
    if len(x) > 3:
        theta_uz = x[3]
        no = x[4]
    
    # Derive Actual Volume and Volume Derivative
    # v_cyl: [m^3] / dv_cyl: [m^3/degCA] / stroke: [m]
    v_cyl, dv_cyl, stroke = cyl_vol(ca, par_model.eng)
    
    # Derive kappa, specific gas constant and Temperature
    kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb = static_cylinder_conditions(
        q_comb, p_cyl, v_cyl, par_model, par_op, num_sym)
    
    # Derive Ignition Delay [s]
    tau_ign = ign_del_model(p_cyl, theta_cyl, xi_o2, par_model)
    
    # Derive Prepared Fuel Mass [kg]
    # Actual crank angle minus ignition delay, converted to microseconds after TDC
    t_prep = (ca / (par_op.eng_spd * 360) - tau_ign) * 1e6
    
    # Start of Energizing converted to microseconds after TDC
    t_soe = soe / (par_op.eng_spd * 360) * 1e6
    
    # Injected fuel at actual crank angle minus ignition delay [kg]
    m_fuel_prep, m_tot, _ = algebraic_injector_model(t_prep, par_op.p_rail, t_soe, doe)
    
    # Calculate Phi = 1/lambda
    phi = (m_fuel_prep * par_model.thermo['fuel']['air_fuel_ratio_st'] * 
          par_model.thermo['xi']['air']['O2'] /
          par_op.m_cyl_tot / par_op.xi_o2)
    
    # Derive Heat Release Rate [J/degCA]
    dq_comb = combustion_model(q_comb, m_fuel_prep, tau_ign, par_op.eng_spd, par_model)
    
    # Wall Heat Transfer, Cylinder Pressure Derivative, and IMEP Derivative
    dp_cyl, dimep, dq_wall = dynamic_cylinder_conditions(
        ca, p_cyl, dq_comb, v_cyl, dv_cyl, stroke,
        theta_cyl, kappa, spec_r, par_model, par_op)
    
    if len(x) > 3:
        # Derive dThetaUZ and thetaBZ [K] from a Two-Zone Model
        d_theta_uz, theta_bz = two_zone_model(
            q_comb, theta_uz, dp_cyl, p_cyl, dv_cyl, v_cyl,
            theta_cyl, kappa, m_tot, par_op, par_model, x_bg, x_bz)
        
        # Derive NOx concentration in the exhaust
        d_no, no_ppm, no_eq = nox_model(
            q_comb, theta_uz, theta_bz, no, m_tot, par_op, par_model,
            x_bg, v_cyl, dv_cyl, p_cyl, dp_cyl, theta_cyl)
    
    # Write Derivatives to xdot
    if len(x) > 3:
        xdot = np.array([dp_cyl, dq_comb, dimep, d_theta_uz, d_no])
        y = np.array([theta_cyl, kappa, dq_wall, spec_r, m_fuel_prep, 
                     tau_ign, x_bg, theta_bz, no_ppm, phi, no_eq, x_bz])
    else:
        xdot = np.array([dp_cyl, dq_comb, dimep])
        y = np.array([theta_cyl, kappa, dq_wall, spec_r, m_fuel_prep, 
                     tau_ign, phi, xi_o2])
    
    return xdot, y
