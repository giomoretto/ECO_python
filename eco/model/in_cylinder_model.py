"""In-cylinder model"""

import numpy as np
from typing import Tuple, Any
from .subfunctions.in_cylinder import (
    cyl_vol,
    static_cylinder_conditions,
    dynamic_cylinder_conditions
)


def in_cylinder_model(x: np.ndarray, u: float, ca: float,
                     par_model: Any, par_op: Any,
                     num_sym: str = 'Num') -> Tuple[np.ndarray, np.ndarray]:
    """In-Cylinder Model calculating state derivatives
    
    Args:
        x: States [p_cyl [Pa], q_comb [J], IMEP [Pa]]
        u: Input dQcomb [J/degCA]
        ca: Crank angle [degCA]
        par_model: Model parameters
        par_op: Operating point parameters
        num_sym: 'Num' for numerical, 'Sym' for symbolic
    
    Returns:
        xdot: State derivatives [stateUnit/degCA]
        y: Outputs [theta_cyl [K], kappa [-], dq_wall [J/degCA], spec_r [J/(kg*K)]]
    """
    # Define States
    p_cyl = x[0]
    q_comb = x[1]
    # imep = x[2]
    
    # Derive Actual Volume and Volume Derivative
    # v_cyl: [m^3] / dv_cyl: [m^3/degCA] / stroke: [m]
    v_cyl, dv_cyl, stroke = cyl_vol(ca, par_model.eng)
    
    # Derive kappa, specific gas constant and Temperature
    kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb = static_cylinder_conditions(
        q_comb, p_cyl, v_cyl, par_model, par_op, num_sym)
    
    # Derive Heat Release rate [J/degCA]
    dq_comb = u
    
    # Wall Heat Transfer [J/degCA], Cylinder Pressure Derivative [Pa/degCA]
    # and Indicated Mean Effective Pressure Derivative [Pa/degCA]
    dp_cyl, dimep, dq_wall = dynamic_cylinder_conditions(
        ca, p_cyl, dq_comb, v_cyl, dv_cyl, stroke,
        theta_cyl, kappa, spec_r, par_model, par_op)
    
    # Write Derivatives to xdot
    xdot = np.array([dp_cyl, dq_comb, dimep])
    y = np.array([theta_cyl, kappa, dq_wall, spec_r])
    
    return xdot, y
