"""In-cylinder model (CasADi version)"""

import casadi as ca
from .subfunctions.in_cylinder import (
    cyl_vol,
    static_cylinder_conditions,
    dynamic_cylinder_conditions,
)


def in_cylinder_model(x, u, ca_deg, par_model, par_op):
    """In-Cylinder Model calculating state derivatives (CasADi symbolic).

    Args:
        x: CasADi SX states [p_cyl [Pa], q_comb [J], IMEP [Pa]]
        u: Input dQcomb [J/degCA]
        ca_deg: Crank angle [degCA]
        par_model: Model parameters
        par_op: Operating point parameters

    Returns:
        xdot: State derivatives [stateUnit/degCA]
        y: Outputs [theta_cyl [K], kappa [-], dq_wall [J/degCA], spec_r [J/(kg*K)]]
    """
    p_cyl = x[0]
    q_comb = x[1]

    v_cyl, dv_cyl, stroke = cyl_vol(ca_deg, par_model.eng)

    kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb = \
        static_cylinder_conditions(q_comb, p_cyl, v_cyl, par_model, par_op)

    dq_comb = u

    dp_cyl, dimep, dq_wall = dynamic_cylinder_conditions(
        ca_deg, p_cyl, dq_comb, v_cyl, dv_cyl, stroke,
        theta_cyl, kappa, spec_r, par_model, par_op)

    xdot = ca.vertcat(dp_cyl, dq_comb, dimep)
    y = ca.vertcat(theta_cyl, kappa, dq_wall, spec_r)

    return xdot, y
