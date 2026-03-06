"""Export the complete engine model as an AcadosModel.

This module builds CasADi symbolic expressions for the engine ODE and
wraps them into an ``AcadosModel`` that can be used directly with
``AcadosOcp`` / ``AcadosOcpSolver``.
"""

import casadi as ca
from acados_template import AcadosModel
from eco.model_casadi.model_parameters import ModelParameters
from .complete_model import complete_model


def export_complete_model(par_model: ModelParameters,
                          par_op,
                          n_inj: int = 1,
                          en_nox: bool = True) -> AcadosModel:
    """Create an AcadosModel from the CasADi engine model.

    The independent variable is the crank angle *phi* [degCA aTDC].

    States
    ------
    * ``p_cyl``   – cylinder pressure [Pa]
    * ``q_comb``  – cumulative heat release [J]
    * ``imep``    – indicated mean effective pressure [Pa]
    * ``theta_uz`` – unburned-zone temperature [K]  (if *en_nox*)
    * ``no``      – NO mole-fraction in burned zone [-]  (if *en_nox*)

    Controls
    --------
    * ``soe_1 … soe_n`` – start of energising per injection [degCA aTDC]
    * ``doe_1 … doe_n`` – duration of energising per injection [µs]

    Args:
        par_model: ``ModelParameters`` instance.
        par_op: Operating-point object (carries ``eng_spd``, ``p_rail``, …).
        n_inj: Number of injections (default 1).
        en_nox: Include the two NOx-related states (default True).

    Returns:
        model: Populated ``AcadosModel`` ready for use with ``AcadosOcp``.
    """
    model_name = 'ECO_engine'

    # -------------------------------------------------------------- states
    if en_nox:
        nx = 5
        p_cyl    = ca.SX.sym('p_cyl')
        q_comb   = ca.SX.sym('q_comb')
        imep     = ca.SX.sym('imep')
        theta_uz = ca.SX.sym('theta_uz')
        no       = ca.SX.sym('no')
        x = ca.vertcat(p_cyl, q_comb, imep, theta_uz, no)
        x_labels = [
            r'$p_\mathrm{cyl}$ [Pa]',
            r'$Q_\mathrm{comb}$ [J]',
            r'IMEP [Pa]',
            r'$\theta_\mathrm{uz}$ [K]',
            r'NO [-]',
        ]
    else:
        nx = 3
        p_cyl  = ca.SX.sym('p_cyl')
        q_comb = ca.SX.sym('q_comb')
        imep   = ca.SX.sym('imep')
        x = ca.vertcat(p_cyl, q_comb, imep)
        x_labels = [
            r'$p_\mathrm{cyl}$ [Pa]',
            r'$Q_\mathrm{comb}$ [J]',
            r'IMEP [Pa]',
        ]

    # ------------------------------------------------------------ controls
    nu = 2 * n_inj
    soe = ca.SX.sym('soe', n_inj)
    doe = ca.SX.sym('doe', n_inj)
    u = ca.vertcat(soe, doe)
    u_labels = ([f'SOE_{i+1} [degCA]' for i in range(n_inj)] +
                [f'DOE_{i+1} [µs]' for i in range(n_inj)])

    # ----------------------------------------- crank-angle as extra state
    # acados ERK does not pass model.t into the generated CasADi function,
    # so we augment the state vector: x_aug = [x; phi] with dphi/dt = 1.
    phi = ca.SX.sym('phi')
    x_aug = ca.vertcat(x, phi)
    nx_aug = nx + 1

    # ------------------------------------------------------ explicit ODE
    f_expl_phys, y = complete_model(x, u, phi, par_model, par_op,
                                    en_nox=en_nox)
    # augmented ODE: [dx/dt; dphi/dt] = [f_expl_phys; 1.0]
    f_expl = ca.vertcat(f_expl_phys, 1.0)

    # ------------------------------------------------------ xdot & implicit
    xdot = ca.SX.sym('xdot', nx_aug)
    f_impl = xdot - f_expl

    # -------------------------------------------------- populate AcadosModel
    model = AcadosModel()
    model.name = model_name
    model.x = x_aug
    model.xdot = xdot
    model.u = u
    model.p = ca.SX.sym('p', 0)   # no parameters for now
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    # meta information
    model.x_labels = x_labels + [r'$\varphi$ [degCA aTDC]']
    model.u_labels = u_labels

    return model
