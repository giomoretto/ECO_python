"""Complete simulation using AcadosSim integrator.

This module provides the same functionality as ``complete_simulation``
but uses the acados ``AcadosSimSolver`` (compiled C integrator) instead
of a hand-coded Python RK4/Euler loop.
"""

import numpy as np
from typing import Dict, Any

from acados_template import AcadosSim, AcadosSimSolver

from eco.model.model_parameters import ModelParameters
from eco.model_casadi.export_complete_model import export_complete_model
from eco.model_casadi.complete_model import complete_model as complete_model_casadi
import casadi as ca


def create_acados_sim(par_model: ModelParameters,
                      par_op: Any,
                      n_inj: int = 1,
                      en_nox: bool = True,
                      delta_phi: float = 0.5,
                      n_int: int = 1,
                      integrator: str = 'RK4') -> AcadosSimSolver:
    """Build and return an ``AcadosSimSolver`` for the engine model.

    Args:
        par_model: Model parameters.
        par_op: Operating-point object.
        n_inj: Number of injections.
        en_nox: Include NOx states.
        delta_phi: Crank-angle step size [degCA] (integration interval).
        n_int: Number of integrator sub-steps per interval.
        integrator: ``'RK4'`` (explicit RK4) or ``'EulerFW'`` (explicit Euler).

    Returns:
        Compiled ``AcadosSimSolver``.
    """
    model = export_complete_model(par_model, par_op, n_inj=n_inj,
                                  en_nox=en_nox)

    sim = AcadosSim()
    sim.model = model

    # Integration interval length (in crank-angle degrees)
    sim.solver_options.T = delta_phi

    # Integrator type: explicit Runge-Kutta
    sim.solver_options.integrator_type = 'ERK'

    if integrator == 'EulerFW':
        sim.solver_options.num_stages = 1   # 1 stage = explicit Euler
    else:  # 'RK4' or default
        sim.solver_options.num_stages = 4   # 4 stages = classic RK4

    sim.solver_options.num_steps = n_int

    return AcadosSimSolver(sim)


def acados_simulation(ca_vec: np.ndarray,
                      x0: np.ndarray,
                      u: np.ndarray,
                      par_sim: Any,
                      par_model: ModelParameters) -> Dict[str, np.ndarray]:
    """Run a complete engine simulation with the acados integrator.

    Drop-in replacement for ``complete_simulation`` – same arguments, same
    return dictionary.

    Args:
        ca_vec: Crank-angle vector [degCA] of length *K*.
        x0: Initial state vector (n_states,).
        u: Input matrix (n_inputs, K).
        par_sim: Simulation parameters with fields
                 ``par_sim.op`` (OperatingPoint) and
                 ``par_sim.opts`` (dict with ``'delta_phi'``, ``'int'``,
                 ``'n_int'``).
        par_model: ``ModelParameters`` instance.

    Returns:
        Dictionary ``{'ca', 'x', 'u', 'y', 'xdot'}`` with arrays shaped
        (n_states/n_outputs, K).
    """
    par_op = par_sim.op
    n_inj = par_model.n_inputs // 2
    en_nox = par_model.n_states > 3

    # Read simulation settings
    delta_phi = par_sim.opts.get('delta_phi', 0.5)
    integrator = par_sim.opts.get('int', 'RK4')
    n_int = par_sim.opts.get('n_int', 1)

    # Build acados integrator
    sim_solver = create_acados_sim(
        par_model, par_op,
        n_inj=n_inj, en_nox=en_nox,
        delta_phi=delta_phi, n_int=n_int,
        integrator=integrator,
    )

    # Build a CasADi function for the algebraic outputs y
    nx = par_model.n_states
    x_sym = ca.SX.sym('x', nx)
    u_sym = ca.SX.sym('u', par_model.n_inputs)
    phi_sym = ca.SX.sym('phi')
    _, y_expr = complete_model_casadi(x_sym, u_sym, phi_sym,
                                      par_model, par_op, en_nox=en_nox)
    f_output = ca.Function('f_output', [x_sym, u_sym, phi_sym], [y_expr])

    # The acados model has augmented states: x_aug = [x_phys; phi]
    nx_aug = nx + 1

    K = len(ca_vec)
    x_traj = np.zeros((nx, K))
    y_traj = np.zeros((par_model.n_outputs, K))
    xdot_traj = np.zeros((nx, K))
    x_traj[:, 0] = x0

    for kk in range(K):
        x_k = x_traj[:, kk]
        u_k = u[:, kk]
        ca_k = ca_vec[kk]

        # Compute outputs and xdot at current point
        y_k = np.array(f_output(x_k, u_k, ca_k)).flatten()
        y_traj[:, kk] = y_k

        # Determine step size for this interval
        if kk < K - 1:
            dt = ca_vec[kk + 1] - ca_vec[kk]
        else:
            dt = ca_vec[kk] - ca_vec[kk - 1]

        # Integrate one step (augmented state includes phi)
        x_aug = np.append(x_k, ca_k)
        sim_solver.set('T', dt)
        sim_solver.set('x', x_aug)
        sim_solver.set('u', u_k)
        status = sim_solver.solve()
        if status != 0:
            raise RuntimeError(
                f'AcadosSimSolver returned status {status} at CA={ca_k:.2f}')
        x_next_aug = sim_solver.get('x')
        x_next = x_next_aug[:nx]  # strip phi from result

        # Store xdot ≈ (x_next - x_k) / dt
        xdot_traj[:, kk] = (x_next - x_k) / dt

        if kk < K - 1:
            x_traj[:, kk + 1] = x_next

    return {
        'ca': ca_vec,
        'x': x_traj,
        'u': u,
        'y': y_traj,
        'xdot': xdot_traj,
    }

