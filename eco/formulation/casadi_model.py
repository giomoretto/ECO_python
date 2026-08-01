"""Bridge module: expose the CasADi symbolic engine model for OCP formulation."""

from eco.model_casadi.complete_model import complete_model


def complete_model_sym(x, u, ca_deg, par_model, par_op, en_nox=True):
    """Symbolic CasADi model for use inside acados OCP.

    Thin wrapper around eco.model_casadi.complete_model.complete_model.
    """
    return complete_model(x, u, ca_deg, par_model, par_op, en_nox=en_nox)
