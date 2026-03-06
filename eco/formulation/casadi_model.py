"""CasADi symbolic model for acados OCP formulation

This module provides the ``complete_model_sym`` entry-point used by the
acados OCP builder.  The actual CasADi model implementation now lives in
``eco.model_casadi``; this file is a thin compatibility wrapper.
"""

from eco.model_casadi.complete_model import complete_model as complete_model_sym
