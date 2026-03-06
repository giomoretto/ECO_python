"""CasADi engine model module

This module contains the physical models for engine combustion simulation,
using CasADi symbolic operations for compatibility with automatic
differentiation and optimal control solvers (e.g. acados).
"""

from eco.model.model_parameters import ModelParameters
from .complete_model import complete_model
from .in_cylinder_model import in_cylinder_model
from .combustion_model import combustion_model
from .algebraic_injector_model import algebraic_injector_model
from .ign_del_model import ign_del_model
from .ignition_delay_joerg import ignition_delay_joerg

__all__ = [
    'ModelParameters',
    'complete_model',
    'in_cylinder_model',
    'combustion_model',
    'algebraic_injector_model',
    'ign_del_model',
    'ignition_delay_joerg',
]
