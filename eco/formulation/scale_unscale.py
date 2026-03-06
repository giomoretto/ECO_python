"""Scale and unscale variables for optimization

This module handles variable scaling/unscaling for the optimization problem.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def scale_unscale(unparsed_vars: np.ndarray, var_names: List[str],
                 opt_opts: Any, scale_char: str,
                 is_grad: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale or unscale variables based on optimization options
    
    Args:
        unparsed_vars: Variables to be scaled/unscaled
        var_names: Names of the variables
        opt_opts: Optimization options containing scaling information
        scale_char: 'scale' or 'unscale'
        is_grad: If True, only apply scaling (no offset) - for gradients
    
    Returns:
        parsed_vars: Scaled/unscaled variables
        scale_vec: Scaling vector
        offs_vec: Offset vector
    """
    unparsed_vars = np.atleast_1d(unparsed_vars)
    
    if hasattr(opt_opts, 'scale_offs_vars') and opt_opts.scale_offs_vars:
        # Get Opts
        unscale_opts_vars = opt_opts.scale_offs_vars
        unscale_opts_offs = opt_opts.offs
        unscale_opts_scale = opt_opts.scale
        
        # Initialize vectors
        scale_vec = np.ones_like(unparsed_vars, dtype=float)
        offs_vec = np.zeros_like(unparsed_vars, dtype=float)
        
        # Apply scaling for each variable
        for i, var_name in enumerate(var_names):
            if var_name in unscale_opts_vars:
                idx = unscale_opts_vars.index(var_name)
                scale_vec[i] = unscale_opts_scale[idx]
                offs_vec[i] = unscale_opts_offs[idx]
        
        # Scale or unscale
        if scale_char == 'unscale':
            if is_grad:
                parsed_vars = unparsed_vars * scale_vec
                offs_vec = np.zeros_like(unparsed_vars)
            else:
                parsed_vars = unparsed_vars * scale_vec + offs_vec
        else:  # scale
            if is_grad:
                parsed_vars = unparsed_vars / scale_vec
                offs_vec = np.zeros_like(unparsed_vars)
            else:
                parsed_vars = (unparsed_vars - offs_vec) / scale_vec
    else:
        # No scaling
        parsed_vars = unparsed_vars
        scale_vec = np.ones_like(unparsed_vars)
        offs_vec = np.zeros_like(unparsed_vars)
    
    return parsed_vars, scale_vec, offs_vec
