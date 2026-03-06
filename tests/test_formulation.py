#!/usr/bin/env python3
"""Tests for formulation module"""

import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

import numpy as np
from eco.formulation.scale_unscale import scale_unscale


class MockOptParams:
    """Mock optimization parameters for testing"""
    def __init__(self):
        # Legacy scale dictionary (for reference)
        self.scale_dict = {
            'pCyl': 1e7,
            'QComb': 1e3,
            'IMEP': 1e3,
            'Theta': 1e3,
            'NO': 1e-5,
            'NOppm': 1e3,
            'SOE': 10,
            'DOE': 10
        }
        
        # Actual format expected by scale_unscale
        self.scale_offs_vars = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO', 'NOppm', 'SOE', 'DOE']
        self.scale = [1e7, 1e3, 1e3, 1e3, 1e-5, 1e3, 10, 10]
        self.offs = [0, 0, 0, 0, 0, 0, 0, 0]


def test_scale_unscale_values():
    """Test scaling and unscaling of values"""
    par_opt = MockOptParams()
    
    # Test with control variables
    u_orig = np.array([-10, -5, 15, 20])
    u_names = ['SOE', 'SOE', 'DOE', 'DOE']
    
    # Scale
    u_scaled, u_scale, u_offs = scale_unscale(u_orig, u_names, par_opt, 'scale', False)
    
    # Check scaling factors
    assert u_scale[0] == 10
    assert u_offs[0] == 0
    
    # Unscale
    u_unscaled, _, _ = scale_unscale(u_scaled, u_names, par_opt, 'unscale', False)
    
    # Check roundtrip
    np.testing.assert_allclose(u_orig, u_unscaled, rtol=1e-10)


def test_scale_unscale_states():
    """Test scaling of state variables"""
    par_opt = MockOptParams()
    
    # Test with state variables
    x_orig = np.array([1e7, 500, 800, 1500, 1e-5])
    x_names = ['pCyl', 'QComb', 'IMEP', 'Theta', 'NO']
    
    # Scale
    x_scaled, x_scale, x_offs = scale_unscale(x_orig, x_names, par_opt, 'scale', False)
    
    # Check that values are scaled to order of magnitude ~1
    assert np.all(np.abs(x_scaled) < 1000)
    
    # Unscale
    x_unscaled, _, _ = scale_unscale(x_scaled, x_names, par_opt, 'unscale', False)
    
    # Check roundtrip
    np.testing.assert_allclose(x_orig, x_unscaled, rtol=1e-10)


def test_scale_unscale_gradients():
    """Test scaling of gradients"""
    par_opt = MockOptParams()
    
    # Test gradient scaling
    grad_orig = np.array([1.0, 2.0])
    var_names = ['SOE', 'DOE']
    
    # Scale gradient
    grad_scaled, _, _ = scale_unscale(grad_orig, var_names, par_opt, 'scale', True)
    
    # Gradient scaling should be inverse of value scaling
    expected_scale = np.array([1/10, 1/10])
    expected_grad = grad_orig * expected_scale
    
    np.testing.assert_allclose(grad_scaled, expected_grad, rtol=1e-10)
    
    # Unscale gradient
    grad_unscaled, _, _ = scale_unscale(grad_scaled, var_names, par_opt, 'unscale', True)
    
    # Check roundtrip
    np.testing.assert_allclose(grad_orig, grad_unscaled, rtol=1e-10)


def test_scale_unscale_vector():
    """Test scaling of vector inputs"""
    par_opt = MockOptParams()
    
    # Test with multiple values
    values = np.array([1e7, 2e7, 1.5e7])
    names = ['pCyl', 'pCyl', 'pCyl']
    
    # Scale
    scaled, scale, offs = scale_unscale(values, names, par_opt, 'scale', False)
    
    # All should use same scaling factor
    assert len(set(scale)) == 1
    assert scale[0] == 1e7
    
    # Unscale
    unscaled, _, _ = scale_unscale(scaled, names, par_opt, 'unscale', False)
    
    np.testing.assert_allclose(values, unscaled, rtol=1e-10)


def test_scale_unscale_mixed_types():
    """Test scaling of mixed variable types"""
    par_opt = MockOptParams()
    
    # Mix of states and controls
    values = np.array([1e7, 500, -10, 15])
    names = ['pCyl', 'QComb', 'SOE', 'DOE']
    
    # Scale
    scaled, scale, offs = scale_unscale(values, names, par_opt, 'scale', False)
    
    # Check each has correct scaling
    assert scale[0] == 1e7  # pCyl
    assert scale[1] == 1e3  # QComb
    assert scale[2] == 10   # SOE
    assert scale[3] == 10   # DOE
    
    # Unscale
    unscaled, _, _ = scale_unscale(scaled, names, par_opt, 'unscale', False)
    
    np.testing.assert_allclose(values, unscaled, rtol=1e-10)


if __name__ == '__main__':
    print("Running formulation tests...")
    
    # Run all test functions
    test_functions = [
        test_scale_unscale_values,
        test_scale_unscale_states,
        test_scale_unscale_gradients,
        test_scale_unscale_vector,
        test_scale_unscale_mixed_types
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
