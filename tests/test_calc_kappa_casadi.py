"""Test for calc_kappa_casadi function"""

import numpy as np
import pytest

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False

from eco.model.subfunctions.in_cylinder import calc_kappa, calc_kappa_casadi


@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not available")
def test_calc_kappa_casadi_numerical():
    """Test calc_kappa_casadi with numerical values"""
    
    # Define test inputs
    xi = {
        'N2': 0.75,
        'O2': 0.21,
        'CO2': 0.03,
        'H2O': 0.01,
        'CxHy': 0.0
    }
    
    spec_r = {
        'N2': 296.8,
        'O2': 259.8,
        'CO2': 188.9,
        'H2O': 461.5,
        'CxHy': 82.7
    }
    
    # Test at high temperature (theta > 1000)
    theta_high = 1500.0
    kappa_casadi_high = calc_kappa_casadi(xi, theta_high, spec_r)
    kappa_numpy_high = calc_kappa(xi, theta_high, spec_r)
    
    # Convert CasADi result to float
    if isinstance(kappa_casadi_high, (ca.SX, ca.MX, ca.DM)):
        kappa_casadi_high = float(kappa_casadi_high)
    
    # Should be close to numpy version
    assert abs(kappa_casadi_high - kappa_numpy_high) < 1e-6, \
        f"High temp: CasADi={kappa_casadi_high}, NumPy={kappa_numpy_high}"
    
    # Test at low temperature (theta <= 1000)
    theta_low = 800.0
    kappa_casadi_low = calc_kappa_casadi(xi, theta_low, spec_r)
    kappa_numpy_low = calc_kappa(xi, theta_low, spec_r)
    
    # Convert CasADi result to float
    if isinstance(kappa_casadi_low, (ca.SX, ca.MX, ca.DM)):
        kappa_casadi_low = float(kappa_casadi_low)
    
    # Should be close to numpy version
    assert abs(kappa_casadi_low - kappa_numpy_low) < 1e-6, \
        f"Low temp: CasADi={kappa_casadi_low}, NumPy={kappa_numpy_low}"
    
    print(f"High temp (1500K): CasADi={kappa_casadi_high:.6f}, NumPy={kappa_numpy_high:.6f}")
    print(f"Low temp (800K): CasADi={kappa_casadi_low:.6f}, NumPy={kappa_numpy_low:.6f}")


@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not available")
def test_calc_kappa_casadi_symbolic():
    """Test calc_kappa_casadi with symbolic variables"""
    
    # Define symbolic temperature
    theta_sym = ca.SX.sym('theta')
    
    # Define symbolic mass fractions
    xi_sym = {
        'N2': ca.SX.sym('xi_N2'),
        'O2': ca.SX.sym('xi_O2'),
        'CO2': ca.SX.sym('xi_CO2'),
        'H2O': ca.SX.sym('xi_H2O'),
        'CxHy': ca.SX.sym('xi_CxHy')
    }
    
    spec_r = {
        'N2': 296.8,
        'O2': 259.8,
        'CO2': 188.9,
        'H2O': 461.5,
        'CxHy': 82.7
    }
    
    # Calculate kappa symbolically
    kappa_sym = calc_kappa_casadi(xi_sym, theta_sym, spec_r)
    
    # Should be a symbolic expression
    assert isinstance(kappa_sym, (ca.SX, ca.MX)), \
        f"Expected symbolic result, got {type(kappa_sym)}"
    
    # Create a CasADi function
    kappa_func = ca.Function('kappa', 
                            [theta_sym, xi_sym['N2'], xi_sym['O2'], 
                             xi_sym['CO2'], xi_sym['H2O'], xi_sym['CxHy']],
                            [kappa_sym])
    
    # Evaluate at a test point
    theta_test = 1500.0
    xi_test = [0.75, 0.21, 0.03, 0.01, 0.0]
    kappa_result = float(kappa_func(theta_test, *xi_test))
    
    # Compare with numerical version
    xi_dict = {
        'N2': 0.75,
        'O2': 0.21,
        'CO2': 0.03,
        'H2O': 0.01,
        'CxHy': 0.0
    }
    kappa_numpy = calc_kappa(xi_dict, theta_test, spec_r)
    
    assert abs(kappa_result - kappa_numpy) < 1e-6, \
        f"Symbolic evaluation: CasADi={kappa_result}, NumPy={kappa_numpy}"
    
    print(f"Symbolic function created successfully")
    print(f"Test evaluation: CasADi={kappa_result:.6f}, NumPy={kappa_numpy:.6f}")


@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not available")
def test_calc_kappa_casadi_gradient():
    """Test that calc_kappa_casadi can compute gradients"""
    
    # Define symbolic temperature
    theta_sym = ca.SX.sym('theta')
    
    # Define test mass fractions (numerical)
    xi = {
        'N2': 0.75,
        'O2': 0.21,
        'CO2': 0.03,
        'H2O': 0.01,
        'CxHy': 0.0
    }
    
    spec_r = {
        'N2': 296.8,
        'O2': 259.8,
        'CO2': 188.9,
        'H2O': 461.5,
        'CxHy': 82.7
    }
    
    # Calculate kappa symbolically
    kappa_sym = calc_kappa_casadi(xi, theta_sym, spec_r)
    
    # Compute gradient
    grad_kappa = ca.gradient(kappa_sym, theta_sym)
    
    # Create function for gradient
    grad_func = ca.Function('grad_kappa', [theta_sym], [grad_kappa])
    
    # Evaluate gradient at test point
    theta_test = 1500.0
    grad_value = float(grad_func(theta_test))
    
    # Gradient should be non-zero and reasonable
    assert abs(grad_value) > 1e-10, "Gradient should be non-zero"
    assert abs(grad_value) < 1.0, "Gradient should be reasonable magnitude"
    
    print(f"Gradient at theta={theta_test}K: {grad_value:.10f}")
    print("Gradient computation successful!")


if __name__ == "__main__":
    if CASADI_AVAILABLE:
        print("Testing calc_kappa_casadi with numerical values...")
        test_calc_kappa_casadi_numerical()
        print("\nTesting calc_kappa_casadi with symbolic variables...")
        test_calc_kappa_casadi_symbolic()
        print("\nTesting gradient computation...")
        test_calc_kappa_casadi_gradient()
        print("\nAll tests passed!")
    else:
        print("CasADi not available, skipping tests")
