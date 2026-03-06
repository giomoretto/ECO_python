"""Tests for model module"""

import numpy as np
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

from eco.model_casadi import ModelParameters
from eco.model_casadi.subfunctions.in_cylinder import cyl_vol, calc_kappa
from eco.model_casadi.subfunctions.ignition_delay import saturate_input
from eco.model_casadi.subfunctions.combustion import eval_comb_weighting_fun
from eco.model_casadi.ignition_delay_joerg import ignition_delay_joerg
from eco.model_casadi.algebraic_injector_model import algebraic_injector_model


def test_model_parameters():
    """Test ModelParameters initialization"""
    print("Testing ModelParameters...")
    par = ModelParameters()
    
    # Check engine geometry
    assert par.eng['vol_dis'] > 0, "Volume displacement should be positive"
    assert par.eng['epsilon'] == 16.5, "Compression ratio mismatch"
    assert par.eng['num_cyl'] == 4, "Number of cylinders mismatch"
    
    # Check thermodynamic properties
    assert par.thermo['fuel']['low_heat_val'] == 42.6e6, "LHV mismatch"
    assert abs(par.thermo['psi']['air']['O2'] - 0.21) < 1e-6, "O2 fraction mismatch"
    
    # Check specific gas constants
    assert 'air' in par.thermo['gas']['spec_r'], "Air spec_r missing"
    assert 'bg' in par.thermo['gas']['spec_r'], "Burnt gas spec_r missing"
    
    print("✓ ModelParameters test passed")


def test_cyl_vol():
    """Test cylinder volume calculation"""
    print("Testing cyl_vol...")
    par = ModelParameters()
    
    # Test at TDC (ca = 0)
    v_tdc, dv_tdc, stroke_tdc = cyl_vol(0, par.eng)
    expected_v_tdc = par.eng['vol_clear']
    assert abs(v_tdc - expected_v_tdc) < 1e-9, f"Volume at TDC mismatch: {v_tdc} vs {expected_v_tdc}"
    
    # Test at BDC (ca = 180)
    v_bdc, dv_bdc, stroke_bdc = cyl_vol(180, par.eng)
    expected_v_bdc = par.eng['vol_clear'] + par.eng['vol_dis']
    assert abs(v_bdc - expected_v_bdc) < 1e-7, f"Volume at BDC mismatch: {v_bdc} vs {expected_v_bdc}"
    
    # Check volume derivative is negative during compression
    v_10, dv_10, _ = cyl_vol(-10, par.eng)
    assert dv_10 < 0, "Volume derivative should be negative during compression"
    
    print("✓ cyl_vol test passed")


def test_saturate_input():
    """Test saturate_input function"""
    print("Testing saturate_input...")
    
    # Test smooth saturation
    u_min, u_max = 0, 10
    
    # Below minimum
    y = saturate_input(-5, u_min, u_max, 'smooth')
    assert y >= u_min - 0.1, "Saturated value should be near minimum"
    
    # Above maximum
    y = saturate_input(15, u_min, u_max, 'smooth')
    assert y <= u_max + 0.1, "Saturated value should be near maximum"
    
    # Within range
    y = saturate_input(5, u_min, u_max, 'smooth')
    assert abs(y - 5) < 0.1, "Value within range should remain approximately unchanged"
    
    # Test exact saturation
    y = saturate_input(np.array([-5, 5, 15]), u_min, u_max, 'exact')
    assert np.allclose(y, [0, 5, 10]), "Exact saturation failed"
    
    print("✓ saturate_input test passed")


def test_eval_comb_weighting_fun():
    """Test combustion weighting function"""
    print("Testing eval_comb_weighting_fun...")
    
    tau_min = 2.7e-4
    tau_max = 2.9e-4
    
    # Test below minimum -> should give gamma ≈ 0 (diffusive)
    gamma = eval_comb_weighting_fun(2.5e-4, tau_min, tau_max)
    assert gamma < 0.1, f"Gamma should be near 0 for low tau: {gamma}"
    
    # Test above maximum -> should give gamma ≈ 1 (premixed)
    gamma = eval_comb_weighting_fun(3.0e-4, tau_min, tau_max)
    assert gamma > 0.9, f"Gamma should be near 1 for high tau: {gamma}"
    
    # Test in middle -> should give gamma ≈ 0.5
    gamma = eval_comb_weighting_fun(2.8e-4, tau_min, tau_max)
    assert 0.3 < gamma < 0.7, f"Gamma should be near 0.5 for middle tau: {gamma}"
    
    print("✓ eval_comb_weighting_fun test passed")


def test_calc_kappa():
    """Test kappa calculation"""
    print("Testing calc_kappa...")
    par = ModelParameters()
    
    # Test with air composition
    xi = {'N2': 0.77, 'O2': 0.23, 'CO2': 0, 'H2O': 0, 'CxHy': 0}
    theta = 600  # K
    kappa = calc_kappa(xi, theta, par.thermo['gas']['spec_r'])
    
    # For air at moderate temperature, kappa should be around 1.35-1.4
    assert 1.3 < kappa < 1.45, f"Kappa for air at 600K seems wrong: {kappa}"
    
    # Test at higher temperature
    kappa_high = calc_kappa(xi, 1500, par.thermo['gas']['spec_r'])
    assert kappa_high < kappa, "Kappa should decrease with temperature"
    
    print("✓ calc_kappa test passed")


def test_ignition_delay_joerg():
    """Test ignition delay calculation"""
    print("Testing ignition_delay_joerg...")
    
    # Typical conditions
    p = 80e5  # Pa
    T = 1000  # K
    x_o2 = 0.21
    
    tau = ignition_delay_joerg(p, T, x_o2)
    
    # Ignition delay should be positive and reasonable (milliseconds range)
    assert tau > 0, "Ignition delay should be positive"
    assert 0.01 < tau < 100, f"Ignition delay seems unreasonable: {tau} ms"
    
    # Higher pressure should reduce ignition delay
    tau_high_p = ignition_delay_joerg(120e5, T, x_o2)
    assert tau_high_p < tau, "Higher pressure should reduce ignition delay"
    
    # Higher temperature should reduce ignition delay
    tau_high_t = ignition_delay_joerg(p, 1200, x_o2)
    assert tau_high_t < tau, "Higher temperature should reduce ignition delay"
    
    print("✓ ignition_delay_joerg test passed")


def test_algebraic_injector_model():
    """Test algebraic injector model"""
    print("Testing algebraic_injector_model...")
    
    # Test single injection
    t = np.array([0, 100, 200, 300, 400, 500])  # µs
    p_rail = 1000e5  # Pa
    soe = np.array([50])  # µs
    doe = np.array([200])  # µs
    
    m_act, m_tot, doi = algebraic_injector_model(t, p_rail, soe, doe)
    
    # Check that total mass is positive
    assert m_tot > 0, "Total mass should be positive"
    
    # Check that mass trajectory is monotonically increasing or constant
    assert np.all(np.diff(m_act) >= -1e-10), "Mass should be non-decreasing"
    
    # Check that final mass doesn't exceed total mass significantly
    assert m_act[-1] <= m_tot * 1.1, "Final mass shouldn't exceed total mass"
    
    # Test multiple injections
    soe_multi = np.array([50, 250])
    doe_multi = np.array([150, 150])
    m_act_multi, m_tot_multi, _ = algebraic_injector_model(t, p_rail, soe_multi, doe_multi)
    
    # Multiple injections should give more total mass (approximately 2x for similar DOE)
    assert m_tot_multi > m_tot * 0.8, "Multiple injections should give more mass"
    
    print("✓ algebraic_injector_model test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Model Module Tests")
    print("=" * 60)
    
    try:
        test_model_parameters()
        test_cyl_vol()
        test_saturate_input()
        test_eval_comb_weighting_fun()
        test_calc_kappa()
        test_ignition_delay_joerg()
        test_algebraic_injector_model()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
