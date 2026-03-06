"""Tests for simulation module"""

import numpy as np
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

from eco.model_casadi import ModelParameters
from eco.simulation import par_op_def, complete_simulation


def test_par_op_def():
    """Test operating point definition"""
    print("Testing par_op_def...")
    par_model = ModelParameters()
    
    # Test with default values
    par_op = par_op_def(par_model)
    
    # Check that required attributes exist
    assert hasattr(par_op, 'eng_spd'), "Missing eng_spd attribute"
    assert hasattr(par_op, 'p_im'), "Missing p_im attribute"
    assert hasattr(par_op, 'm_cyl_tot'), "Missing m_cyl_tot attribute"
    assert hasattr(par_op, 'xi_air'), "Missing xi_air attribute"
    assert hasattr(par_op, 'kappa_ivc'), "Missing kappa_ivc attribute"
    
    # Check physical plausibility
    assert par_op.m_cyl_tot > 0, "Total cylinder mass should be positive"
    assert 0 < par_op.xi_air < 1, "Air mass fraction should be between 0 and 1"
    assert 1 < par_op.kappa_ivc < 1.5, f"Kappa at IVC seems wrong: {par_op.kappa_ivc}"
    assert par_op.theta_ivc > 250, "Temperature at IVC seems too low"
    
    print("✓ par_op_def test passed")


def test_complete_simulation_basic():
    """Test complete simulation with basic setup"""
    print("Testing complete_simulation (basic)...")
    
    # Setup model and operating point
    par_model = ModelParameters()
    par_model.n_states = 3
    par_model.n_outputs = 8
    par_model.n_inputs = 2
    
    par_op = par_op_def(par_model)
    
    # Setup simulation parameters
    class ParSim:
        pass
    
    par_sim = ParSim()
    par_sim.op = par_op
    par_sim.opts = {'delta_phi': 2.0, 'int': 'RK4', 'n_int': 1}
    
    # Create crank angle vector (small range during compression)
    ca = np.arange(-172, -160, 2.0)
    
    # Initial conditions
    x0 = np.array([par_op.p_int, 0, 0])
    
    # Input trajectory (injection far in future, so no combustion in this range)
    u = np.tile(np.array([50, 80]), (len(ca), 1)).T
    
    # Run simulation
    try:
        simout = complete_simulation(ca, x0, u, par_sim, par_model)
        
        # Check outputs
        assert 'ca' in simout, "Missing 'ca' in output"
        assert 'x' in simout, "Missing 'x' in output"
        assert 'y' in simout, "Missing 'y' in output"
        
        # Check dimensions
        assert simout['x'].shape == (3, len(ca)), f"Wrong state shape: {simout['x'].shape}"
        assert simout['y'].shape == (8, len(ca)), f"Wrong output shape: {simout['y'].shape}"
        
        # Check that the simulation completed all steps
        assert len(simout['ca']) == len(ca), "Simulation didn't complete"
        
        # Print diagnostics
        p_cyl = simout['x'][0, :]
        print(f"  Pressure range: {p_cyl.min()/1e5:.2f} - {p_cyl.max()/1e5:.2f} bar")
        q_comb = simout['x'][1, :]
        print(f"  Heat release range: {q_comb.min():.2e} - {q_comb.max():.2e} J")
        
        print("✓ complete_simulation (basic) test passed")
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all simulation tests"""
    print("=" * 60)
    print("Running Simulation Module Tests")
    print("=" * 60)
    
    try:
        test_par_op_def()
        test_complete_simulation_basic()
        
        print("=" * 60)
        print("✓ All simulation tests passed!")
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
