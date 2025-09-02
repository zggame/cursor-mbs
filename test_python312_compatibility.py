#!/usr/bin/env python3
"""
Python 3.12 Compatibility Test
This script tests that all imports and basic functionality work with Python 3.12
"""

import sys
import importlib

def test_python_version():
    """Test that we're running Python 3.12+"""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 12):
        raise RuntimeError(f"Python 3.12+ required, found {major}.{minor}")
    print("✓ Python version check passed")

def test_imports():
    """Test that all required packages can be imported"""
    required_packages = [
        'numpy',
        'matplotlib',
        'scipy',
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            raise ImportError(f"Failed to import {package}: {e}")

def test_mbs_simulation():
    """Test that the MBS simulation can be imported and basic functionality works"""
    try:
        from mbs_simulation import MBSSimulation, MBSVisualizer
        print("✓ MBS simulation imported successfully")
        
        # Test basic functionality
        sim = MBSSimulation(
            n_loans=5,  # Small number for quick test
            correlation=0.3,
            attachment_point=0.01,
            detachment_point=0.05,
            security_term_years=7
        )
        print("✓ MBS simulation created successfully")
        
        # Test Monte Carlo simulation
        results = sim.run_monte_carlo(n_simulations=10, use_parallel=False)  # Small number for test
        print("✓ Monte Carlo simulation completed")
        
        # Test visualizer
        viz = MBSVisualizer(sim)
        print("✓ Visualizer created successfully")
        
    except Exception as e:
        raise RuntimeError(f"MBS simulation test failed: {e}")

def main():
    """Run all compatibility tests"""
    print("PYTHON 3.12 COMPATIBILITY TEST")
    print("=" * 50)
    
    try:
        test_python_version()
        test_imports()
        test_mbs_simulation()
        
        print("\n" + "=" * 50)
        print("ALL COMPATIBILITY TESTS PASSED! ✓")
        print("=" * 50)
        print("\nThe codebase is compatible with Python 3.12")
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
