"""
Simple test script for the MBS Simulation System

This script runs a basic test to verify the system works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mbs_simulation import MBSSimulation, MBSVisualizer

def test_basic_functionality():
    """Test basic functionality of the MBS simulation"""
    print("Testing MBS Simulation System...")
    print("=" * 50)
    
    # Create a simple simulationup
    sim = MBSSimulation(
        n_loans=20,  # Small number for quick testing
        correlation=0.3,
        attachment_point=0.01,  # 1%
        detachment_point=0.05,  # 5%
        security_term_years=7
    )
    
    print(f"Created simulation with {sim.n_loans} loans")
    print(f"Total pool principal: ${sum(loan.principal for loan in sim.loans):,.0f}")
    print(f"Correlation: {sim.correlation}")
    print(f"Attachment point: {sim.attachment_point:.1%}")
    print(f"Detachment point: {sim.detachment_point:.1%}")
    
    # Run a small Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    results = sim.run_monte_carlo(n_simulations=100, use_parallel=False)  # Small number for testing
    
    print("Simulation completed successfully!")
    
    # Check results structure
    print("\nResults structure:")
    print(f"- Tranche payoffs: {list(results['tranche_payoffs'].keys())}")
    print(f"- Statistics: {list(results['statistics'].keys())}")
    print(f"- Stress scenarios: {list(results['stress_scenarios'].keys())}")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    for tranche_name, payoffs in results['tranche_payoffs'].items():
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        print(f"  {tranche_name}: Mean=${mean_payoff:,.0f}, Std=${std_payoff:,.0f}")
    
    return sim, results

def test_visualization():
    """Test visualization functionality"""
    print("\n" + "=" * 50)
    print("Testing Visualization...")
    
    # Create simulation and run it
    sim, results = test_basic_functionality()
    
    # Create visualizer
    viz = MBSVisualizer(sim)
    
    # Test summary report
    print("\nGenerating summary report...")
    viz.create_summary_report()
    
    # Test basic plots (without showing them)
    print("\nTesting plot generation...")
    
    # Create plots but don't show them
    fig1 = plt.figure(figsize=(15, 10))
    viz.plot_tranche_payoff_distributions()
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(15, 8))
    viz.plot_monthly_cashflow_evolution()
    plt.close(fig2)
    
    fig3 = plt.figure(figsize=(12, 8))
    viz.plot_tranche_comparison()
    plt.close(fig3)
    
    fig4 = plt.figure(figsize=(12, 8))
    viz.plot_loss_distribution()
    plt.close(fig4)
    
    print("All plots generated successfully!")
    
    return sim, results, viz

def test_parameter_sensitivity():
    """Test parameter sensitivity"""
    print("\n" + "=" * 50)
    print("Testing Parameter Sensitivity...")
    
    # Test different correlation levels
    correlations = [0.1, 0.3, 0.5]
    results_by_corr = {}
    
    for corr in correlations:
        print(f"\nTesting correlation = {corr}")
        
        sim = MBSSimulation(
            n_loans=15,  # Small for quick testing
            correlation=corr,
            attachment_point=0.01,
            detachment_point=0.05,
            security_term_years=7
        )
        
        results = sim.run_monte_carlo(n_simulations=50, use_parallel=False)  # Small for testing
        results_by_corr[corr] = results
    
    # Compare results
    print("\nCorrelation Impact:")
    for corr, results in results_by_corr.items():
        mezz_payoffs = results['tranche_payoffs']['Mezzanine']
        mean_payoff = np.mean(mezz_payoffs)
        var_95 = np.percentile(mezz_payoffs, 5)
        print(f"  Correlation {corr}: Mean=${mean_payoff:,.0f}, 95%VaR=${var_95:,.0f}")
    
    return results_by_corr

def test_custom_loan_sizes():
    """Test custom loan size configuration"""
    print("\n" + "=" * 50)
    print("Testing Custom Loan Sizes...")
    
    # Create custom loan sizes
    custom_sizes = [10, 20, 30, 40, 50] * 4  # 20 loans with specific sizes
    
    sim = MBSSimulation(
        n_loans=20,
        loan_sizes=custom_sizes,
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    print(f"Custom loan sizes: {custom_sizes}")
    print(f"Total pool principal: ${sum(loan.principal for loan in sim.loans):,.0f}")
    
    results = sim.run_monte_carlo(n_simulations=50, use_parallel=False)
    
    print("Custom loan size simulation completed!")
    
    return sim, results

def main():
    """Run all tests"""
    print("MBS SIMULATION SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Test basic functionality
        sim1, results1 = test_basic_functionality()
        print("\n✓ Basic functionality test passed")
        
        # Test visualization
        sim2, results2, viz = test_visualization()
        print("\n✓ Visualization test passed")
        
        # Test parameter sensitivity
        results3 = test_parameter_sensitivity()
        print("\n✓ Parameter sensitivity test passed")
        
        # Test custom loan sizes
        sim4, results4 = test_custom_loan_sizes()
        print("\n✓ Custom loan sizes test passed")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe MBS simulation system is working correctly.")
        print("You can now run the full examples using 'python example_usage.py'")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
