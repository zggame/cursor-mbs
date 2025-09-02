"""
Example usage of the MBS Simulation System

This script demonstrates various scenarios and parameter configurations
for the Multi-Family CRT (MBS) security simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mbs_simulation import MBSSimulation, MBSVisualizer

def example_1_basic_simulation():
    """Basic simulation with default parameters"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Simulation")
    print("=" * 60)
    
    # Create simulation with default parameters
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,  # 1%
        detachment_point=0.05,  # 5%
        security_term_years=7,
        loan_term_years=10,
        amortization_years=30
    )
    
    # Run Monte Carlo simulation with parallel processing
    results = sim.run_monte_carlo(n_simulations=1000, use_parallel=True)
    
    # Create visualizer
    viz = MBSVisualizer(sim)
    
    # Generate summary report
    viz.create_summary_report()
    
    return sim, results, viz

def example_2_different_correlation_levels():
    """Compare different correlation levels"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Different Correlation Levels")
    print("=" * 60)
    
    correlations = [0.1, 0.3, 0.5, 0.7]
    results_by_correlation = {}
    
    for corr in correlations:
        print(f"\nRunning simulation with correlation = {corr}")
        
        sim = MBSSimulation(
            n_loans=50,
            correlation=corr,
            attachment_point=0.01,
            detachment_point=0.05,
            security_term_years=7
        )
        
        results = sim.run_monte_carlo(n_simulations=500, use_parallel=True)
        results_by_correlation[corr] = results
    
    # Compare results
    print("\nCorrelation Impact Analysis:")
    print("-" * 40)
    
    for corr, results in results_by_correlation.items():
        print(f"\nCorrelation {corr}:")
        for tranche_name, payoffs in results['tranche_payoffs'].items():
            mean_payoff = np.mean(payoffs)
            std_payoff = np.std(payoffs)
            var_95 = np.percentile(payoffs, 5)
            print(f"  {tranche_name}: Mean=${mean_payoff:,.0f}, Std=${std_payoff:,.0f}, 95%VaR=${var_95:,.0f}")
    
    return results_by_correlation

def example_3_different_tranche_levels():
    """Compare different tranche attachment/detachment points"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Tranche Levels")
    print("=" * 60)
    
    tranche_configs = [
        (0.005, 0.025),  # 0.5% to 2.5%
        (0.01, 0.05),    # 1% to 5%
        (0.02, 0.08),    # 2% to 8%
        (0.03, 0.10)     # 3% to 10%
    ]
    
    results_by_config = {}
    
    for att, det in tranche_configs:
        print(f"\nRunning simulation with attachment={att:.1%}, detachment={det:.1%}")
        
        sim = MBSSimulation(
            n_loans=50,
            correlation=0.3,
            attachment_point=att,
            detachment_point=det,
            security_term_years=7
        )
        
        results = sim.run_monte_carlo(n_simulations=500)
        results_by_config[f"{att:.1%}-{det:.1%}"] = results
    
    # Compare mezzanine tranche performance
    print("\nMezzanine Tranche Performance Comparison:")
    print("-" * 50)
    
    for config_name, results in results_by_config.items():
        mezz_payoffs = results['tranche_payoffs']['Mezzanine']
        mean_payoff = np.mean(mezz_payoffs)
        var_95 = np.percentile(mezz_payoffs, 5)
        var_99 = np.percentile(mezz_payoffs, 1)
        print(f"  {config_name}: Mean=${mean_payoff:,.0f}, 95%VaR=${var_95:,.0f}, 99%VaR=${var_99:,.0f}")
    
    return results_by_config

def example_4_different_loan_sizes():
    """Compare different loan size distributions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Different Loan Size Distributions")
    print("=" * 60)
    
    # Create different loan size distributions
    n_loans = 50
    
    # Small loans (2M-50M)
    small_loans = np.random.lognormal(mean=np.log(15), sigma=0.8, size=n_loans)
    small_loans = np.clip(small_loans, 2, 50)
    
    # Medium loans (20M-200M)
    medium_loans = np.random.lognormal(mean=np.log(80), sigma=0.8, size=n_loans)
    medium_loans = np.clip(medium_loans, 20, 200)
    
    # Large loans (100M-500M)
    large_loans = np.random.lognormal(mean=np.log(250), sigma=0.6, size=n_loans)
    large_loans = np.clip(large_loans, 100, 500)
    
    loan_distributions = {
        "Small Loans": small_loans,
        "Medium Loans": medium_loans,
        "Large Loans": large_loans
    }
    
    results_by_distribution = {}
    
    for dist_name, loan_sizes in loan_distributions.items():
        print(f"\nRunning simulation with {dist_name}")
        print(f"  Average loan size: ${np.mean(loan_sizes):,.0f}")
        print(f"  Total pool size: ${np.sum(loan_sizes):,.0f}")
        
        sim = MBSSimulation(
            n_loans=n_loans,
            loan_sizes=loan_sizes,
            correlation=0.3,
            attachment_point=0.01,
            detachment_point=0.05,
            security_term_years=7
        )
        
        results = sim.run_monte_carlo(n_simulations=500, use_parallel=True)
        results_by_distribution[dist_name] = results
    
    # Compare results
    print("\nImpact of Loan Size Distribution:")
    print("-" * 40)
    
    for dist_name, results in results_by_distribution.items():
        print(f"\n{dist_name}:")
        for tranche_name, payoffs in results['tranche_payoffs'].items():
            mean_payoff = np.mean(payoffs)
            var_95 = np.percentile(payoffs, 5)
            print(f"  {tranche_name}: Mean=${mean_payoff:,.0f}, 95%VaR=${var_95:,.0f}")
    
    return results_by_distribution

def example_5_stress_testing():
    """Stress testing with extreme scenarios"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Stress Testing")
    print("=" * 60)
    
    # Create a simulation with higher default rates for stress testing
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.5,  # Higher correlation
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    # Modify loan assumptions for stress scenario
    for loan in sim.loans:
        # Increase default rates by 3x
        loan.annual_default_rate *= 3
        loan.monthly_default_rate = loan.annual_default_rate / 12
        
        # Increase LGD rates by 1.5x
        loan.annual_lgd_rate *= 1.5
        loan.monthly_lgd_rate = loan.annual_lgd_rate
    
    results = sim.run_monte_carlo(n_simulations=2000, use_parallel=True)  # More simulations for stress testing
    
    viz = MBSVisualizer(sim)
    
    print("\nStress Test Results:")
    print("-" * 30)
    
    for tranche_name, payoffs in results['tranche_payoffs'].items():
        mean_payoff = np.mean(payoffs)
        var_99 = np.percentile(payoffs, 1)
        var_999 = np.percentile(payoffs, 0.1)  # 99.9% VaR
        worst_case = np.min(payoffs)
        
        print(f"\n{tranche_name} Tranche:")
        print(f"  Mean payoff: ${mean_payoff:,.0f}")
        print(f"  99% VaR: ${var_99:,.0f}")
        print(f"  99.9% VaR: ${var_999:,.0f}")
        print(f"  Worst case: ${worst_case:,.0f}")
    
    # Plot stress test results
    viz.plot_tranche_payoff_distributions()
    viz.plot_loss_distribution()
    
    return sim, results, viz

def example_6_sensitivity_analysis():
    """Sensitivity analysis for key parameters"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Sensitivity Analysis")
    print("=" * 60)
    
    # Test different parameter combinations
    base_sim = MBSSimulation(n_loans=30, security_term_years=7)
    
    # Parameter ranges to test
    correlations = [0.1, 0.3, 0.5]
    attachment_points = [0.005, 0.01, 0.02]
    
    sensitivity_results = {}
    
    for corr in correlations:
        for att in attachment_points:
            det = att + 0.04  # Keep mezzanine layer size constant
            
            print(f"\nTesting: correlation={corr}, attachment={att:.1%}")
            
            sim = MBSSimulation(
                n_loans=30,
                correlation=corr,
                attachment_point=att,
                detachment_point=det,
                security_term_years=7
            )
            
            results = sim.run_monte_carlo(n_simulations=300, use_parallel=True)
            
            # Store mezzanine tranche results
            mezz_payoffs = results['tranche_payoffs']['Mezzanine']
            key = f"corr_{corr}_att_{att:.3f}"
            sensitivity_results[key] = {
                'mean': np.mean(mezz_payoffs),
                'std': np.std(mezz_payoffs),
                'var_95': np.percentile(mezz_payoffs, 5),
                'correlation': corr,
                'attachment': att
            }
    
    # Create sensitivity heatmap
    print("\nSensitivity Analysis Results:")
    print("-" * 40)
    
    # Create data for heatmap
    corr_values = sorted(list(set([r['correlation'] for r in sensitivity_results.values()])))
    att_values = sorted(list(set([r['attachment'] for r in sensitivity_results.values()])))
    
    mean_matrix = np.zeros((len(corr_values), len(att_values)))
    var_matrix = np.zeros((len(corr_values), len(att_values)))
    
    for i, corr in enumerate(corr_values):
        for j, att in enumerate(att_values):
            key = f"corr_{corr}_att_{att:.3f}"
            if key in sensitivity_results:
                mean_matrix[i, j] = sensitivity_results[key]['mean']
                var_matrix[i, j] = sensitivity_results[key]['var_95']
    
    # Plot sensitivity heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean payoff heatmap
    im1 = ax1.imshow(mean_matrix, cmap='Blues', aspect='auto')
    ax1.set_title('Mean Mezzanine Payoff')
    ax1.set_xlabel('Attachment Point')
    ax1.set_ylabel('Correlation')
    ax1.set_xticks(range(len(att_values)))
    ax1.set_xticklabels([f'{att:.1%}' for att in att_values])
    ax1.set_yticks(range(len(corr_values)))
    ax1.set_yticklabels(corr_values)
    plt.colorbar(im1, ax=ax1)
    
    # 95% VaR heatmap
    im2 = ax2.imshow(var_matrix, cmap='Reds', aspect='auto')
    ax2.set_title('95% VaR Mezzanine Payoff')
    ax2.set_xlabel('Attachment Point')
    ax2.set_ylabel('Correlation')
    ax2.set_xticks(range(len(att_values)))
    ax2.set_xticklabels([f'{att:.1%}' for att in att_values])
    ax2.set_yticks(range(len(corr_values)))
    ax2.set_yticklabels(corr_values)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return sensitivity_results

def main():
    """Run all examples"""
    print("MBS SIMULATION EXAMPLES")
    print("=" * 80)
    
    # Run all examples
    results = {}
    
    # Example 1: Basic simulation
    sim1, res1, viz1 = example_1_basic_simulation()
    results['basic'] = (sim1, res1, viz1)
    
    # Example 2: Different correlations
    res2 = example_2_different_correlation_levels()
    results['correlations'] = res2
    
    # Example 3: Different tranche levels
    res3 = example_3_different_tranche_levels()
    results['tranche_levels'] = res3
    
    # Example 4: Different loan sizes
    res4 = example_4_different_loan_sizes()
    results['loan_sizes'] = res4
    
    # Example 5: Stress testing
    sim5, res5, viz5 = example_5_stress_testing()
    results['stress_test'] = (sim5, res5, viz5)
    
    # Example 6: Sensitivity analysis
    res6 = example_6_sensitivity_analysis()
    results['sensitivity'] = res6
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()
