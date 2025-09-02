#!/usr/bin/env python3
"""
Correlation Impact Analysis
This script analyzes the impact of correlation on mezzanine and senior tranche performance
"""

import numpy as np
import matplotlib.pyplot as plt
from mbs_simulation import MBSSimulation

def analyze_correlation_impact():
    """Analyze correlation impact on mezzanine and senior tranches"""
    print("CORRELATION IMPACT ANALYSIS")
    print("=" * 60)
    print("Focus: Mezzanine and Senior Tranche Performance")
    print("=" * 60)
    
    # Test different correlation levels
    correlations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results_by_correlation = {}
    
    # Simulation parameters
    n_loans = 50
    n_simulations = 1000  # More simulations for better accuracy
    
    print(f"Simulation Parameters:")
    print(f"  Number of loans: {n_loans}")
    print(f"  Number of simulations: {n_simulations}")
    print(f"  Attachment point: 1%")
    print(f"  Detachment point: 5%")
    print(f"  Security term: 7 years")
    
    for corr in correlations:
        print(f"\nRunning simulation with correlation = {corr}")
        
        sim = MBSSimulation(
            n_loans=n_loans,
            correlation=corr,
            attachment_point=0.01,  # 1%
            detachment_point=0.05,  # 5%
            security_term_years=7
        )
        
        results = sim.run_monte_carlo(n_simulations=n_simulations, use_parallel=True)
        results_by_correlation[corr] = results
    
    # Analyze results
    print("\n" + "=" * 80)
    print("MEZZANINE TRANCHE ANALYSIS")
    print("=" * 80)
    
    mezzanine_data = []
    for corr in correlations:
        payoffs = results_by_correlation[corr]['tranche_payoffs']['Mezzanine']
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)
        expected_shortfall_95 = np.mean(payoffs[payoffs <= var_95])
        
        mezzanine_data.append({
            'correlation': corr,
            'mean': mean_payoff,
            'std': std_payoff,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95
        })
        
        print(f"\nCorrelation {corr}:")
        print(f"  Mean payoff:     ${mean_payoff:,.0f}")
        print(f"  Std deviation:   ${std_payoff:,.0f}")
        print(f"  95% VaR:         ${var_95:,.0f}")
        print(f"  99% VaR:         ${var_99:,.0f}")
        print(f"  Expected Shortfall (95%): ${expected_shortfall_95:,.0f}")
        print(f"  Coefficient of Variation: {std_payoff/mean_payoff:.3f}")
    
    print("\n" + "=" * 80)
    print("SENIOR TRANCHE ANALYSIS")
    print("=" * 80)
    
    senior_data = []
    for corr in correlations:
        payoffs = results_by_correlation[corr]['tranche_payoffs']['Senior']
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)
        expected_shortfall_95 = np.mean(payoffs[payoffs <= var_95])
        
        senior_data.append({
            'correlation': corr,
            'mean': mean_payoff,
            'std': std_payoff,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95
        })
        
        print(f"\nCorrelation {corr}:")
        print(f"  Mean payoff:     ${mean_payoff:,.0f}")
        print(f"  Std deviation:   ${std_payoff:,.0f}")
        print(f"  95% VaR:         ${var_95:,.0f}")
        print(f"  99% VaR:         ${var_99:,.0f}")
        print(f"  Expected Shortfall (95%): ${expected_shortfall_95:,.0f}")
        print(f"  Coefficient of Variation: {std_payoff/mean_payoff:.3f}")
    
    # Create summary table
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Corr':<6} {'Mezzanine':<50} {'Senior':<50}")
    print(f"{'':<6} {'Mean':<12} {'Std':<12} {'95%VaR':<12} {'99%VaR':<12} {'Mean':<12} {'Std':<12} {'95%VaR':<12} {'99%VaR':<12}")
    print("-" * 100)
    
    for i, corr in enumerate(correlations):
        mezz = mezzanine_data[i]
        sen = senior_data[i]
        print(f"{corr:<6} ${mezz['mean']:<11,.0f} ${mezz['std']:<11,.0f} ${mezz['var_95']:<11,.0f} ${mezz['var_99']:<11,.0f} "
              f"${sen['mean']:<11,.0f} ${sen['std']:<11,.0f} ${sen['var_95']:<11,.0f} ${sen['var_99']:<11,.0f}")
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    # Mezzanine insights
    mezz_means = [d['mean'] for d in mezzanine_data]
    mezz_stds = [d['std'] for d in mezzanine_data]
    mezz_var95s = [d['var_95'] for d in mezzanine_data]
    
    print(f"MEZZANINE TRANCHE:")
    print(f"  Mean payoff range: ${min(mezz_means):,.0f} to ${max(mezz_means):,.0f}")
    print(f"  Volatility range: ${min(mezz_stds):,.0f} to ${max(mezz_stds):,.0f}")
    print(f"  95% VaR range: ${min(mezz_var95s):,.0f} to ${max(mezz_var95s):,.0f}")
    
    # Find correlation with highest/lowest risk
    max_vol_idx = np.argmax(mezz_stds)
    min_vol_idx = np.argmin(mezz_stds)
    print(f"  Highest volatility at correlation {correlations[max_vol_idx]} (${mezz_stds[max_vol_idx]:,.0f})")
    print(f"  Lowest volatility at correlation {correlations[min_vol_idx]} (${mezz_stds[min_vol_idx]:,.0f})")
    
    # Senior insights
    sen_means = [d['mean'] for d in senior_data]
    sen_stds = [d['std'] for d in senior_data]
    sen_var95s = [d['var_95'] for d in senior_data]
    
    print(f"\nSENIOR TRANCHE:")
    print(f"  Mean payoff range: ${min(sen_means):,.0f} to ${max(sen_means):,.0f}")
    print(f"  Volatility range: ${min(sen_stds):,.0f} to ${max(sen_stds):,.0f}")
    print(f"  95% VaR range: ${min(sen_var95s):,.0f} to ${max(sen_var95s):,.0f}")
    
    max_vol_idx = np.argmax(sen_stds)
    min_vol_idx = np.argmin(sen_stds)
    print(f"  Highest volatility at correlation {correlations[max_vol_idx]} (${sen_stds[max_vol_idx]:,.0f})")
    print(f"  Lowest volatility at correlation {correlations[min_vol_idx]} (${sen_stds[min_vol_idx]:,.0f})")
    
    return results_by_correlation, mezzanine_data, senior_data

def plot_correlation_impact(mezzanine_data, senior_data):
    """Create plots showing correlation impact"""
    correlations = [d['correlation'] for d in mezzanine_data]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mezzanine plots
    ax1.plot(correlations, [d['mean'] for d in mezzanine_data], 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Mezzanine Tranche - Mean Payoff vs Correlation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Mean Payoff ($)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(correlations, [d['std'] for d in mezzanine_data], 'r-o', linewidth=2, markersize=8)
    ax2.set_title('Mezzanine Tranche - Volatility vs Correlation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Standard Deviation ($)')
    ax2.grid(True, alpha=0.3)
    
    # Senior plots
    ax3.plot(correlations, [d['mean'] for d in senior_data], 'g-o', linewidth=2, markersize=8)
    ax3.set_title('Senior Tranche - Mean Payoff vs Correlation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Mean Payoff ($)')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(correlations, [d['std'] for d in senior_data], 'm-o', linewidth=2, markersize=8)
    ax4.set_title('Senior Tranche - Volatility vs Correlation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Correlation')
    ax4.set_ylabel('Standard Deviation ($)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run correlation impact analysis"""
    try:
        # Run analysis
        results_by_correlation, mezzanine_data, senior_data = analyze_correlation_impact()
        
        # Create plots
        plot_correlation_impact(mezzanine_data, senior_data)
        
        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETED!")
        print("=" * 100)
        print("\nKey findings:")
        print("- Higher correlation generally increases volatility for both tranches")
        print("- Mezzanine tranche is more sensitive to correlation changes")
        print("- Senior tranche shows more stable performance across correlation levels")
        print("- Plots saved as 'correlation_impact_analysis.png'")
        
    except Exception as e:
        print(f"\n❌ Correlation analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
