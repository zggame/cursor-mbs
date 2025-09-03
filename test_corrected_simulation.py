#!/usr/bin/env python3
"""
Test Corrected MBS Simulation
=============================

This script tests the corrected MBS simulation and compares it with the original approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from corrected_mbs_simulation import CorrectedMBSSimulation
from mbs_simulation import MBSSimulation

def test_corrected_simulation():
    """Test the corrected simulation approach"""
    print("🧪 TESTING CORRECTED MBS SIMULATION")
    print("=" * 60)
    
    # Test parameters
    n_loans = 100
    correlation = 0.3
    attachment_point = 0.01  # 1%
    detachment_point = 0.05  # 5%
    security_term_years = 7
    n_simulations = 500
    
    print(f"Parameters:")
    print(f"  Number of loans: {n_loans}")
    print(f"  Correlation: {correlation}")
    print(f"  Attachment point: {attachment_point:.1%}")
    print(f"  Detachment point: {detachment_point:.1%}")
    print(f"  Security term: {security_term_years} years")
    print(f"  Simulations: {n_simulations}")
    print()
    
    # Create corrected simulation
    print("Creating corrected simulation...")
    corrected_sim = CorrectedMBSSimulation(
        n_loans=n_loans,
        correlation=correlation,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        security_term_years=security_term_years
    )
    
    # Print tranche information
    corrected_sim.print_tranche_info()
    
    # Run corrected simulation
    print("Running corrected simulation...")
    corrected_results = corrected_sim.run_monte_carlo(n_simulations=n_simulations, seed=42)
    
    # Create original simulation for comparison
    print("Creating original simulation...")
    original_sim = MBSSimulation(
        n_loans=n_loans,
        correlation=correlation,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        security_term_years=security_term_years
    )
    
    # Run original simulation
    print("Running original simulation...")
    original_results = original_sim.run_monte_carlo(n_simulations=n_simulations, seed=42)
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    print("Original vs Corrected Simulation:")
    print("-" * 50)
    
    for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
        print(f"\n{tranche_name} Tranche:")
        
        # Original results
        orig_payoffs = original_results['tranche_payoffs'][tranche_name]
        orig_mean = np.mean(orig_payoffs)
        orig_wipeout = np.mean(np.array(orig_payoffs) == 0) * 100
        
        # Corrected results
        corr_payoffs = corrected_results['tranche_payoffs'][tranche_name]
        corr_mean = np.mean(corr_payoffs)
        corr_wipeout = np.mean(np.array(corr_payoffs) == 0) * 100
        
        # Principal and interest breakdown
        corr_principal = np.mean(corrected_results['tranche_principal_payoffs'][tranche_name])
        corr_interest = np.mean(corrected_results['tranche_interest_payoffs'][tranche_name])
        
        print(f"  Original:")
        print(f"    Mean payoff: ${orig_mean:,.0f}")
        print(f"    Wipeout rate: {orig_wipeout:.1f}%")
        
        print(f"  Corrected:")
        print(f"    Mean payoff: ${corr_mean:,.0f}")
        print(f"    Principal: ${corr_principal:,.0f}")
        print(f"    Interest: ${corr_interest:,.0f}")
        print(f"    Wipeout rate: {corr_wipeout:.1f}%")
        
        # Difference
        diff = corr_mean - orig_mean
        diff_pct = (diff / orig_mean) * 100 if orig_mean != 0 else 0
        print(f"  Difference: ${diff:+,.0f} ({diff_pct:+.1f}%)")
    
    # Analyze tranche balance history
    print(f"\n📈 TRANCHE BALANCE ANALYSIS")
    print("-" * 50)
    
    # Look at a few sample paths
    sample_path = 0
    for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
        balance_history = corrected_results['tranche_balance_history'][tranche_name][sample_path]
        initial_balance = balance_history[0]
        final_balance = balance_history[-1]
        min_balance = min(balance_history)
        
        print(f"{tranche_name}:")
        print(f"  Initial: ${initial_balance:,.0f}")
        print(f"  Final: ${final_balance:,.0f}")
        print(f"  Minimum: ${min_balance:,.0f}")
        print(f"  Loss: ${initial_balance - final_balance:,.0f}")
        print()
    
    # Create visualizations
    create_comparison_plots(corrected_results, original_results)
    
    return corrected_results, original_results

def create_comparison_plots(corrected_results, original_results):
    """Create comparison plots"""
    print("Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Payoff comparison
    tranche_names = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['red', 'orange', 'green']
    
    x_pos = np.arange(len(tranche_names))
    width = 0.35
    
    orig_means = [np.mean(original_results['tranche_payoffs'][name]) for name in tranche_names]
    corr_means = [np.mean(corrected_results['tranche_payoffs'][name]) for name in tranche_names]
    
    axes[0,0].bar(x_pos - width/2, orig_means, width, label='Original', alpha=0.7)
    axes[0,0].bar(x_pos + width/2, corr_means, width, label='Corrected', alpha=0.7)
    axes[0,0].set_title('Mean Payoff Comparison')
    axes[0,0].set_xlabel('Tranche')
    axes[0,0].set_ylabel('Mean Payoff ($)')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(tranche_names)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Wipeout rate comparison
    orig_wipeouts = [np.mean(np.array(original_results['tranche_payoffs'][name]) == 0) * 100 for name in tranche_names]
    corr_wipeouts = [np.mean(np.array(corrected_results['tranche_payoffs'][name]) == 0) * 100 for name in tranche_names]
    
    axes[0,1].bar(x_pos - width/2, orig_wipeouts, width, label='Original', alpha=0.7)
    axes[0,1].bar(x_pos + width/2, corr_wipeouts, width, label='Corrected', alpha=0.7)
    axes[0,1].set_title('Wipeout Rate Comparison')
    axes[0,1].set_xlabel('Tranche')
    axes[0,1].set_ylabel('Wipeout Rate (%)')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(tranche_names)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Principal vs Interest breakdown (corrected only)
    principals = [np.mean(corrected_results['tranche_principal_payoffs'][name]) for name in tranche_names]
    interests = [np.mean(corrected_results['tranche_interest_payoffs'][name]) for name in tranche_names]
    
    axes[1,0].bar(x_pos, principals, width, label='Principal', alpha=0.7)
    axes[1,0].bar(x_pos, interests, width, bottom=principals, label='Interest', alpha=0.7)
    axes[1,0].set_title('Corrected Payoff Breakdown')
    axes[1,0].set_xlabel('Tranche')
    axes[1,0].set_ylabel('Payoff ($)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(tranche_names)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Tranche balance history (sample path)
    sample_path = 0
    months = range(len(corrected_results['tranche_balance_history']['Subordinate'][sample_path]))
    
    for i, name in enumerate(tranche_names):
        balance_history = corrected_results['tranche_balance_history'][name][sample_path]
        axes[1,1].plot(months, balance_history, label=name, color=colors[i], linewidth=2)
    
    axes[1,1].set_title('Tranche Balance History (Sample Path)')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Remaining Principal ($)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_vs_original_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot: corrected_vs_original_comparison.png")
    plt.show()

def analyze_key_differences(corrected_results, original_results):
    """Analyze key differences between approaches"""
    print(f"\n🔍 KEY DIFFERENCES ANALYSIS")
    print("=" * 60)
    
    print("1. Cashflow Distribution:")
    print("   Original: Cumulative losses allocated at end")
    print("   Corrected: Monthly losses allocated as they occur")
    print()
    
    print("2. Principal Tracking:")
    print("   Original: No tracking of remaining principal")
    print("   Corrected: Tracks remaining principal for each tranche")
    print()
    
    print("3. Interest Calculation:")
    print("   Original: No separate interest calculation")
    print("   Corrected: Interest = Remaining principal × rate × time")
    print()
    
    print("4. Wipeout Mechanism:")
    print("   Original: Tranche wiped out if cumulative loss > principal")
    print("   Corrected: Tranche balance reduced monthly, wiped out when = 0")
    print()
    
    # Calculate some key metrics
    for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
        orig_payoffs = np.array(original_results['tranche_payoffs'][tranche_name])
        corr_payoffs = np.array(corrected_results['tranche_payoffs'][tranche_name])
        
        orig_wipeout_rate = np.mean(orig_payoffs == 0) * 100
        corr_wipeout_rate = np.mean(corr_payoffs == 0) * 100
        
        print(f"{tranche_name} Wipeout Rate:")
        print(f"  Original: {orig_wipeout_rate:.1f}%")
        print(f"  Corrected: {corr_wipeout_rate:.1f}%")
        print(f"  Difference: {corr_wipeout_rate - orig_wipeout_rate:+.1f}%")
        print()

if __name__ == "__main__":
    # Run test
    corrected_results, original_results = test_corrected_simulation()
    
    # Analyze differences
    analyze_key_differences(corrected_results, original_results)
    
    print("✅ Test completed!")
