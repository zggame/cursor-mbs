#!/usr/bin/env python3
"""
Correlation Impact on Tail Risk Analysis
=======================================

This script analyzes how different correlation levels affect tail risk
in MBS simulations, focusing on worst-case scenarios (1% and 5% tail events).

Correlation levels tested: 0.01, 0.1, 0.2, 0.3, 0.4
Simulation paths: 1000 (with option to test with 50 first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mbs_simulation import MBSSimulation
import json
from datetime import datetime
import time

def run_correlation_analysis(n_paths=1000, test_mode=False):
    """Run comprehensive correlation impact analysis on tail risk"""
    
    if test_mode:
        n_paths = 50
        print("=== TEST MODE: Running with 50 paths for quick validation ===")
    else:
        print(f"=== FULL ANALYSIS: Running with {n_paths} paths ===")
    
    print("Analyzing correlation impact on tail risk...")
    print("Correlation levels: 0.01, 0.1, 0.2, 0.3, 0.4")
    print()
    
    # Correlation levels to test
    correlations = [0.01, 0.1, 0.2, 0.3, 0.4]
    
    # Storage for results
    all_results = {}
    correlation_summary = {}
    
    for i, correlation in enumerate(correlations):
        print(f"Running correlation {correlation} ({i+1}/{len(correlations)})...")
        start_time = time.time()
        
        # Create simulation with current correlation
        simulation = MBSSimulation(
            n_loans=100,  # Larger pool for better tail analysis
            correlation=correlation,
            attachment_point=0.02,  # 2% attachment point
            detachment_point=0.08,  # 8% detachment point
            security_term_years=7,
            loan_term_years=10,
            amortization_years=30
        )
        
        # Run simulation with detailed tracking
        results = simulation.run_monte_carlo(
            n_simulations=n_paths,
            seed=42 + i,  # Different seed for each correlation
            detailed_tracking=True
        )
        
        # Store results
        all_results[correlation] = {
            'simulation': simulation,
            'results': results,
            'tail_risk': results['tail_risk_analysis']
        }
        
        # Calculate summary metrics
        correlation_summary[correlation] = {}
        for tranche_name, metrics in results['tail_risk_analysis'].items():
            correlation_summary[correlation][tranche_name] = {
                'avg_payoff': metrics['avg_payoff'],
                'worst_5pct_avg_payoff': metrics['worst_5pct_avg_payoff'],
                'worst_1pct_avg_payoff': metrics['worst_1pct_avg_payoff'],
                'worst_case_payoff': metrics['worst_case_payoff'],
                'avg_loss': metrics['avg_loss'],
                'worst_5pct_avg_loss': metrics['worst_5pct_avg_loss'],
                'worst_1pct_avg_loss': metrics['worst_1pct_avg_loss'],
                'worst_case_loss': metrics['worst_case_loss'],
                'avg_loss_pct': metrics['avg_loss_pct'],
                'worst_5pct_avg_loss_pct': metrics['worst_5pct_avg_loss_pct'],
                'worst_1pct_avg_loss_pct': metrics['worst_1pct_avg_loss_pct'],
                'worst_case_loss_pct': metrics['worst_case_loss_pct'],
                'avg_default_rate': metrics['avg_default_rate'],
                'worst_5pct_avg_default_rate': metrics['worst_5pct_avg_default_rate'],
                'worst_1pct_avg_default_rate': metrics['worst_1pct_avg_default_rate'],
                'worst_case_default_rate': metrics['worst_case_default_rate'],
                'var_95': metrics['var_95'],
                'var_99': metrics['var_99'],
                'expected_shortfall_95': metrics['expected_shortfall_95'],
                'expected_shortfall_99': metrics['expected_shortfall_99']
            }
        
        elapsed_time = time.time() - start_time
        print(f"  Completed in {elapsed_time:.1f} seconds")
        print(f"  Total loss records: {len(results['detailed_loss_records'])}")
        print()
    
    # Print summary results
    print("=== CORRELATION IMPACT SUMMARY ===")
    print_summary_results(correlation_summary)
    
    # Export detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_test" if test_mode else "_full"
    
    # Export summary to CSV
    export_summary_to_csv(correlation_summary, f"correlation_tail_risk_summary{mode_suffix}_{timestamp}.csv")
    
    # Export detailed data for each correlation
    for correlation in correlations:
        simulation = all_results[correlation]['simulation']
        filename = f"correlation_{correlation}_detailed_data{mode_suffix}_{timestamp}.csv"
        simulation.export_detailed_loss_data(filename)
        print(f"Exported detailed data for correlation {correlation}: {filename}")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    create_correlation_visualizations(correlation_summary, all_results, test_mode)
    
    return all_results, correlation_summary

def print_summary_results(correlation_summary):
    """Print summary of correlation impact on tail risk"""
    
    # Create summary table
    print("\nCorrelation Impact on Tail Risk Metrics:")
    print("=" * 80)
    
    for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
        print(f"\n{tranche_name} Tranche:")
        print("-" * 40)
        print(f"{'Correlation':<12} {'Avg Payoff':<12} {'Worst 5%':<12} {'Worst 1%':<12} {'Worst Case':<12}")
        print("-" * 60)
        
        for correlation in sorted(correlation_summary.keys()):
            metrics = correlation_summary[correlation][tranche_name]
            print(f"{correlation:<12} "
                  f"${metrics['avg_payoff']:>10,.0f} "
                  f"${metrics['worst_5pct_avg_payoff']:>10,.0f} "
                  f"${metrics['worst_1pct_avg_payoff']:>10,.0f} "
                  f"${metrics['worst_case_payoff']:>10,.0f}")
    
    print("\n" + "=" * 80)
    print("Default Rate Impact:")
    print("-" * 40)
    print(f"{'Correlation':<12} {'Avg Default':<12} {'Worst 5%':<12} {'Worst 1%':<12} {'Worst Case':<12}")
    print("-" * 60)
    
    for correlation in sorted(correlation_summary.keys()):
        metrics = correlation_summary[correlation]['Mezzanine']  # Use Mezzanine as reference
        print(f"{correlation:<12} "
              f"{metrics['avg_default_rate']:>10.2%} "
              f"{metrics['worst_5pct_avg_default_rate']:>10.2%} "
              f"{metrics['worst_1pct_avg_default_rate']:>10.2%} "
              f"{metrics['worst_case_default_rate']:>10.2%}")

def export_summary_to_csv(correlation_summary, filename):
    """Export summary results to CSV"""
    
    # Prepare data for CSV
    data = []
    for correlation in sorted(correlation_summary.keys()):
        for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
            metrics = correlation_summary[correlation][tranche_name]
            row = {
                'correlation': correlation,
                'tranche': tranche_name,
                'avg_payoff': metrics['avg_payoff'],
                'worst_5pct_avg_payoff': metrics['worst_5pct_avg_payoff'],
                'worst_1pct_avg_payoff': metrics['worst_1pct_avg_payoff'],
                'worst_case_payoff': metrics['worst_case_payoff'],
                'avg_loss': metrics['avg_loss'],
                'worst_5pct_avg_loss': metrics['worst_5pct_avg_loss'],
                'worst_1pct_avg_loss': metrics['worst_1pct_avg_loss'],
                'worst_case_loss': metrics['worst_case_loss'],
                'avg_loss_pct': metrics['avg_loss_pct'],
                'worst_5pct_avg_loss_pct': metrics['worst_5pct_avg_loss_pct'],
                'worst_1pct_avg_loss_pct': metrics['worst_1pct_avg_loss_pct'],
                'worst_case_loss_pct': metrics['worst_case_loss_pct'],
                'avg_default_rate': metrics['avg_default_rate'],
                'worst_5pct_avg_default_rate': metrics['worst_5pct_avg_default_rate'],
                'worst_1pct_avg_default_rate': metrics['worst_1pct_avg_default_rate'],
                'worst_case_default_rate': metrics['worst_case_default_rate'],
                'var_95': metrics['var_95'],
                'var_99': metrics['var_99'],
                'expected_shortfall_95': metrics['expected_shortfall_95'],
                'expected_shortfall_99': metrics['expected_shortfall_99']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Summary exported to: {filename}")

def create_correlation_visualizations(correlation_summary, all_results, test_mode):
    """Create comprehensive visualizations for correlation impact"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Payoff vs Correlation for different tail levels
    ax1 = plt.subplot(3, 3, 1)
    plot_payoff_vs_correlation(ax1, correlation_summary)
    
    # 2. Loss percentage vs Correlation
    ax2 = plt.subplot(3, 3, 2)
    plot_loss_pct_vs_correlation(ax2, correlation_summary)
    
    # 3. Default rate vs Correlation
    ax3 = plt.subplot(3, 3, 3)
    plot_default_rate_vs_correlation(ax3, correlation_summary)
    
    # 4. VaR vs Correlation
    ax4 = plt.subplot(3, 3, 4)
    plot_var_vs_correlation(ax4, correlation_summary)
    
    # 5. Expected Shortfall vs Correlation
    ax5 = plt.subplot(3, 3, 5)
    plot_es_vs_correlation(ax5, correlation_summary)
    
    # 6. Tail risk comparison heatmap
    ax6 = plt.subplot(3, 3, 6)
    plot_tail_risk_heatmap(ax6, correlation_summary)
    
    # 7. Payoff distributions comparison
    ax7 = plt.subplot(3, 3, 7)
    plot_payoff_distributions_comparison(ax7, all_results)
    
    # 8. Loss path comparison for worst cases
    ax8 = plt.subplot(3, 3, 8)
    plot_worst_case_loss_paths_comparison(ax8, all_results)
    
    # 9. Correlation sensitivity analysis
    ax9 = plt.subplot(3, 3, 9)
    plot_correlation_sensitivity(ax9, correlation_summary)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_test" if test_mode else "_full"
    filename = f"correlation_tail_risk_analysis{mode_suffix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved to: {filename}")

def plot_payoff_vs_correlation(ax, correlation_summary):
    """Plot payoff vs correlation for different tail levels"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche in enumerate(tranches):
        avg_payoffs = [correlation_summary[corr][tranche]['avg_payoff'] for corr in correlations]
        worst_5pct = [correlation_summary[corr][tranche]['worst_5pct_avg_payoff'] for corr in correlations]
        worst_1pct = [correlation_summary[corr][tranche]['worst_1pct_avg_payoff'] for corr in correlations]
        
        ax.plot(correlations, avg_payoffs, 'o-', label=f'{tranche} Avg', color=colors[i], linewidth=2)
        ax.plot(correlations, worst_5pct, 's--', label=f'{tranche} Worst 5%', color=colors[i], alpha=0.7)
        ax.plot(correlations, worst_1pct, '^:', label=f'{tranche} Worst 1%', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Payoff ($)')
    ax.set_title('Payoff vs Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_loss_pct_vs_correlation(ax, correlation_summary):
    """Plot loss percentage vs correlation"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche in enumerate(tranches):
        avg_loss_pct = [correlation_summary[corr][tranche]['avg_loss_pct'] for corr in correlations]
        worst_5pct_loss = [correlation_summary[corr][tranche]['worst_5pct_avg_loss_pct'] for corr in correlations]
        worst_1pct_loss = [correlation_summary[corr][tranche]['worst_1pct_avg_loss_pct'] for corr in correlations]
        
        ax.plot(correlations, avg_loss_pct, 'o-', label=f'{tranche} Avg', color=colors[i], linewidth=2)
        ax.plot(correlations, worst_5pct_loss, 's--', label=f'{tranche} Worst 5%', color=colors[i], alpha=0.7)
        ax.plot(correlations, worst_1pct_loss, '^:', label=f'{tranche} Worst 1%', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Loss Percentage')
    ax.set_title('Loss % vs Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_default_rate_vs_correlation(ax, correlation_summary):
    """Plot default rate vs correlation"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche in enumerate(tranches):
        avg_default = [correlation_summary[corr][tranche]['avg_default_rate'] for corr in correlations]
        worst_5pct_default = [correlation_summary[corr][tranche]['worst_5pct_avg_default_rate'] for corr in correlations]
        worst_1pct_default = [correlation_summary[corr][tranche]['worst_1pct_avg_default_rate'] for corr in correlations]
        
        ax.plot(correlations, avg_default, 'o-', label=f'{tranche} Avg', color=colors[i], linewidth=2)
        ax.plot(correlations, worst_5pct_default, 's--', label=f'{tranche} Worst 5%', color=colors[i], alpha=0.7)
        ax.plot(correlations, worst_1pct_default, '^:', label=f'{tranche} Worst 1%', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Default Rate')
    ax.set_title('Default Rate vs Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_var_vs_correlation(ax, correlation_summary):
    """Plot VaR vs correlation"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche in enumerate(tranches):
        var_95 = [correlation_summary[corr][tranche]['var_95'] for corr in correlations]
        var_99 = [correlation_summary[corr][tranche]['var_99'] for corr in correlations]
        
        ax.plot(correlations, var_95, 'o-', label=f'{tranche} 95% VaR', color=colors[i], linewidth=2)
        ax.plot(correlations, var_99, 's--', label=f'{tranche} 99% VaR', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('VaR ($)')
    ax.set_title('VaR vs Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_es_vs_correlation(ax, correlation_summary):
    """Plot Expected Shortfall vs correlation"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche in enumerate(tranches):
        es_95 = [correlation_summary[corr][tranche]['expected_shortfall_95'] for corr in correlations]
        es_99 = [correlation_summary[corr][tranche]['expected_shortfall_99'] for corr in correlations]
        
        ax.plot(correlations, es_95, 'o-', label=f'{tranche} 95% ES', color=colors[i], linewidth=2)
        ax.plot(correlations, es_99, 's--', label=f'{tranche} 99% ES', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Expected Shortfall ($)')
    ax.set_title('Expected Shortfall vs Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def plot_tail_risk_heatmap(ax, correlation_summary):
    """Plot tail risk heatmap"""
    
    correlations = sorted(correlation_summary.keys())
    tranches = ['Subordinate', 'Mezzanine', 'Senior']
    metrics = ['worst_1pct_avg_payoff', 'worst_5pct_avg_payoff', 'avg_payoff']
    metric_labels = ['Worst 1%', 'Worst 5%', 'Average']
    
    data = []
    for metric in metrics:
        row = []
        for corr in correlations:
            # Use Mezzanine tranche as reference
            row.append(correlation_summary[corr]['Mezzanine'][metric])
        data.append(row)
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(correlations)):
            text = ax.text(j, i, f'${data[i][j]:,.0f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(correlations)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_title('Tail Risk Heatmap (Mezzanine Tranche)')
    
    plt.colorbar(im, ax=ax, label='Payoff ($)')

def plot_payoff_distributions_comparison(ax, all_results):
    """Plot payoff distributions comparison across correlations"""
    
    correlations = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(correlations)))
    
    for i, correlation in enumerate(correlations):
        detailed_records = all_results[correlation]['results']['detailed_loss_records']
        
        # Get Mezzanine tranche payoffs
        mezzanine_payoffs = [record.tranche_payoff for record in detailed_records 
                           if record.tranche_name == 'Mezzanine']
        
        ax.hist(mezzanine_payoffs, bins=30, alpha=0.6, density=True, 
                label=f'Corr={correlation}', color=colors[i])
    
    ax.set_xlabel('Mezzanine Tranche Payoff ($)')
    ax.set_ylabel('Density')
    ax.set_title('Payoff Distributions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_worst_case_loss_paths_comparison(ax, all_results):
    """Plot worst case loss paths comparison"""
    
    correlations = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(correlations)))
    
    for i, correlation in enumerate(correlations):
        detailed_records = all_results[correlation]['results']['detailed_loss_records']
        
        # Get worst 1% Mezzanine tranche records
        mezzanine_records = [record for record in detailed_records 
                           if record.tranche_name == 'Mezzanine']
        mezzanine_records.sort(key=lambda x: x.tranche_payoff)
        worst_1pct = mezzanine_records[:len(mezzanine_records)//100]  # 1%
        
        # Plot a few representative paths
        for record in worst_1pct[:5]:  # Plot first 5 worst paths
            ax.plot(record.cumulative_losses, alpha=0.3, color=colors[i])
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative Loss ($)')
    ax.set_title('Worst 1% Loss Paths Comparison')
    ax.grid(True, alpha=0.3)

def plot_correlation_sensitivity(ax, correlation_summary):
    """Plot correlation sensitivity analysis"""
    
    correlations = sorted(correlation_summary.keys())
    
    # Calculate sensitivity (change in worst 1% payoff per unit change in correlation)
    mezzanine_worst_1pct = [correlation_summary[corr]['Mezzanine']['worst_1pct_avg_payoff'] 
                           for corr in correlations]
    
    # Calculate percentage change from baseline (correlation 0.01)
    baseline = mezzanine_worst_1pct[0]
    pct_changes = [(payoff - baseline) / baseline * 100 for payoff in mezzanine_worst_1pct]
    
    ax.plot(correlations, pct_changes, 'o-', linewidth=2, markersize=8, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Change in Worst 1% Payoff (%)')
    ax.set_title('Correlation Sensitivity (vs 0.01 baseline)')
    ax.grid(True, alpha=0.3)

def main():
    """Main function to run the analysis"""
    
    print("=== CORRELATION IMPACT ON TAIL RISK ANALYSIS ===")
    print("This analysis examines how different correlation levels affect tail risk")
    print("in MBS simulations, focusing on worst-case scenarios.")
    print()
    
    # First run test with 50 paths
    print("Step 1: Running test with 50 paths...")
    test_results, test_summary = run_correlation_analysis(n_paths=50, test_mode=True)
    
    print("\n" + "="*80)
    print("TEST COMPLETED. Review results above.")
    print("If satisfied, proceed to full analysis with 1000 paths.")
    print("="*80)
    
    # Ask user if they want to proceed with full analysis
    response = input("\nProceed with full analysis (1000 paths)? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nStep 2: Running full analysis with 1000 paths...")
        full_results, full_summary = run_correlation_analysis(n_paths=1000, test_mode=False)
        
        print("\n" + "="*80)
        print("FULL ANALYSIS COMPLETED!")
        print("="*80)
    else:
        print("Full analysis skipped. Test results are available above.")

if __name__ == "__main__":
    main()
