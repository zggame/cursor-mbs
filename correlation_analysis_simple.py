#!/usr/bin/env python3
"""
Simplified Correlation Impact Analysis
======================================

This script analyzes correlation impact on tail risk with a simple test first,
then runs the full analysis.

Test: 2 correlations (0.01, 0.1) with 50 paths
Full: 5 correlations (0.01, 0.1, 0.2, 0.3, 0.4) with 1000 paths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mbs_simulation import MBSSimulation
from datetime import datetime
import time

def run_test_analysis():
    """Run test analysis with 2 correlations and 50 paths"""
    print("=== TEST ANALYSIS ===")
    print("Testing correlations: 0.01, 0.1")
    print("Using 50 paths per correlation")
    print()
    
    # Test correlation levels
    correlations = [0.01, 0.1]
    
    # Storage for results
    results_summary = {}
    
    for i, correlation in enumerate(correlations):
        print(f"Running correlation {correlation} ({i+1}/{len(correlations)})...")
        start_time = time.time()
        
        # Create simulation with current correlation
        simulation = MBSSimulation(
            n_loans=50,  # Smaller pool for faster execution
            correlation=correlation,
            attachment_point=0.02,  # 2% attachment point
            detachment_point=0.08,  # 8% detachment point
            security_term_years=7,
            loan_term_years=10,
            amortization_years=30
        )
        
        # Run simulation with detailed tracking
        results = simulation.run_monte_carlo(
            n_simulations=50,  # Small number for quick testing
            seed=42 + i,  # Different seed for each correlation
            detailed_tracking=True
        )
        
        # Store key metrics
        results_summary[correlation] = {}
        for tranche_name, metrics in results['tail_risk_analysis'].items():
            results_summary[correlation][tranche_name] = {
                'avg_payoff': metrics['avg_payoff'],
                'worst_5pct_avg_payoff': metrics['worst_5pct_avg_payoff'],
                'worst_1pct_avg_payoff': metrics['worst_1pct_avg_payoff'],
                'worst_case_payoff': metrics['worst_case_payoff'],
                'avg_loss_pct': metrics['avg_loss_pct'],
                'worst_5pct_avg_loss_pct': metrics['worst_5pct_avg_loss_pct'],
                'worst_1pct_avg_loss_pct': metrics['worst_1pct_avg_loss_pct'],
                'worst_case_loss_pct': metrics['worst_case_loss_pct'],
                'avg_default_rate': metrics['avg_default_rate'],
                'worst_5pct_avg_default_rate': metrics['worst_5pct_avg_default_rate'],
                'worst_1pct_avg_default_rate': metrics['worst_1pct_avg_default_rate'],
                'worst_case_default_rate': metrics['worst_case_default_rate']
            }
        
        elapsed_time = time.time() - start_time
        print(f"  Completed in {elapsed_time:.1f} seconds")
        print(f"  Total loss records: {len(results['detailed_loss_records'])}")
        print()
    
    # Print summary results
    print("=== TEST RESULTS ===")
    print_correlation_summary(results_summary)
    
    # Create simple visualization
    create_visualization(results_summary, "test")
    
    return results_summary

def run_full_analysis():
    """Run full analysis with 5 correlations and 1000 paths"""
    print("\n" + "="*60)
    print("=== FULL ANALYSIS ===")
    print("Testing correlations: 0.01, 0.1, 0.2, 0.3, 0.4")
    print("Using 1000 paths per correlation")
    print("="*60)
    print()
    
    # Full correlation levels
    correlations = [0.01, 0.1, 0.2, 0.3, 0.4]
    
    # Storage for results
    results_summary = {}
    
    for i, correlation in enumerate(correlations):
        print(f"Running correlation {correlation} ({i+1}/{len(correlations)})...")
        start_time = time.time()
        
        # Create simulation with current correlation
        simulation = MBSSimulation(
            n_loans=100,  # Larger pool for better analysis
            correlation=correlation,
            attachment_point=0.02,  # 2% attachment point
            detachment_point=0.08,  # 8% detachment point
            security_term_years=7,
            loan_term_years=10,
            amortization_years=30
        )
        
        # Run simulation with detailed tracking
        results = simulation.run_monte_carlo(
            n_simulations=1000,  # Full analysis
            seed=42 + i,  # Different seed for each correlation
            detailed_tracking=True
        )
        
        # Store key metrics
        results_summary[correlation] = {}
        for tranche_name, metrics in results['tail_risk_analysis'].items():
            results_summary[correlation][tranche_name] = {
                'avg_payoff': metrics['avg_payoff'],
                'worst_5pct_avg_payoff': metrics['worst_5pct_avg_payoff'],
                'worst_1pct_avg_payoff': metrics['worst_1pct_avg_payoff'],
                'worst_case_payoff': metrics['worst_case_payoff'],
                'avg_loss_pct': metrics['avg_loss_pct'],
                'worst_5pct_avg_loss_pct': metrics['worst_5pct_avg_loss_pct'],
                'worst_1pct_avg_loss_pct': metrics['worst_1pct_avg_loss_pct'],
                'worst_case_loss_pct': metrics['worst_case_loss_pct'],
                'avg_default_rate': metrics['avg_default_rate'],
                'worst_5pct_avg_default_rate': metrics['worst_5pct_avg_default_rate'],
                'worst_1pct_avg_default_rate': metrics['worst_1pct_avg_default_rate'],
                'worst_case_default_rate': metrics['worst_case_default_rate']
            }
        
        elapsed_time = time.time() - start_time
        print(f"  Completed in {elapsed_time:.1f} seconds")
        print(f"  Total loss records: {len(results['detailed_loss_records'])}")
        print()
    
    # Print summary results
    print("=== FULL ANALYSIS RESULTS ===")
    print_correlation_summary(results_summary)
    
    # Export results to CSV
    export_results_to_csv(results_summary, "full_correlation_analysis.csv")
    
    # Create visualization
    create_visualization(results_summary, "full")
    
    return results_summary

def print_correlation_summary(results_summary):
    """Print summary of correlation impact"""
    
    print("\nCorrelation Impact on Tail Risk (Mezzanine Tranche):")
    print("=" * 60)
    print(f"{'Correlation':<12} {'Avg Payoff':<12} {'Worst 5%':<12} {'Worst 1%':<12} {'Worst Case':<12}")
    print("-" * 60)
    
    for correlation in sorted(results_summary.keys()):
        metrics = results_summary[correlation]['Mezzanine']
        print(f"{correlation:<12} "
              f"${metrics['avg_payoff']:>10,.0f} "
              f"${metrics['worst_5pct_avg_payoff']:>10,.0f} "
              f"${metrics['worst_1pct_avg_payoff']:>10,.0f} "
              f"${metrics['worst_case_payoff']:>10,.0f}")
    
    print("\n" + "=" * 60)
    print("Default Rate Impact:")
    print("-" * 40)
    print(f"{'Correlation':<12} {'Avg Default':<12} {'Worst 5%':<12} {'Worst 1%':<12}")
    print("-" * 50)
    
    for correlation in sorted(results_summary.keys()):
        metrics = results_summary[correlation]['Mezzanine']
        print(f"{correlation:<12} "
              f"{metrics['avg_default_rate']:>10.2%} "
              f"{metrics['worst_5pct_avg_default_rate']:>10.2%} "
              f"{metrics['worst_1pct_avg_default_rate']:>10.2%}")

def create_visualization(results_summary, mode):
    """Create visualization of correlation impact"""
    
    correlations = sorted(results_summary.keys())
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Payoff vs Correlation
    mezzanine_payoffs = [results_summary[corr]['Mezzanine']['avg_payoff'] for corr in correlations]
    worst_5pct = [results_summary[corr]['Mezzanine']['worst_5pct_avg_payoff'] for corr in correlations]
    worst_1pct = [results_summary[corr]['Mezzanine']['worst_1pct_avg_payoff'] for corr in correlations]
    
    ax1.plot(correlations, mezzanine_payoffs, 'o-', label='Average', linewidth=2)
    ax1.plot(correlations, worst_5pct, 's--', label='Worst 5%', alpha=0.7)
    ax1.plot(correlations, worst_1pct, '^:', label='Worst 1%', alpha=0.7)
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Payoff ($)')
    ax1.set_title('Mezzanine Tranche Payoff vs Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Percentage vs Correlation
    avg_loss_pct = [results_summary[corr]['Mezzanine']['avg_loss_pct'] for corr in correlations]
    worst_5pct_loss = [results_summary[corr]['Mezzanine']['worst_5pct_avg_loss_pct'] for corr in correlations]
    worst_1pct_loss = [results_summary[corr]['Mezzanine']['worst_1pct_avg_loss_pct'] for corr in correlations]
    
    ax2.plot(correlations, avg_loss_pct, 'o-', label='Average', linewidth=2)
    ax2.plot(correlations, worst_5pct_loss, 's--', label='Worst 5%', alpha=0.7)
    ax2.plot(correlations, worst_1pct_loss, '^:', label='Worst 1%', alpha=0.7)
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Loss Percentage')
    ax2.set_title('Loss % vs Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Default Rate vs Correlation
    avg_default = [results_summary[corr]['Mezzanine']['avg_default_rate'] for corr in correlations]
    worst_5pct_default = [results_summary[corr]['Mezzanine']['worst_5pct_avg_default_rate'] for corr in correlations]
    worst_1pct_default = [results_summary[corr]['Mezzanine']['worst_1pct_avg_default_rate'] for corr in correlations]
    
    ax3.plot(correlations, avg_default, 'o-', label='Average', linewidth=2)
    ax3.plot(correlations, worst_5pct_default, 's--', label='Worst 5%', alpha=0.7)
    ax3.plot(correlations, worst_1pct_default, '^:', label='Worst 1%', alpha=0.7)
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Default Rate')
    ax3.set_title('Default Rate vs Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation sensitivity
    baseline = worst_1pct[0]
    pct_changes = [(payoff - baseline) / baseline * 100 for payoff in worst_1pct]
    
    ax4.plot(correlations, pct_changes, 'o-', linewidth=2, markersize=8, color='red')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Correlation')
    ax4.set_ylabel('Change in Worst 1% Payoff (%)')
    ax4.set_title('Correlation Sensitivity (vs 0.01 baseline)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"correlation_analysis_{mode}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: {filename}")

def export_results_to_csv(results_summary, filename):
    """Export results to CSV"""
    
    data = []
    for correlation in sorted(results_summary.keys()):
        for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
            metrics = results_summary[correlation][tranche_name]
            row = {
                'correlation': correlation,
                'tranche': tranche_name,
                'avg_payoff': metrics['avg_payoff'],
                'worst_5pct_avg_payoff': metrics['worst_5pct_avg_payoff'],
                'worst_1pct_avg_payoff': metrics['worst_1pct_avg_payoff'],
                'worst_case_payoff': metrics['worst_case_payoff'],
                'avg_loss_pct': metrics['avg_loss_pct'],
                'worst_5pct_avg_loss_pct': metrics['worst_5pct_avg_loss_pct'],
                'worst_1pct_avg_loss_pct': metrics['worst_1pct_avg_loss_pct'],
                'worst_case_loss_pct': metrics['worst_case_loss_pct'],
                'avg_default_rate': metrics['avg_default_rate'],
                'worst_5pct_avg_default_rate': metrics['worst_5pct_avg_default_rate'],
                'worst_1pct_avg_default_rate': metrics['worst_1pct_avg_default_rate'],
                'worst_case_default_rate': metrics['worst_case_default_rate']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results exported to: {filename}")

def main():
    """Main function to run the analysis"""
    
    print("=== CORRELATION IMPACT ON TAIL RISK ANALYSIS ===")
    print("This analysis examines how different correlation levels affect tail risk")
    print("in MBS simulations, focusing on worst-case scenarios.")
    print()
    
    # Step 1: Run test analysis
    test_results = run_test_analysis()
    
    print("\n" + "="*60)
    print("TEST ANALYSIS COMPLETED!")
    print("="*60)
    
    # Step 2: Run full analysis
    full_results = run_full_analysis()
    
    print("\n" + "="*60)
    print("FULL ANALYSIS COMPLETED!")
    print("="*60)
    
    print("\n=== SUMMARY ===")
    print("Test analysis: 2 correlations (0.01, 0.1) with 50 paths each")
    print("Full analysis: 5 correlations (0.01, 0.1, 0.2, 0.3, 0.4) with 1000 paths each")
    print("All results exported to CSV files")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()
