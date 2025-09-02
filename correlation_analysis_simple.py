#!/usr/bin/env python3
"""
Simplified Correlation Impact Analysis
=====================================

This script analyzes correlation impact on tail risk with a flexible function
that takes paths and correlations as input parameters.

Test: 2 correlations (0.01, 0.1) with 50 paths
Full: 5 correlations (0.01, 0.1, 0.2, 0.3, 0.4) with 1000 paths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mbs_simulation import MBSSimulation
from datetime import datetime
import time
import os

def run_correlation_analysis(correlations, n_paths, n_loans=100, mode="analysis"):
    """
    Run correlation impact analysis with specified parameters
    
    Args:
        correlations (list): List of correlation values to test
        n_paths (int): Number of simulation paths per correlation
        n_loans (int): Number of loans in the pool
        mode (str): Analysis mode for naming outputs
    """
    print(f"=== {mode.upper()} ANALYSIS ===")
    print(f"Testing correlations: {', '.join(map(str, correlations))}")
    print(f"Using {n_paths} paths per correlation")
    print(f"Pool size: {n_loans} loans")
    print(f"Using PARALLEL processing for speed")
    print()
    
    # Create output directory
    output_dir = "analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Storage for results
    results_summary = {}
    all_path_data = {}  # Store all path-level data
    
    for i, correlation in enumerate(correlations):
        print(f"Running correlation {correlation} ({i+1}/{len(correlations)})...")
        start_time = time.time()
        
        # Create simulation with current correlation
        print(f"  Creating simulation with {n_loans} loans...")
        simulation = MBSSimulation(
            n_loans=n_loans,
            correlation=correlation,
            attachment_point=0.01,  # 1%
            detachment_point=0.05,  # 5%
            security_term_years=7,
            loan_term_years=10,
            amortization_years=30
        )
        
        # Print loan information for the first correlation
        if i == 0:
            simulation.print_loan_summary()
            # Export loan information to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            loan_filename = os.path.join(output_dir, f"loan_information_{mode}_{timestamp}.csv")
            export_loan_info_to_csv(simulation, loan_filename)
        
        # Run simulation with parallel processing (fast)
        print(f"  Running Monte Carlo simulation with {n_paths} paths (parallel)...")
        results = simulation.run_monte_carlo(
            n_simulations=n_paths,
            seed=42 + i,  # Different seed for each correlation
            use_parallel=True,  # Enable parallel processing
            detailed_tracking=False  # No detailed tracking for speed
        )
        
        # Store all path-level data
        print(f"  Processing results and calculating tail risk metrics...")
        all_path_data[correlation] = {}
        results_summary[correlation] = {}  # Initialize the correlation entry
        
        # Calculate total losses for each path
        total_losses = np.sum(results['monthly_losses'], axis=1)
        total_principal = sum(loan.principal for loan in simulation.loans)
        loss_percentages = total_losses / total_principal
        
        # Store path-level data for each tranche
        for tranche_name, payoffs in results['tranche_payoffs'].items():
            payoffs_array = np.array(payoffs)
            tranche_principal = simulation.tranches[0].principal  # Assuming all tranches have same principal
            
            # Calculate losses for this tranche
            losses = tranche_principal - payoffs_array
            loss_percentages_tranche = losses / tranche_principal
            
            # Store complete path data
            all_path_data[correlation][tranche_name] = {
                'payoffs': payoffs_array,
                'losses': losses,
                'loss_percentages': loss_percentages_tranche,
                'pool_loss_percentages': loss_percentages,  # Pool-level losses
                'default_rates': loss_percentages * 2  # Rough estimate
            }
            
            # Calculate tail risk metrics by ranking
            sorted_payoffs = np.sort(payoffs_array)
            n_paths_actual = len(payoffs_array)
            
            # Calculate percentiles
            avg_payoff = np.mean(payoffs_array)
            worst_5pct_avg_payoff = np.mean(sorted_payoffs[:max(1, int(n_paths_actual * 0.05))])
            worst_1pct_avg_payoff = np.mean(sorted_payoffs[:max(1, int(n_paths_actual * 0.01))])
            worst_case_payoff = np.min(payoffs_array)
            
            # Calculate loss metrics
            avg_loss = tranche_principal - avg_payoff
            worst_5pct_avg_loss = tranche_principal - worst_5pct_avg_payoff
            worst_1pct_avg_loss = tranche_principal - worst_1pct_avg_payoff
            worst_case_loss = tranche_principal - worst_case_payoff
            
            # Calculate loss percentages
            avg_loss_pct = avg_loss / tranche_principal
            worst_5pct_avg_loss_pct = worst_5pct_avg_loss / tranche_principal
            worst_1pct_avg_loss_pct = worst_1pct_avg_loss / tranche_principal
            worst_case_loss_pct = worst_case_loss / tranche_principal
            
            # Calculate default rate percentiles
            sorted_loss_pct = np.sort(loss_percentages)
            avg_default_rate = np.mean(loss_percentages) * 2  # Rough estimate
            worst_5pct_avg_default_rate = np.mean(sorted_loss_pct[-max(1, int(n_paths_actual * 0.05)):]) * 2
            worst_1pct_avg_default_rate = np.mean(sorted_loss_pct[-max(1, int(n_paths_actual * 0.01)):]) * 2
            worst_case_default_rate = np.max(loss_percentages) * 2
            
            results_summary[correlation][tranche_name] = {
                'avg_payoff': avg_payoff,
                'worst_5pct_avg_payoff': worst_5pct_avg_payoff,
                'worst_1pct_avg_payoff': worst_1pct_avg_payoff,
                'worst_case_payoff': worst_case_payoff,
                'avg_loss_pct': avg_loss_pct,
                'worst_5pct_avg_loss_pct': worst_5pct_avg_loss_pct,
                'worst_1pct_avg_loss_pct': worst_1pct_avg_loss_pct,
                'worst_case_loss_pct': worst_case_loss_pct,
                'avg_default_rate': avg_default_rate,
                'worst_5pct_avg_default_rate': worst_5pct_avg_default_rate,
                'worst_1pct_avg_default_rate': worst_1pct_avg_default_rate,
                'worst_case_default_rate': worst_case_default_rate
            }
        
        elapsed_time = time.time() - start_time
        print(f"  Completed in {elapsed_time:.1f} seconds")
        print(f"  Total simulations: {len(results['tranche_payoffs']['Mezzanine'])}")
        print()
    
    # Print summary results
    print(f"=== {mode.upper()} RESULTS ===")
    print_correlation_summary(results_summary)
    
    # Export summary results to CSV
    print(f"Exporting summary results to CSV...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_filename = os.path.join(output_dir, f"correlation_analysis_summary_{mode}_{timestamp}.csv")
    export_results_to_csv(results_summary, summary_csv_filename)
    
    # Export all path-level data to CSV
    print(f"Exporting all path-level data to CSV...")
    path_data_csv_filename = os.path.join(output_dir, f"correlation_analysis_paths_{mode}_{timestamp}.csv")
    export_path_data_to_csv(all_path_data, path_data_csv_filename)
    
    # Create visualization (save only, don't show to avoid blocking)
    print(f"Creating visualizations...")
    create_visualization(results_summary, mode, output_dir)
    
    return results_summary, all_path_data

def export_path_data_to_csv(all_path_data, filename):
    """Export all path-level data to CSV"""
    
    data = []
    for correlation in sorted(all_path_data.keys()):
        for tranche_name in ['Subordinate', 'Mezzanine', 'Senior']:
            path_data = all_path_data[correlation][tranche_name]
            
            for path_idx in range(len(path_data['payoffs'])):
                row = {
                    'correlation': correlation,
                    'tranche': tranche_name,
                    'path_id': path_idx,
                    'payoff': path_data['payoffs'][path_idx],
                    'loss': path_data['losses'][path_idx],
                    'loss_percentage': path_data['loss_percentages'][path_idx],
                    'pool_loss_percentage': path_data['pool_loss_percentages'][path_idx],
                    'default_rate': path_data['default_rates'][path_idx]
                }
                data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Path-level data exported to: {filename}")

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

def create_visualization(results_summary, mode, output_dir):
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
    
    # Save plot (don't show to avoid blocking)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"correlation_analysis_{mode}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Visualization saved to: {filename}")

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
    print(f"Summary results exported to: {filename}")

def export_loan_info_to_csv(simulation, filename):
    """Export loan information to CSV for further analysis"""
    loan_info = simulation.dump_loan_information()
    
    # Export individual loan details
    loan_df = pd.DataFrame(loan_info['loan_details'])
    loan_df.to_csv(filename, index=False)
    print(f"Loan information exported to: {filename}")
    
    # Also export summary to a separate file
    summary_filename = filename.replace('.csv', '_summary.csv')
    summary_data = {
        'metric': list(loan_info['summary'].keys()),
        'value': list(loan_info['summary'].values())
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    print(f"Loan summary exported to: {summary_filename}")

def main():
    """Main function to run the analysis"""
    
    print("=== CORRELATION IMPACT ON TAIL RISK ANALYSIS ===")
    print("This analysis examines how different correlation levels affect tail risk")
    print("in MBS simulations, focusing on worst-case scenarios.")
    print()
    
    # Step 1: Run test analysis
    test_correlations = [0.01, 0.1]
    test_results, test_path_data = run_correlation_analysis(
        correlations=test_correlations,
        n_paths=50,
        n_loans=50,
        mode="test"
    )
    
    print("\n" + "="*60)
    print("TEST ANALYSIS COMPLETED!")
    print("="*60)
    
    # Step 2: Run full analysis
    full_correlations = [0.01, 0.1, 0.2, 0.3, 0.4]
    full_results, full_path_data = run_correlation_analysis(
        correlations=full_correlations,
        n_paths=500,  # Reduced from 1000 to prevent memory issues
        n_loans=100,
        mode="full"
    )
    
    print("\n" + "="*60)
    print("FULL ANALYSIS COMPLETED!")
    print("="*60)
    
    print("\n=== SUMMARY ===")
    print(f"Test analysis: {len(test_correlations)} correlations with 50 paths each")
    print(f"Full analysis: {len(full_correlations)} correlations with 1000 paths each")
    print("All results exported to CSV files in analysis_output/")
    print("Visualizations saved as PNG files in analysis_output/")
    print("Path-level data available for further analysis")

if __name__ == "__main__":
    main()
