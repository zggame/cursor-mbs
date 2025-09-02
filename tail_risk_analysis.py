#!/usr/bin/env python3
"""
Tail Risk Analysis for MBS Simulation
=====================================

This script demonstrates the tail risk analysis capabilities of the MBS simulation,
focusing on worst-case scenarios (1% and 5% tail events) rather than average performance.

Key Features:
- Detailed loss tracking for each simulation path and tranche
- Analysis of worst 1% and 5% scenarios
- Export of detailed loss data for further analysis
- Visualization of tail risk distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mbs_simulation import MBSSimulation, MBSVisualizer
import seaborn as sns

def run_tail_risk_analysis():
    """Run comprehensive tail risk analysis"""
    print("=== MBS Tail Risk Analysis ===")
    print("Focusing on worst-case scenarios (1% and 5% tail events)")
    print()
    
    # Create simulation with detailed tracking
    print("Creating MBS simulation...")
    simulation = MBSSimulation(
        n_loans=100,  # Larger pool for better tail analysis
        correlation=0.4,  # Higher correlation for more extreme scenarios
        attachment_point=0.02,  # 2% attachment point
        detachment_point=0.08,  # 8% detachment point
        security_term_years=7,
        loan_term_years=10,
        amortization_years=30
    )
    
    # Run simulation with detailed tracking
    print("Running Monte Carlo simulation with detailed loss tracking...")
    print("This may take a few minutes for 10,000 simulations...")
    results = simulation.run_monte_carlo(
        n_simulations=10000,  # Large number for better tail analysis
        seed=42,
        detailed_tracking=True  # Enable detailed loss tracking
    )
    
    print(f"Completed {len(results['detailed_loss_records'])} detailed loss records")
    print()
    
    # Analyze tail risk metrics
    print("=== Tail Risk Analysis Results ===")
    tail_risk = results['tail_risk_analysis']
    
    for tranche_name, metrics in tail_risk.items():
        print(f"\n{tranche_name} Tranche:")
        print(f"  Total Simulations: {metrics['total_simulations']}")
        print(f"  Average Payoff: ${metrics['avg_payoff']:,.2f}")
        print(f"  Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"  Average Loss %: {metrics['avg_loss_pct']:.2%}")
        print(f"  Average Default Rate: {metrics['avg_default_rate']:.2%}")
        
        print(f"\n  Worst 5% Scenarios ({metrics['worst_5pct_count']} cases):")
        print(f"    Average Payoff: ${metrics['worst_5pct_avg_payoff']:,.2f}")
        print(f"    Average Loss: ${metrics['worst_5pct_avg_loss']:,.2f}")
        print(f"    Average Loss %: {metrics['worst_5pct_avg_loss_pct']:.2%}")
        print(f"    Average Default Rate: {metrics['worst_5pct_avg_default_rate']:.2%}")
        print(f"    Average Worst Month Loss: ${metrics['worst_5pct_avg_worst_month_loss']:,.2f}")
        
        print(f"\n  Worst 1% Scenarios ({metrics['worst_1pct_count']} cases):")
        print(f"    Average Payoff: ${metrics['worst_1pct_avg_payoff']:,.2f}")
        print(f"    Average Loss: ${metrics['worst_1pct_avg_loss']:,.2f}")
        print(f"    Average Loss %: {metrics['worst_1pct_avg_loss_pct']:.2%}")
        print(f"    Average Default Rate: {metrics['worst_1pct_avg_default_rate']:.2%}")
        print(f"    Average Worst Month Loss: ${metrics['worst_1pct_avg_worst_month_loss']:,.2f}")
        
        print(f"\n  Risk Metrics:")
        print(f"    95% VaR: ${metrics['var_95']:,.2f}")
        print(f"    99% VaR: ${metrics['var_99']:,.2f}")
        print(f"    95% Expected Shortfall: ${metrics['expected_shortfall_95']:,.2f}")
        print(f"    99% Expected Shortfall: ${metrics['expected_shortfall_99']:,.2f}")
        
        print(f"\n  Worst Case:")
        print(f"    Worst Payoff: ${metrics['worst_case_payoff']:,.2f}")
        print(f"    Maximum Loss: ${metrics['worst_case_loss']:,.2f}")
        print(f"    Maximum Loss %: {metrics['worst_case_loss_pct']:.2%}")
        print(f"    Maximum Default Rate: {metrics['worst_case_default_rate']:.2%}")
    
    # Get detailed worst-case scenarios
    print("\n=== Detailed Worst-Case Scenario Analysis ===")
    worst_5pct = simulation.get_worst_case_scenarios(percentile=5.0)
    worst_1pct = simulation.get_worst_case_scenarios(percentile=1.0)
    
    for tranche_name in worst_5pct.keys():
        print(f"\n{tranche_name} - Worst 5% Scenarios:")
        summary = worst_5pct[tranche_name]['summary']
        print(f"  Count: {summary['count']}")
        print(f"  Average Payoff: ${summary['avg_payoff']:,.2f}")
        print(f"  Average Loss: ${summary['avg_loss']:,.2f}")
        print(f"  Average Default Rate: {summary['avg_default_rate']:.2%}")
        print(f"  Minimum Payoff: ${summary['min_payoff']:,.2f}")
        
        print(f"\n{tranche_name} - Worst 1% Scenarios:")
        summary = worst_1pct[tranche_name]['summary']
        print(f"  Count: {summary['count']}")
        print(f"  Average Payoff: ${summary['avg_payoff']:,.2f}")
        print(f"  Average Loss: ${summary['avg_loss']:,.2f}")
        print(f"  Average Default Rate: {summary['avg_default_rate']:.2%}")
        print(f"  Minimum Payoff: ${summary['min_payoff']:,.2f}")
    
    # Export detailed data
    print("\n=== Exporting Detailed Loss Data ===")
    csv_filename = simulation.export_detailed_loss_data("tail_risk_detailed_data.csv")
    print(f"Detailed loss data exported to: {csv_filename}")
    
    # Create visualizations
    print("\n=== Creating Tail Risk Visualizations ===")
    create_tail_risk_visualizations(simulation, results)
    
    return simulation, results

def create_tail_risk_visualizations(simulation, results):
    """Create comprehensive visualizations for tail risk analysis"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Payoff distributions with tail highlighting
    ax1 = plt.subplot(3, 3, 1)
    plot_payoff_distributions_with_tails(ax1, results)
    
    # 2. Loss percentage distributions
    ax2 = plt.subplot(3, 3, 2)
    plot_loss_percentage_distributions(ax2, results)
    
    # 3. Default rate distributions
    ax3 = plt.subplot(3, 3, 3)
    plot_default_rate_distributions(ax3, results)
    
    # 4. Worst month loss analysis
    ax4 = plt.subplot(3, 3, 4)
    plot_worst_month_analysis(ax4, results)
    
    # 5. Tail risk comparison
    ax5 = plt.subplot(3, 3, 5)
    plot_tail_risk_comparison(ax5, results)
    
    # 6. Cumulative loss paths for worst cases
    ax6 = plt.subplot(3, 3, 6)
    plot_worst_case_loss_paths(ax6, results)
    
    # 7. Risk metrics summary
    ax7 = plt.subplot(3, 3, 7)
    plot_risk_metrics_summary(ax7, results)
    
    # 8. Correlation impact on tail risk
    ax8 = plt.subplot(3, 3, 8)
    plot_correlation_tail_impact(ax8, simulation)
    
    # 9. Tranche comparison
    ax9 = plt.subplot(3, 3, 9)
    plot_tranche_comparison(ax9, results)
    
    plt.tight_layout()
    plt.savefig('tail_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Tail risk visualizations saved to: tail_risk_analysis.png")

def plot_payoff_distributions_with_tails(ax, results):
    """Plot payoff distributions with tail regions highlighted"""
    detailed_records = results['detailed_loss_records']
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'tranche_name': record.tranche_name,
            'tranche_payoff': record.tranche_payoff
        }
        for record in detailed_records
    ])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche_name in enumerate(df['tranche_name'].unique()):
        tranche_data = df[df['tranche_name'] == tranche_name]
        
        # Calculate percentiles
        p5 = np.percentile(tranche_data['tranche_payoff'], 5)
        p1 = np.percentile(tranche_data['tranche_payoff'], 1)
        
        # Plot histogram
        ax.hist(tranche_data['tranche_payoff'], bins=50, alpha=0.7, 
                label=tranche_name, color=colors[i], density=True)
        
        # Highlight tail regions
        ax.axvline(p5, color=colors[i], linestyle='--', alpha=0.8, 
                  label=f'{tranche_name} 5% VaR')
        ax.axvline(p1, color=colors[i], linestyle=':', alpha=0.8, 
                  label=f'{tranche_name} 1% VaR')
    
    ax.set_xlabel('Tranche Payoff ($)')
    ax.set_ylabel('Density')
    ax.set_title('Payoff Distributions with Tail Risk')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_loss_percentage_distributions(ax, results):
    """Plot loss percentage distributions"""
    detailed_records = results['detailed_loss_records']
    
    df = pd.DataFrame([
        {
            'tranche_name': record.tranche_name,
            'loss_percentage': record.loss_percentage
        }
        for record in detailed_records
    ])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche_name in enumerate(df['tranche_name'].unique()):
        tranche_data = df[df['tranche_name'] == tranche_name]
        ax.hist(tranche_data['loss_percentage'], bins=50, alpha=0.7, 
                label=tranche_name, color=colors[i], density=True)
    
    ax.set_xlabel('Loss Percentage')
    ax.set_ylabel('Density')
    ax.set_title('Loss Percentage Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_default_rate_distributions(ax, results):
    """Plot default rate distributions"""
    detailed_records = results['detailed_loss_records']
    
    df = pd.DataFrame([
        {
            'tranche_name': record.tranche_name,
            'default_rate': record.default_rate
        }
        for record in detailed_records
    ])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche_name in enumerate(df['tranche_name'].unique()):
        tranche_data = df[df['tranche_name'] == tranche_name]
        ax.hist(tranche_data['default_rate'], bins=50, alpha=0.7, 
                label=tranche_name, color=colors[i], density=True)
    
    ax.set_xlabel('Default Rate')
    ax.set_ylabel('Density')
    ax.set_title('Default Rate Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_worst_month_analysis(ax, results):
    """Plot worst month loss analysis"""
    detailed_records = results['detailed_loss_records']
    
    df = pd.DataFrame([
        {
            'tranche_name': record.tranche_name,
            'worst_month_loss': record.worst_month_loss,
            'worst_month': record.worst_month
        }
        for record in detailed_records
    ])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, tranche_name in enumerate(df['tranche_name'].unique()):
        tranche_data = df[df['tranche_name'] == tranche_name]
        ax.scatter(tranche_data['worst_month'], tranche_data['worst_month_loss'], 
                  alpha=0.6, label=tranche_name, color=colors[i], s=20)
    
    ax.set_xlabel('Worst Month')
    ax.set_ylabel('Worst Month Loss ($)')
    ax.set_title('Worst Month Loss Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_tail_risk_comparison(ax, results):
    """Plot tail risk comparison across tranches"""
    tail_risk = results['tail_risk_analysis']
    
    tranches = list(tail_risk.keys())
    worst_5pct_payoffs = [tail_risk[t]['worst_5pct_avg_payoff'] for t in tranches]
    worst_1pct_payoffs = [tail_risk[t]['worst_1pct_avg_payoff'] for t in tranches]
    avg_payoffs = [tail_risk[t]['avg_payoff'] for t in tranches]
    
    x = np.arange(len(tranches))
    width = 0.25
    
    ax.bar(x - width, avg_payoffs, width, label='Average', alpha=0.8)
    ax.bar(x, worst_5pct_payoffs, width, label='Worst 5%', alpha=0.8)
    ax.bar(x + width, worst_1pct_payoffs, width, label='Worst 1%', alpha=0.8)
    
    ax.set_xlabel('Tranche')
    ax.set_ylabel('Payoff ($)')
    ax.set_title('Tail Risk Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tranches)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_worst_case_loss_paths(ax, results):
    """Plot cumulative loss paths for worst cases"""
    detailed_records = results['detailed_loss_records']
    
    # Get worst 1% scenarios for each tranche
    worst_scenarios = {}
    for record in detailed_records:
        if record.tranche_name not in worst_scenarios:
            worst_scenarios[record.tranche_name] = []
        worst_scenarios[record.tranche_name].append(record)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (tranche_name, records) in enumerate(worst_scenarios.items()):
        # Sort by payoff and get worst 1%
        records.sort(key=lambda x: x.tranche_payoff)
        worst_1pct = records[:len(records)//100]  # 1%
        
        for record in worst_1pct:
            ax.plot(record.cumulative_losses, alpha=0.3, color=colors[i])
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative Loss ($)')
    ax.set_title('Worst 1% Loss Paths')
    ax.grid(True, alpha=0.3)

def plot_risk_metrics_summary(ax, results):
    """Plot risk metrics summary"""
    tail_risk = results['tail_risk_analysis']
    
    tranches = list(tail_risk.keys())
    var_95 = [tail_risk[t]['var_95'] for t in tranches]
    var_99 = [tail_risk[t]['var_99'] for t in tranches]
    es_95 = [tail_risk[t]['expected_shortfall_95'] for t in tranches]
    es_99 = [tail_risk[t]['expected_shortfall_99'] for t in tranches]
    
    x = np.arange(len(tranches))
    width = 0.2
    
    ax.bar(x - 1.5*width, var_95, width, label='95% VaR', alpha=0.8)
    ax.bar(x - 0.5*width, var_99, width, label='99% VaR', alpha=0.8)
    ax.bar(x + 0.5*width, es_95, width, label='95% ES', alpha=0.8)
    ax.bar(x + 1.5*width, es_99, width, label='99% ES', alpha=0.8)
    
    ax.set_xlabel('Tranche')
    ax.set_ylabel('Payoff ($)')
    ax.set_title('Risk Metrics Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(tranches)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_correlation_tail_impact(ax, simulation):
    """Plot impact of correlation on tail risk"""
    correlations = [0.1, 0.2, 0.3, 0.4, 0.5]
    worst_1pct_payoffs = []
    
    for corr in correlations:
        # Create temporary simulation with different correlation
        temp_sim = MBSSimulation(
            n_loans=simulation.n_loans,
            loan_sizes=[loan.principal for loan in simulation.loans],
            correlation=corr,
            attachment_point=simulation.attachment_point,
            detachment_point=simulation.detachment_point,
            security_term_years=simulation.security_term_years,
            loan_term_years=simulation.loans[0].term_years,
            amortization_years=simulation.loans[0].amortization_years
        )
        
        # Run smaller simulation for quick analysis
        results = temp_sim.run_monte_carlo(n_simulations=1000, detailed_tracking=True)
        tail_risk = results['tail_risk_analysis']
        
        # Get worst 1% payoff for mezzanine tranche
        worst_1pct_payoffs.append(tail_risk['Mezzanine']['worst_1pct_avg_payoff'])
    
    ax.plot(correlations, worst_1pct_payoffs, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Worst 1% Average Payoff ($)')
    ax.set_title('Correlation Impact on Tail Risk')
    ax.grid(True, alpha=0.3)

def plot_tranche_comparison(ax, results):
    """Plot comprehensive tranche comparison"""
    tail_risk = results['tail_risk_analysis']
    
    tranches = list(tail_risk.keys())
    metrics = ['avg_payoff', 'worst_5pct_avg_payoff', 'worst_1pct_avg_payoff', 'worst_case_payoff']
    metric_labels = ['Average', 'Worst 5%', 'Worst 1%', 'Worst Case']
    
    data = []
    for metric in metrics:
        data.append([tail_risk[t][metric] for t in tranches])
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(tranches)):
            text = ax.text(j, i, f'${data[i][j]:,.0f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xticks(range(len(tranches)))
    ax.set_xticklabels(tranches)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_title('Tranche Performance Comparison')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Payoff ($)')

if __name__ == "__main__":
    # Run the tail risk analysis
    simulation, results = run_tail_risk_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Key files generated:")
    print("- tail_risk_analysis.png: Comprehensive visualizations")
    print("- tail_risk_detailed_data.csv: Detailed loss data for further analysis")
    print("\nThe analysis focuses on worst-case scenarios (1% and 5% tail events)")
    print("rather than average performance, providing insights for risk management.")
