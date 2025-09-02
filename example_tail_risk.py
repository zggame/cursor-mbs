#!/usr/bin/env python3
"""
Simple Example: Tail Risk Analysis
==================================

This script demonstrates the new tail risk analysis capabilities
with a smaller simulation for quick testing and understanding.
"""

import numpy as np
from mbs_simulation import MBSSimulation

def main():
    print("=== Simple Tail Risk Analysis Example ===")
    print("Focusing on worst-case scenarios (1% and 5% tail events)")
    print()
    
    # Create a smaller simulation for quick testing
    print("Creating MBS simulation...")
    simulation = MBSSimulation(
        n_loans=50,  # Smaller pool for faster execution
        correlation=0.3,
        attachment_point=0.02,  # 2% attachment point
        detachment_point=0.08,  # 8% detachment point
        security_term_years=7,
        loan_term_years=10,
        amortization_years=30
    )
    
    # Run simulation with detailed tracking
    print("Running Monte Carlo simulation with detailed loss tracking...")
    results = simulation.run_monte_carlo(
        n_simulations=1000,  # Smaller number for quick testing
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
        
        print(f"\n  Worst 1% Scenarios ({metrics['worst_1pct_count']} cases):")
        print(f"    Average Payoff: ${metrics['worst_1pct_avg_payoff']:,.2f}")
        print(f"    Average Loss: ${metrics['worst_1pct_avg_loss']:,.2f}")
        print(f"    Average Loss %: {metrics['worst_1pct_avg_loss_pct']:.2%}")
        print(f"    Average Default Rate: {metrics['worst_1pct_avg_default_rate']:.2%}")
        
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
    csv_filename = simulation.export_detailed_loss_data("example_tail_risk_data.csv")
    print(f"Detailed loss data exported to: {csv_filename}")
    
    print("\n=== Analysis Complete ===")
    print("Key insights:")
    print("- The analysis focuses on worst-case scenarios rather than average performance")
    print("- 1% and 5% tail events show significantly worse outcomes than averages")
    print("- Detailed loss data is exported for further analysis")
    print("- This approach is crucial for risk management and stress testing")

if __name__ == "__main__":
    main()
