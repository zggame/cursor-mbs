#!/usr/bin/env python3
"""
Simple MBS Simulation Demo
This script runs a quick demonstration of the MBS simulation system.
"""

from mbs_simulation import MBSSimulation, MBSVisualizer
import numpy as np

def main():
    print("MBS SIMULATION DEMO")
    print("=" * 50)
    
    # Create simulation
    print("Creating simulation...")
    sim = MBSSimulation(
        n_loans=25,
        correlation=0.3,
        attachment_point=0.01,  # 1%
        detachment_point=0.05,  # 5%
        security_term_years=7
    )
    
    print(f"✓ Created simulation with {sim.n_loans} loans")
    print(f"✓ Total pool principal: ${sum(loan.principal for loan in sim.loans):,.0f}")
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    results = sim.run_monte_carlo(n_simulations=300, use_parallel=True)
    print("✓ Simulation completed!")
    
    # Create visualizer
    viz = MBSVisualizer(sim)
    
    # Generate summary report
    print("\n" + "=" * 50)
    viz.create_summary_report()
    
    # Show basic statistics
    print("\nQUICK STATISTICS:")
    print("-" * 30)
    for tranche_name, payoffs in results['tranche_payoffs'].items():
        mean_payoff = np.mean(payoffs)
        var_95 = np.percentile(payoffs, 5)
        print(f"{tranche_name}: Mean=${mean_payoff:,.0f}, 95%VaR=${var_95:,.0f}")
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
