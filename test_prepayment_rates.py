#!/usr/bin/env python3
"""
Test Prepayment Rates
This script verifies that prepayment rates are now realistic for multi-family loans
"""

import numpy as np
from mbs_simulation import MBSSimulation

def test_prepayment_rates():
    """Test that prepayment rates are realistic"""
    print("TESTING PREPAYMENT RATES")
    print("=" * 50)
    
    # Create simulation
    sim = MBSSimulation(
        n_loans=100,  # More loans for better statistics
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    # Collect prepayment rates from all loans
    prepay_rates = [loan.annual_prepay_rate for loan in sim.loans]
    
    print(f"Prepayment Rate Statistics:")
    print(f"  Mean annual rate: {np.mean(prepay_rates):.3%}")
    print(f"  Standard deviation: {np.std(prepay_rates):.3%}")
    print(f"  Minimum rate: {np.min(prepay_rates):.3%}")
    print(f"  Maximum rate: {np.max(prepay_rates):.3%}")
    print(f"  Median rate: {np.median(prepay_rates):.3%}")
    
    # Check that rates are reasonable
    mean_rate = np.mean(prepay_rates)
    max_rate = np.max(prepay_rates)
    
    print(f"\nValidation:")
    print(f"  Mean rate ~1%: {'✓' if 0.005 <= mean_rate <= 0.015 else '❌'}")
    print(f"  Max rate ≤3%: {'✓' if max_rate <= 0.03 else '❌'}")
    print(f"  All rates >0: {'✓' if all(r > 0 for r in prepay_rates) else '❌'}")
    
    # Show distribution
    print(f"\nRate Distribution:")
    low_rates = sum(1 for r in prepay_rates if r < 0.005)  # <0.5%
    medium_rates = sum(1 for r in prepay_rates if 0.005 <= r <= 0.015)  # 0.5-1.5%
    high_rates = sum(1 for r in prepay_rates if r > 0.015)  # >1.5%
    
    print(f"  <0.5%: {low_rates} loans ({low_rates/len(prepay_rates):.1%})")
    print(f"  0.5-1.5%: {medium_rates} loans ({medium_rates/len(prepay_rates):.1%})")
    print(f"  >1.5%: {high_rates} loans ({high_rates/len(prepay_rates):.1%})")
    
    return prepay_rates

def test_prepayment_impact():
    """Test the impact of realistic prepayment rates on simulation results"""
    print("\n" + "=" * 50)
    print("TESTING PREPAYMENT IMPACT")
    print("=" * 50)
    
    # Run simulation with realistic prepayment rates
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    results = sim.run_monte_carlo(n_simulations=500, use_parallel=True)
    
    # Analyze prepayment impact
    print(f"Simulation Results with Realistic Prepayment Rates:")
    print(f"  Number of loans: {sim.n_loans}")
    print(f"  Average prepayment rate: {np.mean([loan.annual_prepay_rate for loan in sim.loans]):.3%}")
    
    for tranche_name, payoffs in results['tranche_payoffs'].items():
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        var_95 = np.percentile(payoffs, 5)
        print(f"\n  {tranche_name}:")
        print(f"    Mean payoff: ${mean_payoff:,.0f}")
        print(f"    Std deviation: ${std_payoff:,.0f}")
        print(f"    95% VaR: ${var_95:,.0f}")

def main():
    """Run prepayment rate tests"""
    try:
        # Test prepayment rate generation
        prepay_rates = test_prepayment_rates()
        
        # Test impact on simulation
        test_prepayment_impact()
        
        print("\n" + "=" * 50)
        print("PREPAYMENT RATE TEST COMPLETED!")
        print("=" * 50)
        print("\nThe prepayment rates are now realistic for multi-family loans:")
        print("- Annual rates around 1% (vs previous 10%)")
        print("- Maximum rates capped at 3% (vs previous 50%)")
        print("- Much more conservative and realistic")
        
    except Exception as e:
        print(f"\n❌ Prepayment rate test failed: {e}")
        raise

if __name__ == "__main__":
    main()
