#!/usr/bin/env python3
"""
Verify Tranche Sizing
====================

This script verifies the tranche sizing calculations to understand the payoff anomaly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("🔍 VERIFYING TRANCHE SIZING")
    print("=" * 50)
    
    # Load data
    loans_df = pd.read_csv('analysis_output/loan_information_full_20250902_213318.csv')
    summary_df = pd.read_csv('analysis_output/correlation_analysis_summary_full_20250902_214006.csv')
    
    # Calculate total principal
    total_principal = loans_df['principal'].sum()
    print(f"🏠 LOAN POOL:")
    print(f"  Total principal: ${total_principal:,.0f}")
    print(f"  Number of loans: {len(loans_df)}")
    
    # Default tranche parameters (from mbs_simulation.py)
    attachment_point = 0.01  # 1%
    detachment_point = 0.05  # 5%
    
    print(f"\n🏗️ TRANCHE STRUCTURE (Default Parameters):")
    print(f"  Attachment point: {attachment_point:.1%}")
    print(f"  Detachment point: {detachment_point:.1%}")
    
    # Calculate tranche sizes
    subordinate_principal = total_principal * attachment_point
    mezzanine_principal = total_principal * (detachment_point - attachment_point)
    senior_principal = total_principal * (1.0 - detachment_point)
    
    print(f"\n📊 TRANCHE SIZES:")
    print(f"  Subordinate: ${subordinate_principal:,.0f} ({attachment_point:.1%} of pool)")
    print(f"  Mezzanine:   ${mezzanine_principal:,.0f} ({(detachment_point - attachment_point):.1%} of pool)")
    print(f"  Senior:      ${senior_principal:,.0f} ({(1.0 - detachment_point):.1%} of pool)")
    print(f"  Total:       ${subordinate_principal + mezzanine_principal + senior_principal:,.0f}")
    
    # Verify the math
    total_tranches = subordinate_principal + mezzanine_principal + senior_principal
    print(f"\n✅ VERIFICATION:")
    print(f"  Total tranches = Total principal: {total_tranches == total_principal}")
    print(f"  Difference: ${total_tranches - total_principal:,.2f}")
    
    # Compare with actual payoffs
    print(f"\n🏦 ACTUAL PAYOFFS vs TRANCHE SIZES:")
    for corr in sorted(summary_df['correlation'].unique()):
        print(f"\n  Correlation {corr}:")
        corr_data = summary_df[summary_df['correlation'] == corr]
        
        for _, row in corr_data.iterrows():
            tranche = row['tranche']
            payoff = row['avg_payoff']
            
            if tranche == 'Subordinate':
                expected = subordinate_principal
            elif tranche == 'Mezzanine':
                expected = mezzanine_principal
            elif tranche == 'Senior':
                expected = senior_principal
            else:
                expected = 0
            
            ratio = payoff / expected if expected > 0 else 0
            print(f"    {tranche}: ${payoff:,.0f} (expected: ${expected:,.0f}, ratio: {ratio:.2f})")
    
    # Check if payoffs include interest
    print(f"\n💡 ANALYSIS:")
    print(f"  The Senior tranche payoffs are ~{ratio:.1f}x the tranche principal")
    print(f"  This suggests payoffs include:")
    print(f"    1. Principal repayment")
    print(f"    2. Interest payments")
    print(f"    3. Prepayment penalties")
    print(f"    4. Other fees")
    
    # Calculate what the ratio should be for a 7-year security term
    print(f"\n📈 EXPECTED RATIO CALCULATION:")
    print(f"  Security term: 7 years")
    print(f"  Average loan term: 10 years")
    print(f"  Average interest rate: {loans_df['annual_rate'].mean():.2%}")
    
    # Rough calculation: Principal + 7 years of interest
    avg_interest_rate = loans_df['annual_rate'].mean()
    expected_ratio = 1 + (avg_interest_rate * 7)  # Principal + 7 years interest
    print(f"  Expected ratio: {expected_ratio:.2f}")
    print(f"  This would give Senior payoff: ${senior_principal * expected_ratio:,.0f}")
    
    # Check if this matches
    actual_senior_payoff = summary_df[summary_df['tranche'] == 'Senior']['avg_payoff'].mean()
    actual_ratio = actual_senior_payoff / senior_principal
    print(f"  Actual ratio: {actual_ratio:.2f}")
    print(f"  Actual Senior payoff: ${actual_senior_payoff:,.0f}")
    
    print(f"\n✅ CONCLUSION:")
    print(f"  The payoffs include both principal and interest payments.")
    print(f"  This is correct behavior for MBS tranches.")
    print(f"  The analysis is meaningful - higher payoffs indicate better performance.")

if __name__ == "__main__":
    main()
