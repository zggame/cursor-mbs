#!/usr/bin/env python3
"""
Investigate First Month Wipeout
===============================

This script investigates how subordinate tranches could be wiped out in the first month.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("🔍 INVESTIGATING FIRST MONTH WIPEOUT")
    print("=" * 60)
    
    # Load data
    paths_df = pd.read_csv('analysis_output/correlation_analysis_paths_full_20250902_214006.csv')
    loans_df = pd.read_csv('analysis_output/loan_information_full_20250902_213318.csv')
    
    # Analyze subordinate tranche wipeouts
    subordinate_data = paths_df[paths_df['tranche'] == 'Subordinate']
    
    print("📊 SUBORDINATE TRANCHE WIPEOUT ANALYSIS:")
    print("-" * 50)
    
    total_principal = loans_df['principal'].sum()
    print(f"Total loan pool principal: ${total_principal:,.0f}")
    
    # Default tranche parameters
    attachment_point = 0.01  # 1%
    detachment_point = 0.05  # 5%
    
    subordinate_principal = total_principal * attachment_point
    print(f"Subordinate tranche principal: ${subordinate_principal:,.0f} ({attachment_point:.1%} of pool)")
    
    # Analyze by correlation
    for corr in sorted(subordinate_data['correlation'].unique()):
        corr_data = subordinate_data[subordinate_data['correlation'] == corr]
        payoffs = corr_data['payoff']
        
        wipeouts = (payoffs == 0).sum()
        total_paths = len(payoffs)
        wipeout_rate = (wipeouts / total_paths) * 100
        
        print(f"\nCorrelation {corr}:")
        print(f"  Total paths: {total_paths}")
        print(f"  Wipeouts ($0 payoff): {wipeouts} ({wipeout_rate:.1f}%)")
        print(f"  Non-zero payoffs: {total_paths - wipeouts}")
        
        # Calculate what loss would be needed to wipe out subordinate tranche
        # Subordinate tranche is wiped out when cumulative loss >= 1% of total principal
        loss_threshold = total_principal * attachment_point
        print(f"  Loss threshold to wipe out: ${loss_threshold:,.0f} ({attachment_point:.1%} of pool)")
        
        # Check if this is possible in first month
        print(f"  Average loan size: ${loans_df['principal'].mean():,.0f}")
        print(f"  Max loan size: ${loans_df['principal'].max():,.0f}")
        
        # Calculate how many large loans defaulting would wipe out subordinate
        max_loan = loans_df['principal'].max()
        avg_lgd = loans_df['annual_lgd_rate'].mean()
        first_month_loss_from_max_loan = max_loan * avg_lgd
        print(f"  First month loss from largest loan default: ${first_month_loss_from_max_loan:,.0f}")
        
        if first_month_loss_from_max_loan >= loss_threshold:
            print(f"  ⚠️  SINGLE LARGE LOAN DEFAULT COULD WIPEOUT SUBORDINATE!")
        else:
            loans_needed = loss_threshold / first_month_loss_from_max_loan
            print(f"  Number of large loans needed to wipeout: {loans_needed:.1f}")
    
    # Check loan pool characteristics
    print(f"\n🏠 LOAN POOL CHARACTERISTICS:")
    print("-" * 40)
    print(f"Number of loans: {len(loans_df)}")
    print(f"Average loan size: ${loans_df['principal'].mean():,.0f}")
    print(f"Median loan size: ${loans_df['principal'].median():,.0f}")
    print(f"Min loan size: ${loans_df['principal'].min():,.0f}")
    print(f"Max loan size: ${loans_df['principal'].max():,.0f}")
    print(f"Average default rate: {loans_df['annual_default_rate'].mean():.2%}")
    print(f"Average LGD rate: {loans_df['annual_lgd_rate'].mean():.2%}")
    
    # Check for concentration risk
    print(f"\n⚠️  CONCENTRATION RISK ANALYSIS:")
    print("-" * 40)
    
    # Sort loans by size
    sorted_loans = loans_df.sort_values('principal', ascending=False)
    
    # Calculate cumulative percentage
    cumulative_principal = sorted_loans['principal'].cumsum()
    cumulative_percentage = cumulative_principal / total_principal * 100
    
    # Find how many loans make up 1% of pool
    loans_for_1pct = (cumulative_percentage <= 1).sum()
    principal_for_1pct = cumulative_principal.iloc[loans_for_1pct-1] if loans_for_1pct > 0 else 0
    
    print(f"Loans needed to reach 1% of pool: {loans_for_1pct}")
    print(f"Principal in top 1%: ${principal_for_1pct:,.0f}")
    
    # Check top 5 loans
    top_5_loans = sorted_loans.head(5)
    top_5_principal = top_5_loans['principal'].sum()
    top_5_percentage = top_5_principal / total_principal * 100
    
    print(f"Top 5 loans principal: ${top_5_principal:,.0f} ({top_5_percentage:.1f}% of pool)")
    
    # Calculate potential first month loss from top 5 loans
    top_5_avg_lgd = top_5_loans['annual_lgd_rate'].mean()
    top_5_avg_default = top_5_loans['annual_default_rate'].mean()
    
    # Expected loss if all top 5 default in first month
    expected_loss_top_5 = top_5_principal * top_5_avg_lgd
    print(f"Expected loss if top 5 default: ${expected_loss_top_5:,.0f}")
    
    if expected_loss_top_5 >= loss_threshold:
        print(f"🚨 TOP 5 LOANS DEFAULTING COULD WIPEOUT SUBORDINATE!")
    else:
        print(f"Top 5 loans defaulting would not wipeout subordinate")
    
    # Check correlation impact
    print(f"\n🔗 CORRELATION IMPACT:")
    print("-" * 30)
    print("Higher correlation means:")
    print("- More loans default together")
    print("- Less diversification benefit")
    print("- More concentrated losses")
    print("- Higher probability of wipeout")
    
    # Simulate worst case scenario
    print(f"\n💀 WORST CASE SCENARIO:")
    print("-" * 30)
    
    # Assume all largest loans default in first month
    largest_loans = sorted_loans.head(10)  # Top 10 largest
    worst_case_loss = largest_loans['principal'].sum() * largest_loans['annual_lgd_rate'].mean()
    worst_case_percentage = worst_case_loss / total_principal * 100
    
    print(f"Worst case loss (top 10 default): ${worst_case_loss:,.0f} ({worst_case_percentage:.1f}% of pool)")
    
    if worst_case_percentage >= attachment_point:
        print(f"🚨 WORST CASE WOULD WIPEOUT SUBORDINATE!")
        print(f"Subordinate would lose: ${subordinate_principal:,.0f}")
        print(f"Mezzanine would lose: ${(worst_case_percentage - attachment_point) / (detachment_point - attachment_point) * (total_principal * (detachment_point - attachment_point)):,.0f}")
    else:
        print(f"Worst case would not wipeout subordinate")
    
    print(f"\n💡 CONCLUSION:")
    print("-" * 30)
    print("First month wipeouts are possible if:")
    print("1. Several large loans default simultaneously")
    print("2. High correlation leads to concentrated defaults")
    print("3. Loss Given Default is high")
    print("4. Subordinate tranche is thin (1% of pool)")
    print("5. Loan pool has concentration risk")

if __name__ == "__main__":
    main()
