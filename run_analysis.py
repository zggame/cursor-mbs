#!/usr/bin/env python3
"""
Simple MBS Analysis Runner
==========================

This script runs the MBS correlation analysis directly without needing Jupyter.
Just run: python run_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting MBS Correlation Analysis")
    print("=" * 50)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Find analysis files
    analysis_dir = Path("analysis_output")
    csv_files = list(analysis_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in analysis_output directory")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Load data
    try:
        # Load summary data
        summary_files = [f for f in csv_files if 'summary' in f.name and 'full' in f.name]
        if summary_files:
            latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
            summary_df = pd.read_csv(latest_summary)
            print(f"✅ Loaded summary data: {latest_summary.name}")
        else:
            print("⚠️ No summary files found")
            return
            
        # Load path data
        path_files = [f for f in csv_files if 'paths' in f.name and 'full' in f.name]
        if path_files:
            latest_paths = max(path_files, key=lambda x: x.stat().st_mtime)
            paths_df = pd.read_csv(latest_paths)
            print(f"✅ Loaded path data: {latest_paths.name}")
        else:
            print("⚠️ No path files found")
            return
            
        # Load loan data
        loan_files = [f for f in csv_files if 'loan_information' in f.name and not 'summary' in f.name]
        if loan_files:
            latest_loans = max(loan_files, key=lambda x: x.stat().st_mtime)
            loans_df = pd.read_csv(latest_loans)
            print(f"✅ Loaded loan data: {latest_loans.name}")
        else:
            print("⚠️ No loan files found")
            return
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Display basic info
    print(f"\n📊 Summary Data Shape: {summary_df.shape}")
    print(f"📈 Path Data Shape: {paths_df.shape}")
    print(f"🏠 Loan Data Shape: {loans_df.shape}")
    
    # Show correlations tested
    correlations = sorted(summary_df['correlation'].unique())
    print(f"\n🔍 Correlations tested: {correlations}")
    
    # Show tranches
    tranches = summary_df['tranche'].unique()
    print(f"🏦 Tranches: {list(tranches)}")
    
    # Basic analysis
    print("\n" + "="*50)
    print("📋 BASIC ANALYSIS RESULTS")
    print("="*50)
    
    # Mezzanine tranche analysis
    mezz_data = summary_df[summary_df['tranche'] == 'Mezzanine']
    print("\n🏦 MEZZANINE TRANCHE PERFORMANCE:")
    for _, row in mezz_data.iterrows():
        print(f"  Correlation {row['correlation']}:")
        print(f"    Average Payoff: ${row['avg_payoff']:,.0f}")
        print(f"    Worst 1% Payoff: ${row['worst_1pct_avg_payoff']:,.0f}")
        print(f"    Average Loss %: {row['avg_loss_pct']:.2%}")
        print(f"    Average Default Rate: {row['avg_default_rate']:.2%}")
        print()
    
    # Create visualization
    print("📈 Creating visualizations...")
    
    # Correlation impact plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Average payoff vs correlation
    for i, tranche in enumerate(tranches):
        tranche_data = summary_df[summary_df['tranche'] == tranche]
        axes[0,0].plot(tranche_data['correlation'], tranche_data['avg_payoff'], 
                       'o-', label=tranche, color=colors[i], linewidth=2, markersize=8)
    
    axes[0,0].set_title('Average Payoff vs Correlation', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Correlation')
    axes[0,0].set_ylabel('Average Payoff ($)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Worst 1% payoff vs correlation
    for i, tranche in enumerate(tranches):
        tranche_data = summary_df[summary_df['tranche'] == tranche]
        axes[0,1].plot(tranche_data['correlation'], tranche_data['worst_1pct_avg_payoff'], 
                       's--', label=tranche, color=colors[i], linewidth=2, markersize=8)
    
    axes[0,1].set_title('Worst 1% Payoff vs Correlation', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Correlation')
    axes[0,1].set_ylabel('Worst 1% Payoff ($)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Loss percentage vs correlation
    for i, tranche in enumerate(tranches):
        tranche_data = summary_df[summary_df['tranche'] == tranche]
        axes[1,0].plot(tranche_data['correlation'], tranche_data['avg_loss_pct'], 
                       '^:', label=tranche, color=colors[i], linewidth=2, markersize=8)
    
    axes[1,0].set_title('Average Loss % vs Correlation', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Correlation')
    axes[1,0].set_ylabel('Average Loss Percentage')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Default rate vs correlation
    for i, tranche in enumerate(tranches):
        tranche_data = summary_df[summary_df['tranche'] == tranche]
        axes[1,1].plot(tranche_data['correlation'], tranche_data['avg_default_rate'], 
                       'd-.', label=tranche, color=colors[i], linewidth=2, markersize=8)
    
    axes[1,1].set_title('Average Default Rate vs Correlation', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Correlation')
    axes[1,1].set_ylabel('Average Default Rate')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved correlation analysis plot: correlation_analysis_results.png")
    
    # Risk metrics
    print("\n⚠️ RISK METRICS (Mezzanine Tranche):")
    risk_metrics = {}
    
    for corr in correlations:
        corr_data = paths_df[paths_df['correlation'] == corr]
        mezz_payoffs = corr_data[corr_data['tranche'] == 'Mezzanine']['payoff']
        
        var_95 = np.percentile(mezz_payoffs, 5)
        var_99 = np.percentile(mezz_payoffs, 1)
        es_95 = np.mean(mezz_payoffs[mezz_payoffs <= var_95])
        es_99 = np.mean(mezz_payoffs[mezz_payoffs <= var_99])
        
        risk_metrics[corr] = {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'mean': np.mean(mezz_payoffs),
            'std': np.std(mezz_payoffs)
        }
        
        print(f"  Correlation {corr}:")
        print(f"    95% VaR: ${var_95:,.0f}")
        print(f"    99% VaR: ${var_99:,.0f}")
        print(f"    95% ES: ${es_95:,.0f}")
        print(f"    99% ES: ${es_99:,.0f}")
        print(f"    Mean: ${np.mean(mezz_payoffs):,.0f}")
        print(f"    Std Dev: ${np.std(mezz_payoffs):,.0f}")
        print()
    
    # Loan pool summary
    print("🏠 LOAN POOL SUMMARY:")
    print(f"  Total loans: {len(loans_df)}")
    print(f"  Total principal: ${loans_df['principal'].sum():,.0f}")
    print(f"  Average loan size: ${loans_df['principal'].mean():,.0f}")
    print(f"  Average annual rate: {loans_df['annual_rate'].mean():.2%}")
    print(f"  Average term: {loans_df['term_years'].mean():.1f} years")
    print(f"  Average default rate: {loans_df['annual_default_rate'].mean():.2%}")
    
    print("\n✅ Analysis completed successfully!")
    print("📊 Check 'correlation_analysis_results.png' for visualizations")

if __name__ == "__main__":
    main()
