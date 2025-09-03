#!/usr/bin/env python3
"""
MBS Correlation Analysis Script
================================

This script can be run with the Jupyter extension in VS Code/Cursor.
Just add #%% at the beginning of each cell to make it interactive.

Run this file and use the Jupyter extension to execute cells interactively.
"""

#%% [markdown]
# # MBS Correlation Analysis Explorer
# 
# This script explores the correlation impact analysis results and loan data from the MBS simulation.
# 
# ## Overview
# - **Correlation levels tested**: 0.01, 0.1, 0.2, 0.3, 0.4
# - **Simulation paths**: 500 (full analysis)
# - **Loan pool size**: 100 loans
# - **Tranches**: Subordinate, Mezzanine, Senior

#%%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("✅ Libraries imported successfully")

#%%
# Find the most recent analysis files
analysis_dir = Path("analysis_output")
csv_files = list(analysis_dir.glob("*.csv"))
png_files = list(analysis_dir.glob("*.png"))

print(f"Found {len(csv_files)} CSV files and {len(png_files)} PNG files")
print("\nCSV files:")
for f in csv_files:
    print(f"  - {f.name}")
print("\nPNG files:")
for f in png_files:
    print(f"  - {f.name}")

#%%
# Load the most recent summary data
summary_files = [f for f in csv_files if 'summary' in f.name and 'full' in f.name]
latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
summary_df = pd.read_csv(latest_summary)

print(f"📊 Loaded summary data: {latest_summary.name}")
print(f"Shape: {summary_df.shape}")
summary_df.head()

#%%
# Load the most recent path data
path_files = [f for f in csv_files if 'paths' in f.name and 'full' in f.name]
latest_paths = max(path_files, key=lambda x: x.stat().st_mtime)
paths_df = pd.read_csv(latest_paths)

print(f"📈 Loaded path data: {latest_paths.name}")
print(f"Shape: {paths_df.shape}")
paths_df.head()

#%%
# Load loan information
loan_files = [f for f in csv_files if 'loan_information' in f.name and not 'summary' in f.name]
latest_loans = max(loan_files, key=lambda x: x.stat().st_mtime)
loans_df = pd.read_csv(latest_loans)

print(f"🏠 Loaded loan data: {latest_loans.name}")
print(f"Shape: {loans_df.shape}")
loans_df.head()

#%% [markdown]
# ## 2. Explore Loan Pool Characteristics

#%%
# Loan pool summary statistics
print("🏠 LOAN POOL CHARACTERISTICS")
print("=" * 50)

print(f"Total loans: {len(loans_df)}")
print(f"Total principal: ${loans_df['principal'].sum():,.0f}")
print(f"Average loan size: ${loans_df['principal'].mean():,.0f}")
print(f"Min loan size: ${loans_df['principal'].min():,.0f}")
print(f"Max loan size: ${loans_df['principal'].max():,.0f}")
print(f"Standard deviation: ${loans_df['principal'].std():,.0f}")
print(f"Average annual rate: {loans_df['annual_rate'].mean():.2%}")
print(f"Average term: {loans_df['term_years'].mean():.1f} years")

#%%
# Visualize loan distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loan size distribution
axes[0,0].hist(loans_df['principal'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Loan Size Distribution')
axes[0,0].set_xlabel('Principal Amount ($)')
axes[0,0].set_ylabel('Number of Loans')
axes[0,0].grid(True, alpha=0.3)

# Interest rate distribution
axes[0,1].hist(loans_df['annual_rate'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].set_title('Interest Rate Distribution')
axes[0,1].set_xlabel('Annual Rate')
axes[0,1].set_ylabel('Number of Loans')
axes[0,1].grid(True, alpha=0.3)

# Term distribution
axes[1,0].hist(loans_df['term_years'], bins=10, alpha=0.7, color='salmon', edgecolor='black')
axes[1,0].set_title('Loan Term Distribution')
axes[1,0].set_xlabel('Term (Years)')
axes[1,0].set_ylabel('Number of Loans')
axes[1,0].grid(True, alpha=0.3)

# Default rate distribution
axes[1,1].hist(loans_df['annual_default_rate'], bins=15, alpha=0.7, color='gold', edgecolor='black')
axes[1,1].set_title('Default Rate Distribution')
axes[1,1].set_xlabel('Annual Default Rate')
axes[1,1].set_ylabel('Number of Loans')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 3. Correlation Impact Analysis

#%%
# Summary statistics by correlation and tranche
print("📊 CORRELATION IMPACT SUMMARY")
print("=" * 60)

# Pivot table for average payoffs
pivot_payoff = summary_df.pivot_table(
    values='avg_payoff', 
    index='correlation', 
    columns='tranche', 
    aggfunc='mean'
)

print("\nAverage Payoffs by Correlation and Tranche:")
print(pivot_payoff.round(0))

# Pivot table for worst 1% payoffs
pivot_worst1 = summary_df.pivot_table(
    values='worst_1pct_avg_payoff', 
    index='correlation', 
    columns='tranche', 
    aggfunc='mean'
)

print("\nWorst 1% Payoffs by Correlation and Tranche:")
print(pivot_worst1.round(0))

#%%
# Visualize correlation impact on payoffs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

tranches = ['Subordinate', 'Mezzanine', 'Senior']
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
plt.show()

#%% [markdown]
# ## 4. Path-Level Analysis

#%%
# Analyze path-level data
print("📈 PATH-LEVEL ANALYSIS")
print("=" * 40)

print(f"Total simulation paths: {len(paths_df)}")
print(f"Paths per correlation: {len(paths_df) // len(paths_df['correlation'].unique())}")
print(f"Correlations tested: {sorted(paths_df['correlation'].unique())}")
print(f"Tranches: {paths_df['tranche'].unique()}")

# Path-level statistics by correlation
path_stats = paths_df.groupby(['correlation', 'tranche']).agg({
    'payoff': ['mean', 'std', 'min', 'max'],
    'loss_pct': ['mean', 'std', 'min', 'max']
}).round(2)

print("\nPath-level statistics:")
print(path_stats)

#%%
# Distribution of payoffs by correlation
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

correlations = sorted(paths_df['correlation'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(correlations)))

for i, corr in enumerate(correlations):
    corr_data = paths_df[paths_df['correlation'] == corr]
    
    # Mezzanine tranche payoffs
    mezz_data = corr_data[corr_data['tranche'] == 'Mezzanine']['payoff']
    axes[i].hist(mezz_data, bins=30, alpha=0.7, density=True, 
                 color=colors[i], label=f'Corr={corr}')
    axes[i].set_title(f'Correlation {corr} - Mezzanine Payoffs')
    axes[i].set_xlabel('Payoff ($)')
    axes[i].set_ylabel('Density')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 5. Risk Metrics Analysis

#%%
# Calculate and visualize risk metrics
print("⚠️ RISK METRICS ANALYSIS")
print("=" * 40)

# Calculate VaR and Expected Shortfall for each correlation
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

risk_df = pd.DataFrame(risk_metrics).T
print("\nRisk Metrics for Mezzanine Tranche:")
print(risk_df.round(0))

#%%
# Visualize risk metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# VaR comparison
axes[0,0].plot(risk_df.index, risk_df['var_95'], 'o-', label='95% VaR', linewidth=2, markersize=8)
axes[0,0].plot(risk_df.index, risk_df['var_99'], 's--', label='99% VaR', linewidth=2, markersize=8)
axes[0,0].set_title('Value at Risk vs Correlation', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Correlation')
axes[0,0].set_ylabel('VaR ($)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Expected Shortfall comparison
axes[0,1].plot(risk_df.index, risk_df['es_95'], 'o-', label='95% ES', linewidth=2, markersize=8)
axes[0,1].plot(risk_df.index, risk_df['es_99'], 's--', label='99% ES', linewidth=2, markersize=8)
axes[0,1].set_title('Expected Shortfall vs Correlation', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Correlation')
axes[0,1].set_ylabel('Expected Shortfall ($)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Mean and Standard Deviation
axes[1,0].plot(risk_df.index, risk_df['mean'], 'o-', label='Mean', linewidth=2, markersize=8)
axes[1,0].set_title('Mean Payoff vs Correlation', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Correlation')
axes[1,0].set_ylabel('Mean Payoff ($)')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(risk_df.index, risk_df['std'], 's--', label='Std Dev', linewidth=2, markersize=8)
axes[1,1].set_title('Standard Deviation vs Correlation', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Correlation')
axes[1,1].set_ylabel('Standard Deviation ($)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 6. Interactive Analysis Functions

#%%
def analyze_correlation_impact(correlation_value):
    """Analyze the impact of a specific correlation value"""
    
    print(f"🔍 ANALYSIS FOR CORRELATION {correlation_value}")
    print("=" * 50)
    
    # Filter data for this correlation
    corr_data = summary_df[summary_df['correlation'] == correlation_value]
    
    if len(corr_data) == 0:
        print(f"❌ No data found for correlation {correlation_value}")
        return
    
    # Display tranche performance
    for _, row in corr_data.iterrows():
        print(f"\n{row['tranche']} Tranche:")
        print(f"  Average Payoff: ${row['avg_payoff']:,.0f}")
        print(f"  Worst 5% Payoff: ${row['worst_5pct_avg_payoff']:,.0f}")
        print(f"  Worst 1% Payoff: ${row['worst_1pct_avg_payoff']:,.0f}")
        print(f"  Average Loss %: {row['avg_loss_pct']:.2%}")
        print(f"  Average Default Rate: {row['avg_default_rate']:.2%}")
    
    # Path-level analysis
    path_corr_data = paths_df[paths_df['correlation'] == correlation_value]
    mezz_payoffs = path_corr_data[path_corr_data['tranche'] == 'Mezzanine']['payoff']
    
    print(f"\n📈 Path-level Statistics (Mezzanine):")
    print(f"  Number of paths: {len(mezz_payoffs)}")
    print(f"  Mean payoff: ${np.mean(mezz_payoffs):,.0f}")
    print(f"  Std deviation: ${np.std(mezz_payoffs):,.0f}")
    print(f"  Min payoff: ${np.min(mezz_payoffs):,.0f}")
    print(f"  Max payoff: ${np.max(mezz_payoffs):,.0f}")
    print(f"  95% VaR: ${np.percentile(mezz_payoffs, 5):,.0f}")
    print(f"  99% VaR: ${np.percentile(mezz_payoffs, 1):,.0f}")

# Example usage
analyze_correlation_impact(0.2)

#%%
def compare_tranches(correlation_value):
    """Compare performance across all tranches for a given correlation"""
    
    print(f"🔄 TRANCHE COMPARISON FOR CORRELATION {correlation_value}")
    print("=" * 60)
    
    corr_data = summary_df[summary_df['correlation'] == correlation_value]
    
    if len(corr_data) == 0:
        print(f"❌ No data found for correlation {correlation_value}")
        return
    
    # Create comparison table
    comparison_data = []
    for _, row in corr_data.iterrows():
        comparison_data.append({
            'Tranche': row['tranche'],
            'Avg Payoff': f"${row['avg_payoff']:,.0f}",
            'Worst 1%': f"${row['worst_1pct_avg_payoff']:,.0f}",
            'Loss %': f"{row['avg_loss_pct']:.2%}",
            'Default Rate': f"{row['avg_default_rate']:.2%}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['avg_payoff', 'worst_1pct_avg_payoff', 'avg_loss_pct', 'avg_default_rate']
    titles = ['Average Payoff', 'Worst 1% Payoff', 'Average Loss %', 'Average Default Rate']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = corr_data[metric].values
        tranches = corr_data['tranche'].values
        
        axes[i//2, i%2].bar(tranches, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[i//2, i%2].set_title(f'{title} by Tranche')
        axes[i//2, i%2].set_ylabel('Value')
        axes[i//2, i%2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            if metric in ['avg_loss_pct', 'avg_default_rate']:
                axes[i//2, i%2].text(j, v, f'{v:.2%}', ha='center', va='bottom')
            else:
                axes[i//2, i%2].text(j, v, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Example usage
compare_tranches(0.3)

#%% [markdown]
# ## 7. Summary and Key Insights

#%%
# Generate summary insights
print("📋 KEY INSIGHTS FROM CORRELATION ANALYSIS")
print("=" * 60)

# Correlation sensitivity analysis
mezz_data = summary_df[summary_df['tranche'] == 'Mezzanine']
baseline_payoff = mezz_data[mezz_data['correlation'] == 0.01]['avg_payoff'].iloc[0]

print("\n🔍 Correlation Sensitivity (Mezzanine Tranche):")
for _, row in mezz_data.iterrows():
    pct_change = (row['avg_payoff'] - baseline_payoff) / baseline_payoff * 100
    print(f"  Correlation {row['correlation']}: {pct_change:+.1f}% change from baseline")

# Risk ranking by correlation
print("\n⚠️ Risk Ranking by Correlation (Mezzanine):")
worst_1pct_ranking = mezz_data.sort_values('worst_1pct_avg_payoff')
for i, (_, row) in enumerate(worst_1pct_ranking.iterrows()):
    print(f"  {i+1}. Correlation {row['correlation']}: ${row['worst_1pct_avg_payoff']:,.0f}")

# Tranche performance comparison
print("\n🏆 Best Performing Tranche by Metric:")
for corr in sorted(summary_df['correlation'].unique()):
    corr_data = summary_df[summary_df['correlation'] == corr]
    best_tranche = corr_data.loc[corr_data['avg_payoff'].idxmax(), 'tranche']
    print(f"  Correlation {corr}: {best_tranche} tranche has highest average payoff")

print("\n✅ Analysis complete! Use the interactive functions above for deeper exploration.")

#%%
# Interactive exploration - you can modify these values
print("🎯 INTERACTIVE EXPLORATION")
print("=" * 40)
print("Try different correlation values:")
print("Available correlations:", sorted(paths_df['correlation'].unique()))

# Example: Analyze correlation 0.4
analyze_correlation_impact(0.4)

# Example: Compare tranches for correlation 0.1
compare_tranches(0.1)

#%%
# Analyze Subordinate Tranche Wipeouts
print("🔍 SUBORDINATE TRANCHE WIPEOUT ANALYSIS")
print("=" * 60)

# Load path-level data for detailed analysis
paths_df = pd.read_csv('analysis_output/correlation_analysis_paths_full_20250902_214006.csv')

# Analyze subordinate tranche payoffs by correlation
subordinate_data = paths_df[paths_df['tranche'] == 'Subordinate']

print("📊 SUBORDINATE TRANCHE PAYOFF ANALYSIS:")
print("-" * 50)

for corr in sorted(subordinate_data['correlation'].unique()):
    corr_data = subordinate_data[subordinate_data['correlation'] == corr]
    payoffs = corr_data['payoff']
    
    # Count wipeouts (payoffs = 0)
    wipeouts = (payoffs == 0).sum()
    total_paths = len(payoffs)
    wipeout_rate = (wipeouts / total_paths) * 100
    
    # Calculate statistics
    mean_payoff = np.mean(payoffs)
    median_payoff = np.median(payoffs)
    min_payoff = np.min(payoffs)
    max_payoff = np.max(payoffs)
    
    print(f"\nCorrelation {corr}:")
    print(f"  Total paths: {total_paths}")
    print(f"  Wipeouts ($0 payoff): {wipeouts} ({wipeout_rate:.1f}%)")
    print(f"  Non-zero payoffs: {total_paths - wipeouts}")
    print(f"  Mean payoff: ${mean_payoff:,.0f}")
    print(f"  Median payoff: ${median_payoff:,.0f}")
    print(f"  Min payoff: ${min_payoff:,.0f}")
    print(f"  Max payoff: ${max_payoff:,.0f}")

# Visualize wipeout rates
print("\n📈 WIPEOUT RATE VISUALIZATION")
print("-" * 50)

# Calculate wipeout rates for all correlations
wipeout_rates = []
correlations = sorted(subordinate_data['correlation'].unique())

for corr in correlations:
    corr_data = subordinate_data[subordinate_data['correlation'] == corr]
    payoffs = corr_data['payoff']
    wipeout_rate = (payoffs == 0).mean() * 100
    wipeout_rates.append(wipeout_rate)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Wipeout rate vs correlation
ax1.plot(correlations, wipeout_rates, 'ro-', linewidth=2, markersize=8)
ax1.set_title('Subordinate Tranche Wipeout Rate vs Correlation', fontsize=14, fontweight='bold')
ax1.set_xlabel('Correlation')
ax1.set_ylabel('Wipeout Rate (%)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Add value labels
for i, (corr, rate) in enumerate(zip(correlations, wipeout_rates)):
    ax1.annotate(f'{rate:.1f}%', (corr, rate), textcoords="offset points", 
                xytext=(0,10), ha='center')

# Distribution of payoffs for correlation 0.3 (example)
corr_03_data = subordinate_data[subordinate_data['correlation'] == 0.3]['payoff']
ax2.hist(corr_03_data, bins=20, alpha=0.7, color='red', edgecolor='black')
ax2.set_title('Subordinate Payoff Distribution (Correlation 0.3)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Payoff ($)')
ax2.set_ylabel('Number of Paths')
ax2.grid(True, alpha=0.3)

# Add wipeout count annotation
wipeouts_03 = (corr_03_data == 0).sum()
total_03 = len(corr_03_data)
ax2.annotate(f'Wipeouts: {wipeouts_03}/{total_03}\n({wipeouts_03/total_03*100:.1f}%)', 
            xy=(0.7, 0.8), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Summary statistics
print("\n📋 SUMMARY:")
print("-" * 30)
print("Subordinate tranche wipeout rates by correlation:")
for corr, rate in zip(correlations, wipeout_rates):
    print(f"  Correlation {corr}: {rate:.1f}% wipeout rate")

# Find correlation with highest wipeout rate
max_wipeout_idx = np.argmax(wipeout_rates)
print(f"\n🏆 Highest wipeout rate: Correlation {correlations[max_wipeout_idx]} ({wipeout_rates[max_wipeout_idx]:.1f}%)")

# Find correlation with lowest wipeout rate  
min_wipeout_idx = np.argmin(wipeout_rates)
print(f"🥇 Lowest wipeout rate: Correlation {correlations[min_wipeout_idx]} ({wipeout_rates[min_wipeout_idx]:.1f}%)")

print("\n💡 INSIGHTS:")
print("-" * 30)
print("The subordinate tranche is the first-loss piece that absorbs losses.")
print("Higher correlation typically leads to more concentrated losses,")
print("which can result in more frequent wipeouts of the subordinate tranche.")

#%%
# Compare Wipeout Rates Across All Tranches
print("🔄 COMPARING WIPEOUT RATES ACROSS ALL TRANCHES")
print("=" * 60)

# Analyze wipeout rates for all tranches
tranche_wipeout_data = {}

for tranche in ['Subordinate', 'Mezzanine', 'Senior']:
    tranche_data = paths_df[paths_df['tranche'] == tranche]
    wipeout_rates = []
    
    for corr in sorted(tranche_data['correlation'].unique()):
        corr_data = tranche_data[tranche_data['correlation'] == corr]
        payoffs = corr_data['payoff']
        wipeout_rate = (payoffs == 0).mean() * 100
        wipeout_rates.append(wipeout_rate)
    
    tranche_wipeout_data[tranche] = wipeout_rates

# Create comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Wipeout rates comparison
correlations = sorted(paths_df['correlation'].unique())
colors = ['red', 'orange', 'green']

for i, (tranche, rates) in enumerate(tranche_wipeout_data.items()):
    ax1.plot(correlations, rates, 'o-', label=tranche, color=colors[i], 
             linewidth=2, markersize=8)

ax1.set_title('Wipeout Rates by Tranche and Correlation', fontsize=14, fontweight='bold')
ax1.set_xlabel('Correlation')
ax1.set_ylabel('Wipeout Rate (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Bar chart comparison for correlation 0.3
corr_03_wipeouts = []
tranche_names = list(tranche_wipeout_data.keys())

for tranche in tranche_names:
    corr_idx = correlations.index(0.3)
    wipeout_rate = tranche_wipeout_data[tranche][corr_idx]
    corr_03_wipeouts.append(wipeout_rate)

bars = ax2.bar(tranche_names, corr_03_wipeouts, color=colors, alpha=0.7)
ax2.set_title('Wipeout Rates at Correlation 0.3', fontsize=14, fontweight='bold')
ax2.set_ylabel('Wipeout Rate (%)')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, rate in zip(bars, corr_03_wipeouts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed comparison
print("\n📊 DETAILED WIPEOUT COMPARISON:")
print("-" * 50)

for corr in correlations:
    print(f"\nCorrelation {corr}:")
    for tranche in tranche_names:
        corr_idx = correlations.index(corr)
        wipeout_rate = tranche_wipeout_data[tranche][corr_idx]
        print(f"  {tranche}: {wipeout_rate:.1f}% wipeout rate")

# Find most vulnerable tranche
print("\n⚠️ RISK ANALYSIS:")
print("-" * 30)

# Calculate average wipeout rates
avg_wipeouts = {}
for tranche, rates in tranche_wipeout_data.items():
    avg_wipeouts[tranche] = np.mean(rates)

# Rank by vulnerability
sorted_tranches = sorted(avg_wipeouts.items(), key=lambda x: x[1], reverse=True)
print("Tranches ranked by vulnerability (average wipeout rate):")
for i, (tranche, rate) in enumerate(sorted_tranches, 1):
    print(f"  {i}. {tranche}: {rate:.1f}% average wipeout rate")

print(f"\n🏆 Most vulnerable: {sorted_tranches[0][0]} ({sorted_tranches[0][1]:.1f}%)")
print(f"🥇 Least vulnerable: {sorted_tranches[-1][0]} ({sorted_tranches[-1][1]:.1f}%)")

print("\n💡 KEY INSIGHTS:")
print("-" * 30)
print("• Subordinate tranche has the highest wipeout risk (first-loss position)")
print("• Mezzanine tranche has moderate wipeout risk")
print("• Senior tranche has the lowest wipeout risk (protected by subordination)")
print("• Higher correlation generally increases wipeout risk for all tranches")

#%%
# Investigate First Month Wipeout Mechanism
print("🔍 FIRST MONTH WIPEOUT MECHANISM")
print("=" * 60)

# Load loan data for analysis
loans_df = pd.read_csv('analysis_output/loan_information_full_20250902_213318.csv')

# Calculate key metrics
total_principal = loans_df['principal'].sum()
attachment_point = 0.01  # 1%
subordinate_principal = total_principal * attachment_point

print(f"🏠 LOAN POOL ANALYSIS:")
print(f"  Total principal: ${total_principal:,.0f}")
print(f"  Subordinate tranche: ${subordinate_principal:,.0f} ({attachment_point:.1%} of pool)")
print(f"  Number of loans: {len(loans_df)}")
print(f"  Average loan size: ${loans_df['principal'].mean():,.0f}")
print(f"  Max loan size: ${loans_df['principal'].max():,.0f}")

# Check concentration risk
sorted_loans = loans_df.sort_values('principal', ascending=False)
top_5_loans = sorted_loans.head(5)
top_5_principal = top_5_loans['principal'].sum()
top_5_percentage = top_5_principal / total_principal * 100

print(f"\n⚠️  CONCENTRATION RISK:")
print(f"  Top 5 loans: ${top_5_principal:,.0f} ({top_5_percentage:.1f}% of pool)")
print(f"  Average LGD rate: {loans_df['annual_lgd_rate'].mean():.2%}")

# Calculate potential first month loss
avg_lgd = loans_df['annual_lgd_rate'].mean()
potential_first_month_loss = top_5_principal * avg_lgd
loss_percentage = potential_first_month_loss / total_principal * 100

print(f"\n💀 FIRST MONTH LOSS SCENARIO:")
print(f"  If top 5 loans default in first month:")
print(f"  Potential loss: ${potential_first_month_loss:,.0f} ({loss_percentage:.1f}% of pool)")
print(f"  Subordinate loss threshold: ${subordinate_principal:,.0f} ({attachment_point:.1%} of pool)")

if potential_first_month_loss >= subordinate_principal:
    print(f"  🚨 THIS WOULD WIPEOUT THE SUBORDINATE TRANCHE!")
    print(f"  Subordinate would lose: ${subordinate_principal:,.0f}")
    print(f"  Mezzanine would lose: ${potential_first_month_loss - subordinate_principal:,.0f}")
else:
    print(f"  This would not wipeout subordinate")

# Visualize loan concentration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Loan size distribution
ax1.hist(loans_df['principal'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(loans_df['principal'].mean(), color='red', linestyle='--', label=f'Mean: ${loans_df["principal"].mean():,.0f}')
ax1.axvline(loans_df['principal'].max(), color='orange', linestyle='--', label=f'Max: ${loans_df["principal"].max():,.0f}')
ax1.set_title('Loan Size Distribution')
ax1.set_xlabel('Principal Amount ($)')
ax1.set_ylabel('Number of Loans')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative principal by loan rank
loan_ranks = range(1, len(sorted_loans) + 1)
cumulative_principal = sorted_loans['principal'].cumsum()
cumulative_percentage = cumulative_principal / total_principal * 100

ax2.plot(loan_ranks, cumulative_percentage, 'b-', linewidth=2)
ax2.axhline(attachment_point * 100, color='red', linestyle='--', label=f'Subordinate threshold ({attachment_point:.1%})')
ax2.axhline(5, color='orange', linestyle='--', label='Top 5 loans threshold')
ax2.set_title('Cumulative Principal by Loan Rank')
ax2.set_xlabel('Loan Rank (by size)')
ax2.set_ylabel('Cumulative Percentage of Pool (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n💡 EXPLANATION:")
print("-" * 30)
print("Subordinate tranches can be wiped out in the first month because:")
print("1. The subordinate tranche is only 1% of the total pool")
print("2. The top 5 loans represent 30.4% of the pool")
print("3. If these large loans default simultaneously (high correlation),")
print("   the loss can exceed the subordinate tranche's protection")
print("4. Higher correlation increases the probability of concentrated defaults")
print("5. This is why correlation 0.3 and 0.4 show higher wipeout rates")

print(f"\n🔗 CORRELATION IMPACT:")
print("-" * 30)
print("• Low correlation (0.01): Defaults are independent, low wipeout risk")
print("• High correlation (0.3-0.4): Defaults cluster together, high wipeout risk")
print("• The subordinate tranche is the 'first-loss' piece that absorbs initial losses")
print("• Once subordinate is wiped out, losses flow to the mezzanine tranche")
