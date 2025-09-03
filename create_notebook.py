#!/usr/bin/env python3
"""
Script to create a comprehensive Jupyter notebook for MBS correlation analysis
"""

import json

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MBS Correlation Analysis Explorer\n",
                "\n",
                "This notebook explores the correlation impact analysis results and loan data from the MBS simulation.\n",
                "\n",
                "## Overview\n",
                "- **Correlation levels tested**: 0.01, 0.1, 0.2, 0.3, 0.4\n",
                "- **Simulation paths**: 500 (full analysis)\n",
                "- **Loan pool size**: 100 loans\n",
                "- **Tranches**: Subordinate, Mezzanine, Senior"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set up plotting style\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "\n",
                "print(\"✅ Libraries imported successfully\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Analysis Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find the most recent analysis files\n",
                "analysis_dir = Path(\"analysis_output\")\n",
                "csv_files = list(analysis_dir.glob(\"*.csv\"))\n",
                "png_files = list(analysis_dir.glob(\"*.png\"))\n",
                "\n",
                "print(f\"Found {len(csv_files)} CSV files and {len(png_files)} PNG files\")\n",
                "print(\"\\nCSV files:\")\n",
                "for f in csv_files:\n",
                "    print(f\"  - {f.name}\")\n",
                "print(\"\\nPNG files:\")\n",
                "for f in png_files:\n",
                "    print(f\"  - {f.name}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the most recent summary data\n",
                "summary_files = [f for f in csv_files if 'summary' in f.name and 'full' in f.name]\n",
                "latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)\n",
                "summary_df = pd.read_csv(latest_summary)\n",
                "\n",
                "print(f\"📊 Loaded summary data: {latest_summary.name}\")\n",
                "print(f\"Shape: {summary_df.shape}\")\n",
                "summary_df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the most recent path data\n",
                "path_files = [f for f in csv_files if 'paths' in f.name and 'full' in f.name]\n",
                "latest_paths = max(path_files, key=lambda x: x.stat().st_mtime)\n",
                "paths_df = pd.read_csv(latest_paths)\n",
                "\n",
                "print(f\"📈 Loaded path data: {latest_paths.name}\")\n",
                "print(f\"Shape: {paths_df.shape}\")\n",
                "paths_df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load loan information\n",
                "loan_files = [f for f in csv_files if 'loan_information' in f.name and not 'summary' in f.name]\n",
                "latest_loans = max(loan_files, key=lambda x: x.stat().st_mtime)\n",
                "loans_df = pd.read_csv(latest_loans)\n",
                "\n",
                "print(f\"🏠 Loaded loan data: {latest_loans.name}\")\n",
                "print(f\"Shape: {loans_df.shape}\")\n",
                "loans_df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Explore Loan Pool Characteristics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Loan pool summary statistics\n",
                "print(\"🏠 LOAN POOL CHARACTERISTICS\")\n",
                "print(\"=\" * 50)\n",
                "\n",
                "print(f\"Total loans: {len(loans_df)}\")\n",
                "print(f\"Total principal: ${loans_df['principal'].sum():,.0f}\")\n",
                "print(f\"Average loan size: ${loans_df['principal'].mean():,.0f}\")\n",
                "print(f\"Min loan size: ${loans_df['principal'].min():,.0f}\")\n",
                "print(f\"Max loan size: ${loans_df['principal'].max():,.0f}\")\n",
                "print(f\"Standard deviation: ${loans_df['principal'].std():,.0f}\")\n",
                "print(f\"Average annual rate: {loans_df['annual_rate'].mean():.2%}\")\n",
                "print(f\"Average term: {loans_df['term_years'].mean():.1f} years\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize loan distribution\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                "\n",
                "# Loan size distribution\n",
                "axes[0,0].hist(loans_df['principal'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')\n",
                "axes[0,0].set_title('Loan Size Distribution')\n",
                "axes[0,0].set_xlabel('Principal Amount ($)')\n",
                "axes[0,0].set_ylabel('Number of Loans')\n",
                "axes[0,0].grid(True, alpha=0.3)\n",
                "\n",
                "# Interest rate distribution\n",
                "axes[0,1].hist(loans_df['annual_rate'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')\n",
                "axes[0,1].set_title('Interest Rate Distribution')\n",
                "axes[0,1].set_xlabel('Annual Rate')\n",
                "axes[0,1].set_ylabel('Number of Loans')\n",
                "axes[0,1].grid(True, alpha=0.3)\n",
                "\n",
                "# Term distribution\n",
                "axes[1,0].hist(loans_df['term_years'], bins=10, alpha=0.7, color='salmon', edgecolor='black')\n",
                "axes[1,0].set_title('Loan Term Distribution')\n",
                "axes[1,0].set_xlabel('Term (Years)')\n",
                "axes[1,0].set_ylabel('Number of Loans')\n",
                "axes[1,0].grid(True, alpha=0.3)\n",
                "\n",
                "# Default rate distribution\n",
                "axes[1,1].hist(loans_df['annual_default_rate'], bins=15, alpha=0.7, color='gold', edgecolor='black')\n",
                "axes[1,1].set_title('Default Rate Distribution')\n",
                "axes[1,1].set_xlabel('Annual Default Rate')\n",
                "axes[1,1].set_ylabel('Number of Loans')\n",
                "axes[1,1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Correlation Impact Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Summary statistics by correlation and tranche\n",
                "print(\"📊 CORRELATION IMPACT SUMMARY\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Pivot table for average payoffs\n",
                "pivot_payoff = summary_df.pivot_table(\n",
                "    values='avg_payoff', \n",
                "    index='correlation', \n",
                "    columns='tranche', \n",
                "    aggfunc='mean'\n",
                ")\n",
                "\n",
                "print(\"\\nAverage Payoffs by Correlation and Tranche:\")\n",
                "print(pivot_payoff.round(0))\n",
                "\n",
                "# Pivot table for worst 1% payoffs\n",
                "pivot_worst1 = summary_df.pivot_table(\n",
                "    values='worst_1pct_avg_payoff', \n",
                "    index='correlation', \n",
                "    columns='tranche', \n",
                "    aggfunc='mean'\n",
                ")\n",
                "\n",
                "print(\"\\nWorst 1% Payoffs by Correlation and Tranche:\")\n",
                "print(pivot_worst1.round(0))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize correlation impact on payoffs\n",
                "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
                "\n",
                "tranches = ['Subordinate', 'Mezzanine', 'Senior']\n",
                "colors = ['#1f77b4', '#ff7f0e', '#2ca02c']\n",
                "\n",
                "# Average payoff vs correlation\n",
                "for i, tranche in enumerate(tranches):\n",
                "    tranche_data = summary_df[summary_df['tranche'] == tranche]\n",
                "    axes[0,0].plot(tranche_data['correlation'], tranche_data['avg_payoff'], \n",
                "                   'o-', label=tranche, color=colors[i], linewidth=2, markersize=8)\n",
                "\n",
                "axes[0,0].set_title('Average Payoff vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[0,0].set_xlabel('Correlation')\n",
                "axes[0,0].set_ylabel('Average Payoff ($)')\n",
                "axes[0,0].legend()\n",
                "axes[0,0].grid(True, alpha=0.3)\n",
                "\n",
                "# Worst 1% payoff vs correlation\n",
                "for i, tranche in enumerate(tranches):\n",
                "    tranche_data = summary_df[summary_df['tranche'] == tranche]\n",
                "    axes[0,1].plot(tranche_data['correlation'], tranche_data['worst_1pct_avg_payoff'], \n",
                "                   's--', label=tranche, color=colors[i], linewidth=2, markersize=8)\n",
                "\n",
                "axes[0,1].set_title('Worst 1% Payoff vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[0,1].set_xlabel('Correlation')\n",
                "axes[0,1].set_ylabel('Worst 1% Payoff ($)')\n",
                "axes[0,1].legend()\n",
                "axes[0,1].grid(True, alpha=0.3)\n",
                "\n",
                "# Loss percentage vs correlation\n",
                "for i, tranche in enumerate(tranches):\n",
                "    tranche_data = summary_df[summary_df['tranche'] == tranche]\n",
                "    axes[1,0].plot(tranche_data['correlation'], tranche_data['avg_loss_pct'], \n",
                "                   '^:', label=tranche, color=colors[i], linewidth=2, markersize=8)\n",
                "\n",
                "axes[1,0].set_title('Average Loss % vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[1,0].set_xlabel('Correlation')\n",
                "axes[1,0].set_ylabel('Average Loss Percentage')\n",
                "axes[1,0].legend()\n",
                "axes[1,0].grid(True, alpha=0.3)\n",
                "\n",
                "# Default rate vs correlation\n",
                "for i, tranche in enumerate(tranches):\n",
                "    tranche_data = summary_df[summary_df['tranche'] == tranche]\n",
                "    axes[1,1].plot(tranche_data['correlation'], tranche_data['avg_default_rate'], \n",
                "                   'd-.', label=tranche, color=colors[i], linewidth=2, markersize=8)\n",
                "\n",
                "axes[1,1].set_title('Average Default Rate vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[1,1].set_xlabel('Correlation')\n",
                "axes[1,1].set_ylabel('Average Default Rate')\n",
                "axes[1,1].legend()\n",
                "axes[1,1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Path-Level Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze path-level data\n",
                "print(\"📈 PATH-LEVEL ANALYSIS\")\n",
                "print(\"=\" * 40)\n",
                "\n",
                "print(f\"Total simulation paths: {len(paths_df)}\")\n",
                "print(f\"Paths per correlation: {len(paths_df) // len(paths_df['correlation'].unique())}\")\n",
                "print(f\"Correlations tested: {sorted(paths_df['correlation'].unique())}\")\n",
                "print(f\"Tranches: {paths_df['tranche'].unique()}\")\n",
                "\n",
                "# Path-level statistics by correlation\n",
                "path_stats = paths_df.groupby(['correlation', 'tranche']).agg({\n",
                "    'payoff': ['mean', 'std', 'min', 'max'],\n",
                "    'loss_pct': ['mean', 'std', 'min', 'max']\n",
                "}).round(2)\n",
                "\n",
                "print(\"\\nPath-level statistics:\")\n",
                "print(path_stats)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribution of payoffs by correlation\n",
                "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "correlations = sorted(paths_df['correlation'].unique())\n",
                "colors = plt.cm.viridis(np.linspace(0, 1, len(correlations)))\n",
                "\n",
                "for i, corr in enumerate(correlations):\n",
                "    corr_data = paths_df[paths_df['correlation'] == corr]\n",
                "    \n",
                "    # Mezzanine tranche payoffs\n",
                "    mezz_data = corr_data[corr_data['tranche'] == 'Mezzanine']['payoff']\n",
                "    axes[i].hist(mezz_data, bins=30, alpha=0.7, density=True, \n",
                "                 color=colors[i], label=f'Corr={corr}')\n",
                "    axes[i].set_title(f'Correlation {corr} - Mezzanine Payoffs')\n",
                "    axes[i].set_xlabel('Payoff ($)')\n",
                "    axes[i].set_ylabel('Density')\n",
                "    axes[i].grid(True, alpha=0.3)\n",
                "    axes[i].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Risk Metrics Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate and visualize risk metrics\n",
                "print(\"⚠️ RISK METRICS ANALYSIS\")\n",
                "print(\"=\" * 40)\n",
                "\n",
                "# Calculate VaR and Expected Shortfall for each correlation\n",
                "risk_metrics = {}\n",
                "\n",
                "for corr in correlations:\n",
                "    corr_data = paths_df[paths_df['correlation'] == corr]\n",
                "    mezz_payoffs = corr_data[corr_data['tranche'] == 'Mezzanine']['payoff']\n",
                "    \n",
                "    var_95 = np.percentile(mezz_payoffs, 5)\n",
                "    var_99 = np.percentile(mezz_payoffs, 1)\n",
                "    es_95 = np.mean(mezz_payoffs[mezz_payoffs <= var_95])\n",
                "    es_99 = np.mean(mezz_payoffs[mezz_payoffs <= var_99])\n",
                "    \n",
                "    risk_metrics[corr] = {\n",
                "        'var_95': var_95,\n",
                "        'var_99': var_99,\n",
                "        'es_95': es_95,\n",
                "        'es_99': es_99,\n",
                "        'mean': np.mean(mezz_payoffs),\n",
                "        'std': np.std(mezz_payoffs)\n",
                "    }\n",
                "\n",
                "risk_df = pd.DataFrame(risk_metrics).T\n",
                "print(\"\\nRisk Metrics for Mezzanine Tranche:\")\n",
                "print(risk_df.round(0))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize risk metrics\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                "\n",
                "# VaR comparison\n",
                "axes[0,0].plot(risk_df.index, risk_df['var_95'], 'o-', label='95% VaR', linewidth=2, markersize=8)\n",
                "axes[0,0].plot(risk_df.index, risk_df['var_99'], 's--', label='99% VaR', linewidth=2, markersize=8)\n",
                "axes[0,0].set_title('Value at Risk vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[0,0].set_xlabel('Correlation')\n",
                "axes[0,0].set_ylabel('VaR ($)')\n",
                "axes[0,0].legend()\n",
                "axes[0,0].grid(True, alpha=0.3)\n",
                "\n",
                "# Expected Shortfall comparison\n",
                "axes[0,1].plot(risk_df.index, risk_df['es_95'], 'o-', label='95% ES', linewidth=2, markersize=8)\n",
                "axes[0,1].plot(risk_df.index, risk_df['es_99'], 's--', label='99% ES', linewidth=2, markersize=8)\n",
                "axes[0,1].set_title('Expected Shortfall vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[0,1].set_xlabel('Correlation')\n",
                "axes[0,1].set_ylabel('Expected Shortfall ($)')\n",
                "axes[0,1].legend()\n",
                "axes[0,1].grid(True, alpha=0.3)\n",
                "\n",
                "# Mean and Standard Deviation\n",
                "axes[1,0].plot(risk_df.index, risk_df['mean'], 'o-', label='Mean', linewidth=2, markersize=8)\n",
                "axes[1,0].set_title('Mean Payoff vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[1,0].set_xlabel('Correlation')\n",
                "axes[1,0].set_ylabel('Mean Payoff ($)')\n",
                "axes[1,0].grid(True, alpha=0.3)\n",
                "\n",
                "axes[1,1].plot(risk_df.index, risk_df['std'], 's--', label='Std Dev', linewidth=2, markersize=8)\n",
                "axes[1,1].set_title('Standard Deviation vs Correlation', fontsize=14, fontweight='bold')\n",
                "axes[1,1].set_xlabel('Correlation')\n",
                "axes[1,1].set_ylabel('Standard Deviation ($)')\n",
                "axes[1,1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Interactive Analysis Functions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_correlation_impact(correlation_value):\n",
                "    \"\"\"Analyze the impact of a specific correlation value\"\"\"\n",
                "    \n",
                "    print(f\"🔍 ANALYSIS FOR CORRELATION {correlation_value}\")\n",
                "    print(\"=\" * 50)\n",
                "    \n",
                "    # Filter data for this correlation\n",
                "    corr_data = summary_df[summary_df['correlation'] == correlation_value]\n",
                "    \n",
                "    if len(corr_data) == 0:\n",
                "        print(f\"❌ No data found for correlation {correlation_value}\")\n",
                "        return\n",
                "    \n",
                "    # Display tranche performance\n",
                "    for _, row in corr_data.iterrows():\n",
                "        print(f\"\\n{row['tranche']} Tranche:\")\n",
                "        print(f\"  Average Payoff: ${row['avg_payoff']:,.0f}\")\n",
                "        print(f\"  Worst 5% Payoff: ${row['worst_5pct_avg_payoff']:,.0f}\")\n",
                "        print(f\"  Worst 1% Payoff: ${row['worst_1pct_avg_payoff']:,.0f}\")\n",
                "        print(f\"  Average Loss %: {row['avg_loss_pct']:.2%}\")\n",
                "        print(f\"  Average Default Rate: {row['avg_default_rate']:.2%}\")\n",
                "    \n",
                "    # Path-level analysis\n",
                "    path_corr_data = paths_df[paths_df['correlation'] == correlation_value]\n",
                "    mezz_payoffs = path_corr_data[path_corr_data['tranche'] == 'Mezzanine']['payoff']\n",
                "    \n",
                "    print(f\"\\n📈 Path-level Statistics (Mezzanine):\")\n",
                "    print(f\"  Number of paths: {len(mezz_payoffs)}\")\n",
                "    print(f\"  Mean payoff: ${np.mean(mezz_payoffs):,.0f}\")\n",
                "    print(f\"  Std deviation: ${np.std(mezz_payoffs):,.0f}\")\n",
                "    print(f\"  Min payoff: ${np.min(mezz_payoffs):,.0f}\")\n",
                "    print(f\"  Max payoff: ${np.max(mezz_payoffs):,.0f}\")\n",
                "    print(f\"  95% VaR: ${np.percentile(mezz_payoffs, 5):,.0f}\")\n",
                "    print(f\"  99% VaR: ${np.percentile(mezz_payoffs, 1):,.0f}\")\n",
                "\n",
                "# Example usage\n",
                "analyze_correlation_impact(0.2)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compare_tranches(correlation_value):\n",
                "    \"\"\"Compare performance across all tranches for a given correlation\"\"\"\n",
                "    \n",
                "    print(f\"🔄 TRANCHE COMPARISON FOR CORRELATION {correlation_value}\")\n",
                "    print(\"=\" * 60)\n",
                "    \n",
                "    corr_data = summary_df[summary_df['correlation'] == correlation_value]\n",
                "    \n",
                "    if len(corr_data) == 0:\n",
                "        print(f\"❌ No data found for correlation {correlation_value}\")\n",
                "        return\n",
                "    \n",
                "    # Create comparison table\n",
                "    comparison_data = []\n",
                "    for _, row in corr_data.iterrows():\n",
                "        comparison_data.append({\n",
                "            'Tranche': row['tranche'],\n",
                "            'Avg Payoff': f\"${row['avg_payoff']:,.0f}\",\n",
                "            'Worst 1%': f\"${row['worst_1pct_avg_payoff']:,.0f}\",\n",
                "            'Loss %': f\"{row['avg_loss_pct']:.2%}\",\n",
                "            'Default Rate': f\"{row['avg_default_rate']:.2%}\"\n",
                "        })\n",
                "    \n",
                "    comparison_df = pd.DataFrame(comparison_data)\n",
                "    print(comparison_df.to_string(index=False))\n",
                "    \n",
                "    # Visualize comparison\n",
                "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                "    \n",
                "    metrics = ['avg_payoff', 'worst_1pct_avg_payoff', 'avg_loss_pct', 'avg_default_rate']\n",
                "    titles = ['Average Payoff', 'Worst 1% Payoff', 'Average Loss %', 'Average Default Rate']\n",
                "    \n",
                "    for i, (metric, title) in enumerate(zip(metrics, titles)):\n",
                "        values = corr_data[metric].values\n",
                "        tranches = corr_data['tranche'].values\n",
                "        \n",
                "        axes[i//2, i%2].bar(tranches, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])\n",
                "        axes[i//2, i%2].set_title(f'{title} by Tranche')\n",
                "        axes[i//2, i%2].set_ylabel('Value')\n",
                "        axes[i//2, i%2].grid(True, alpha=0.3)\n",
                "        \n",
                "        # Add value labels on bars\n",
                "        for j, v in enumerate(values):\n",
                "            if metric in ['avg_loss_pct', 'avg_default_rate']:\n",
                "                axes[i//2, i%2].text(j, v, f'{v:.2%}', ha='center', va='bottom')\n",
                "            else:\n",
                "                axes[i//2, i%2].text(j, v, f'${v:,.0f}', ha='center', va='bottom')\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "# Example usage\n",
                "compare_tranches(0.3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Summary and Key Insights"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate summary insights\n",
                "print(\"📋 KEY INSIGHTS FROM CORRELATION ANALYSIS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Correlation sensitivity analysis\n",
                "mezz_data = summary_df[summary_df['tranche'] == 'Mezzanine']\n",
                "baseline_payoff = mezz_data[mezz_data['correlation'] == 0.01]['avg_payoff'].iloc[0]\n",
                "\n",
                "print(\"\\n🔍 Correlation Sensitivity (Mezzanine Tranche):\")\n",
                "for _, row in mezz_data.iterrows():\n",
                "    pct_change = (row['avg_payoff'] - baseline_payoff) / baseline_payoff * 100\n",
                "    print(f\"  Correlation {row['correlation']}: {pct_change:+.1f}% change from baseline\")\n",
                "\n",
                "# Risk ranking by correlation\n",
                "print(\"\\n⚠️ Risk Ranking by Correlation (Mezzanine):\")\n",
                "worst_1pct_ranking = mezz_data.sort_values('worst_1pct_avg_payoff')\n",
                "for i, (_, row) in enumerate(worst_1pct_ranking.iterrows()):\n",
                "    print(f\"  {i+1}. Correlation {row['correlation']}: ${row['worst_1pct_avg_payoff']:,.0f}\")\n",
                "\n",
                "# Tranche performance comparison\n",
                "print(\"\\n🏆 Best Performing Tranche by Metric:\")\n",
                "for corr in sorted(summary_df['correlation'].unique()):\n",
                "    corr_data = summary_df[summary_df['correlation'] == corr]\n",
                "    best_tranche = corr_data.loc[corr_data['avg_payoff'].idxmax(), 'tranche']\n",
                "    print(f\"  Correlation {corr}: {best_tranche} tranche has highest average payoff\")\n",
                "\n",
                "print(\"\\n✅ Analysis complete! Use the interactive functions above for deeper exploration.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to file
with open('correlation_analysis_explorer.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Jupyter notebook created successfully!")
print("📊 Notebook: correlation_analysis_explorer.ipynb")
print("🌐 Jupyter server should be running - check your browser for the notebook interface")
