# Multi-Family CRT (MBS) Security Simulation

A comprehensive Python-based simulation system for Multi-Family Credit Risk Transfer (CRT) securities, similar to Freddie Mac's K-Deal structure. This system simulates the performance of structured securities backed by multi-family mortgage loans with three tranches: subordinate, mezzanine, and senior.

## Features

### Core Simulation Capabilities
- **Monte Carlo Simulation**: Simulates loan performance using correlated random variables
- **Parallel Processing**: Multi-core CPU utilization for faster simulations
- **Three-Tranche Structure**: 
  - Subordinate tranche (Freddie Mac retention)
  - Mezzanine tranche (reinsurance layer)
  - Senior tranche (top layer)
- **Flexible Loan Pool**: Configurable number of loans (up to 100) with varying sizes (2M-500M)
- **Realistic Loan Assumptions**: 
  - Monthly prepayment rates (1% annually, much lower than residential)
  - Default rates (conditional on no prepayment)
  - Loss Given Default (LGD) rates
  - 10-year or 15-year terms with 30-year amortization

### Risk Modeling
- **Correlation Modeling**: Gaussian copula approach for loan correlation
- **Stress Testing**: Extreme scenario analysis with 5% and 1% VaR
- **Sensitivity Analysis**: Parameter impact assessment
- **Distribution Analysis**: Support for normal and long-tail distributions

### Visualization & Analysis
- **Tranche Payoff Distributions**: Histograms with VaR indicators
- **Monthly Cashflow Evolution**: Time series analysis with confidence intervals
- **Tranche Comparison**: Box plots and violin plots
- **Loss Distribution Analysis**: Cumulative loss and monthly loss heatmaps
- **Sensitivity Heatmaps**: Parameter impact visualization

## Requirements

- **Python 3.12+**: This project requires Python 3.12 or higher
- **Dependencies**: See `requirements.txt` for package versions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mbs-simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify Python 3.12 compatibility:
```bash
python test_python312_compatibility.py
```

## Quick Start

### Basic Usage

```python
from mbs_simulation import MBSSimulation, MBSVisualizer

# Create simulation
sim = MBSSimulation(
    n_loans=50,
    correlation=0.3,
    attachment_point=0.01,  # 1%
    detachment_point=0.05,  # 5%
    security_term_years=7
)

# Run Monte Carlo simulation with parallel processing
results = sim.run_monte_carlo(n_simulations=1000, use_parallel=True)

# Create visualizations
viz = MBSVisualizer(sim)
viz.create_summary_report()
viz.plot_tranche_payoff_distributions()
```

### Advanced Configuration

```python
# Custom loan sizes
loan_sizes = [25, 50, 100, 200, 300] * 10  # 50 loans with specific sizes

sim = MBSSimulation(
    n_loans=50,
    loan_sizes=loan_sizes,
    correlation=0.4,
    attachment_point=0.02,  # 2%
    detachment_point=0.08,  # 8%
    security_term_years=7,
    loan_term_years=15,
    amortization_years=30
)
```

## System Architecture

### Key Classes

#### `Loan`
Represents individual loans with:
- Principal and remaining balance
- Interest rates and payment schedules
- Prepayment, default, and LGD rates
- Term and amortization parameters

#### `Tranche`
Represents security tranches with:
- Attachment and detachment points
- Principal amounts
- Loss sharing calculations

#### `MBSSimulation`
Main simulation engine with:
- Monte Carlo simulation logic
- Correlation modeling using Gaussian copula
- Statistical analysis and stress testing
- Results aggregation

#### `MBSVisualizer`
Visualization and reporting with:
- Distribution plots
- Time series analysis
- Comparative analysis
- Summary reports

### Simulation Process

1. **Loan Pool Creation**: Generate or specify loan characteristics
2. **Tranche Structure**: Create three tranches based on attachment/detachment points
3. **Correlation Setup**: Generate correlated random variables for defaults
4. **Monthly Simulation**: Simulate each loan's monthly performance
5. **Loss Calculation**: Aggregate losses and allocate to tranches
6. **Statistical Analysis**: Calculate VaR, expected shortfall, and stress scenarios

## Parameter Configuration

### Loan Parameters
- **Number of Loans**: 1-100 (default: 50)
- **Loan Sizes**: 2M-500M (log-normal distribution by default)
- **Interest Rates**: 4.5% ± 0.5% (configurable)
- **Prepayment Rates**: 1% ± 0.5% annually (capped at 3%)
- **Default Rates**: 1% ± 0.5% annually (capped at 5%)
- **LGD Rates**: 40% ± 10% (capped at 20-70%)

### Tranche Parameters
- **Attachment Point**: Loss threshold for mezzanine tranche (default: 1%)
- **Detachment Point**: Loss threshold for senior tranche (default: 5%)
- **Security Term**: 7 years (configurable)
- **Loan Term**: 10 or 15 years
- **Amortization**: 30 years

### Correlation Parameters
- **Correlation Level**: 0.1-0.7 (default: 0.3)
- **Correlation Method**: Gaussian copula

## Example Scenarios

### Example 1: Basic Simulation
```python
# Run basic simulation with default parameters
sim, results, viz = example_1_basic_simulation()
```

### Example 2: Correlation Analysis
```python
# Compare different correlation levels
results = example_2_different_correlation_levels()
```

### Example 3: Tranche Level Analysis
```python
# Test different attachment/detachment points
results = example_3_different_tranche_levels()
```

### Example 4: Loan Size Impact
```python
# Analyze impact of loan size distributions
results = example_4_different_loan_sizes()
```

### Example 5: Stress Testing
```python
# Extreme scenario analysis
sim, results, viz = example_5_stress_testing()
```

### Example 6: Sensitivity Analysis
```python
# Parameter sensitivity assessment
results = example_6_sensitivity_analysis()
```

## Output Analysis

### Statistical Measures
- **Mean Payoff**: Expected tranche performance
- **Standard Deviation**: Risk measure
- **VaR (95% and 99%)**: Value at Risk
- **Expected Shortfall**: Conditional VaR
- **Stress Scenarios**: 5% and 1% worst-case outcomes

### Visualization Types
1. **Distribution Plots**: Histograms with VaR indicators
2. **Time Series**: Monthly cashflow evolution
3. **Comparative Analysis**: Tranche performance comparison
4. **Loss Analysis**: Cumulative and monthly loss patterns
5. **Sensitivity Maps**: Parameter impact heatmaps

## Risk Considerations

### Distribution Choice
- **Normal Distribution**: Good for central tendency
- **Long-tail Distributions**: Better for extreme events
- **Gaussian Copula**: Captures correlation structure

### Correlation Modeling
- **Low Correlation (0.1-0.3)**: Diversified portfolio
- **High Correlation (0.5-0.7)**: Concentrated risk
- **Stress Testing**: Extreme correlation scenarios

### Stress Scenarios
- **Default Rate Shocks**: 3x increase in defaults
- **LGD Shocks**: 1.5x increase in loss severity
- **Correlation Shocks**: Extreme correlation levels

## Performance Considerations

### Simulation Speed
- **Number of Simulations**: 500-2000 recommended
- **Number of Loans**: 30-100 optimal
- **Time Horizon**: 7 years (84 months)
- **Parallel Processing**: Automatically enabled for 100+ simulations
- **CPU Cores**: Utilizes all available cores for optimal performance

### Memory Usage
- **Large Simulations**: Monitor memory for 1000+ simulations
- **Loan Pool Size**: Consider memory for 100+ loans
- **Parallel Overhead**: Minimal memory overhead for parallel processing

## Extensions and Customization

### Adding New Distributions
```python
# Custom distribution for loan sizes
def custom_loan_sizes(n_loans):
    return np.random.weibull(shape=2, scale=100, size=n_loans)
```

### Custom Loan Characteristics
```python
# Modify loan assumptions
for loan in sim.loans:
    loan.annual_default_rate = custom_default_rate(loan.principal)
    loan.annual_prepay_rate = custom_prepay_rate(loan.remaining_balance)
```

### Additional Tranches
```python
# Add more tranches to the structure
def create_custom_tranches(total_principal):
    # Implementation for additional tranches
    pass
```

## Best Practices

### Parameter Validation
- Ensure attachment point < detachment point
- Validate loan sizes within reasonable bounds
- Check correlation levels (0-1 range)

### Simulation Quality
- Use sufficient number of simulations (1000+ for production)
- Validate random number generation
- Test correlation structure

### Risk Management
- Monitor VaR and expected shortfall
- Analyze stress scenarios
- Consider multiple correlation levels

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce number of simulations or loans
2. **Convergence Issues**: Increase simulation count
3. **Correlation Problems**: Check correlation matrix validity

### Performance Optimization
1. **Vectorization**: Use NumPy operations
2. **Parallel Processing**: Consider multiprocessing for large simulations
3. **Memory Management**: Monitor memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Freddie Mac's K-Deal structure
- Based on industry-standard MBS modeling practices
- Uses Gaussian copula for correlation modeling
