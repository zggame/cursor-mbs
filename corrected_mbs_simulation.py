#!/usr/bin/env python3
"""
Corrected MBS Simulation
========================

This version correctly models the cashflow distribution to tranches:
1. Monthly losses are allocated to tranches in order: Subordinate → Mezzanine → Senior
2. Each tranche's principal balance is reduced by its loss share
3. Payoffs = Principal (remaining balance) + Interest (remaining balance × rate × time)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
import multiprocessing as mp
from functools import partial
import pandas as pd
warnings.filterwarnings('ignore')

@dataclass
class Tranche:
    """Represents a tranche in the MBS structure"""
    name: str
    attachment_point: float  # Loss threshold (as percentage of total pool)
    detachment_point: float  # Loss threshold (as percentage of total pool)
    principal: float  # Initial principal amount
    remaining_principal: float  # Current remaining principal
    annual_rate: float  # Annual interest rate for this tranche
    monthly_rate: float  # Monthly interest rate
    
    def __post_init__(self):
        """Calculate monthly rate"""
        self.monthly_rate = self.annual_rate / 12
        self.remaining_principal = self.principal
    
    def absorb_loss(self, loss_amount: float) -> float:
        """Absorb loss and return remaining loss to pass to next tranche"""
        if self.remaining_principal <= 0:
            return loss_amount  # Already wiped out, pass all loss through
        
        if loss_amount <= self.remaining_principal:
            # This tranche can absorb all the loss
            self.remaining_principal -= loss_amount
            return 0.0  # No loss passed through
        else:
            # This tranche gets wiped out, pass remaining loss
            remaining_loss = loss_amount - self.remaining_principal
            self.remaining_principal = 0.0
            return remaining_loss
    
    def calculate_payoff(self, months_held: int) -> Dict[str, float]:
        """Calculate payoff as principal + interest"""
        principal_payoff = self.remaining_principal
        interest_payoff = self.remaining_principal * self.monthly_rate * months_held
        total_payoff = principal_payoff + interest_payoff
        
        return {
            'principal_payoff': principal_payoff,
            'interest_payoff': interest_payoff,
            'total_payoff': total_payoff,
            'months_held': months_held
        }

@dataclass
class Loan:
    """Represents a single loan in the MBS pool"""
    id: int
    principal: float  # Original loan amount
    remaining_balance: float  # Current remaining balance
    term_years: int  # Loan term (10 or 15 years)
    amortization_years: int  # Amortization period (30 years)
    annual_rate: float  # Annual interest rate
    monthly_rate: float  # Monthly interest rate
    
    # Annual rates (converted to monthly)
    annual_prepay_rate: float  # Annual prepayment rate
    annual_default_rate: float  # Annual default rate (conditional on no prepay)
    annual_lgd_rate: float  # Annual loss given default rate
    
    # Monthly rates
    monthly_prepay_rate: float
    monthly_default_rate: float
    monthly_lgd_rate: float
    
    def __post_init__(self):
        """Calculate monthly rates from annual rates"""
        self.monthly_rate = self.annual_rate / 12
        self.monthly_prepay_rate = self.annual_prepay_rate / 12
        self.monthly_default_rate = self.annual_default_rate / 12
        self.monthly_lgd_rate = self.annual_lgd_rate

class CorrectedMBSSimulation:
    """Corrected MBS simulation class with proper cashflow distribution"""
    
    def __init__(self, 
                 n_loans: int = 50,
                 loan_sizes: Optional[List[float]] = None,
                 correlation: float = 0.3,
                 attachment_point: float = 0.01,  # 1%
                 detachment_point: float = 0.05,  # 5%
                 security_term_years: int = 7,
                 loan_term_years: int = 10,
                 amortization_years: int = 30):
        
        self.n_loans = n_loans
        self.correlation = correlation
        self.attachment_point = attachment_point
        self.detachment_point = detachment_point
        self.security_term_years = security_term_years
        self.security_term_months = security_term_years * 12
        
        # Generate loan sizes if not provided
        if loan_sizes is None:
            # Generate random loan sizes between 2M and 500M
            self.loan_sizes = np.random.lognormal(mean=np.log(50), sigma=1.0, size=n_loans)
            self.loan_sizes = np.clip(self.loan_sizes, 2, 500)
        else:
            self.loan_sizes = loan_sizes[:n_loans]
        
        # Create loans
        self.loans = self._create_loans(loan_term_years, amortization_years)
        
        # Create tranches with proper interest rates
        self.tranches = self._create_tranches()
        
        # Simulation results
        self.simulation_results = None
        
    def _create_loans(self, term_years: int, amortization_years: int) -> List[Loan]:
        """Create loan objects with realistic assumptions"""
        loans = []
        
        for i, principal in enumerate(self.loan_sizes):
            # Realistic assumptions for multi-family loans
            annual_rate = np.random.normal(0.045, 0.005)  # 4.5% ± 0.5%
            
            # Prepayment rate: typically 0.5-1.5% annually for multi-family
            annual_prepay_rate = np.random.normal(0.01, 0.005)  # 1% ± 0.5%
            annual_prepay_rate = np.clip(annual_prepay_rate, 0.001, 0.03)  # Cap at 3%
            
            # Default rate: typically 0.5-2% annually
            annual_default_rate = np.random.normal(0.01, 0.005)  # 1% ± 0.5%
            annual_default_rate = np.clip(annual_default_rate, 0.001, 0.05)  # Cap at 5%
            
            # Loss given default: typically 30-50%
            annual_lgd_rate = np.random.normal(0.40, 0.10)  # 40% ± 10%
            annual_lgd_rate = np.clip(annual_lgd_rate, 0.20, 0.70)  # Cap between 20-70%
            
            loan = Loan(
                id=i,
                principal=principal,
                remaining_balance=principal,
                term_years=term_years,
                amortization_years=amortization_years,
                annual_rate=annual_rate,
                monthly_rate=annual_rate / 12,
                annual_prepay_rate=annual_prepay_rate,
                annual_default_rate=annual_default_rate,
                annual_lgd_rate=annual_lgd_rate,
                monthly_prepay_rate=annual_prepay_rate / 12,
                monthly_default_rate=annual_default_rate / 12,
                monthly_lgd_rate=annual_lgd_rate
            )
            loans.append(loan)
        
        return loans
    
    def _create_tranches(self) -> List[Tranche]:
        """Create the three tranches with proper interest rates"""
        total_principal = sum(loan.principal for loan in self.loans)
        
        # Calculate average loan rate
        avg_loan_rate = np.mean([loan.annual_rate for loan in self.loans])
        
        # Subordinate tranche (Freddie keeps this) - avg + 2%
        subordinate_principal = total_principal * self.attachment_point
        subordinate_rate = avg_loan_rate + 0.02  # avg + 2%
        subordinate = Tranche(
            name="Subordinate",
            attachment_point=0.0,
            detachment_point=self.attachment_point,
            principal=subordinate_principal,
            remaining_principal=subordinate_principal,
            annual_rate=subordinate_rate,
            monthly_rate=subordinate_rate / 12
        )
        
        # Mezzanine tranche (reinsurance layer) - avg + 0.5%
        mezzanine_principal = total_principal * (self.detachment_point - self.attachment_point)
        mezzanine_rate = avg_loan_rate + 0.005  # avg + 0.5%
        mezzanine = Tranche(
            name="Mezzanine",
            attachment_point=self.attachment_point,
            detachment_point=self.detachment_point,
            principal=mezzanine_principal,
            remaining_principal=mezzanine_principal,
            annual_rate=mezzanine_rate,
            monthly_rate=mezzanine_rate / 12
        )
        
        # Senior tranche (top layer) - avg - 1.5%
        senior_principal = total_principal * (1.0 - self.detachment_point)
        senior_rate = avg_loan_rate - 0.015  # avg - 1.5%
        senior = Tranche(
            name="Senior",
            attachment_point=self.detachment_point,
            detachment_point=1.0,
            principal=senior_principal,
            remaining_principal=senior_principal,
            annual_rate=senior_rate,
            monthly_rate=senior_rate / 12
        )
        
        return [subordinate, mezzanine, senior]
    
    def _simulate_loan_month(self, loan: Loan, random_state: np.random.RandomState) -> Dict:
        """Simulate one month for a single loan"""
        # Calculate monthly payment (P&I)
        monthly_payment = self._calculate_monthly_payment(loan)
        
        # Check for prepayment
        prepay_prob = random_state.random()
        if prepay_prob < loan.monthly_prepay_rate:
            return {
                'prepaid': True,
                'defaulted': False,
                'payment': loan.remaining_balance,
                'loss': 0.0,
                'remaining_balance': 0.0
            }
        
        # Check for default (conditional on no prepayment)
        default_prob = random_state.random()
        if default_prob < loan.monthly_default_rate:
            # Calculate loss given default
            loss = loan.remaining_balance * loan.monthly_lgd_rate
            return {
                'prepaid': False,
                'defaulted': True,
                'payment': 0.0,
                'loss': loss,
                'remaining_balance': 0.0
            }
        
        # Normal payment
        interest_payment = loan.remaining_balance * loan.monthly_rate
        principal_payment = monthly_payment - interest_payment
        
        # Ensure we don't overpay
        if principal_payment > loan.remaining_balance:
            principal_payment = loan.remaining_balance
        
        new_balance = loan.remaining_balance - principal_payment
        
        return {
            'prepaid': False,
            'defaulted': False,
            'payment': principal_payment + interest_payment,
            'loss': 0.0,
            'remaining_balance': new_balance
        }
    
    def _calculate_monthly_payment(self, loan: Loan) -> float:
        """Calculate monthly P&I payment"""
        r = loan.monthly_rate
        n = loan.amortization_years * 12
        p = loan.principal
        
        if r == 0:
            return p / n
        
        return p * (r * (1 + r)**n) / ((1 + r)**n - 1)
    
    def _simulate_correlated_defaults(self, n_simulations: int) -> np.ndarray:
        """Generate correlated default probabilities using Gaussian copula"""
        # Generate correlated random variables
        correlation_matrix = np.full((self.n_loans, self.n_loans), self.correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate correlated normal random variables
        correlated_normals = np.random.multivariate_normal(
            mean=np.zeros(self.n_loans),
            cov=correlation_matrix,
            size=n_simulations
        )
        
        # Convert to uniform random variables using normal CDF
        uniform_vars = stats.norm.cdf(correlated_normals)
        
        return uniform_vars
    
    def run_monte_carlo(self, n_simulations: int = 1000, seed: int = 42) -> Dict:
        """Run Monte Carlo simulation with correct cashflow distribution"""
        np.random.seed(seed)
        
        # Generate correlated random variables for all simulations
        correlated_uniforms = self._simulate_correlated_defaults(n_simulations)
        
        # Storage for results
        tranche_payoffs = {tranche.name: [] for tranche in self.tranches}
        tranche_principal_payoffs = {tranche.name: [] for tranche in self.tranches}
        tranche_interest_payoffs = {tranche.name: [] for tranche in self.tranches}
        monthly_losses = []
        tranche_balance_history = {tranche.name: [] for tranche in self.tranches}
        
        for sim in range(n_simulations):
            # Reset loans and tranches for this simulation
            for loan in self.loans:
                loan.remaining_balance = loan.principal
            
            for tranche in self.tranches:
                tranche.remaining_principal = tranche.principal
            
            # Track monthly losses and tranche balances
            sim_monthly_losses = []
            sim_tranche_balances = {tranche.name: [] for tranche in self.tranches}
            
            for month in range(self.security_term_months):
                month_loss = 0.0
                
                # Simulate each loan for this month
                for i, loan in enumerate(self.loans):
                    if loan.remaining_balance > 0:
                        # Use correlated random number for this loan/simulation
                        random_state = np.random.RandomState()
                        random_state.seed(int(correlated_uniforms[sim, i] * 1e9 + month))
                        
                        result = self._simulate_loan_month(loan, random_state)
                        month_loss += result['loss']
                        loan.remaining_balance = result['remaining_balance']
                
                # Allocate monthly loss to tranches in order
                remaining_loss = month_loss
                for tranche in self.tranches:
                    remaining_loss = tranche.absorb_loss(remaining_loss)
                    if remaining_loss <= 0:
                        break  # All loss absorbed
                
                sim_monthly_losses.append(month_loss)
                
                # Record tranche balances
                for tranche in self.tranches:
                    sim_tranche_balances[tranche.name].append(tranche.remaining_principal)
            
            # Calculate tranche payoffs
            for tranche in self.tranches:
                payoff_info = tranche.calculate_payoff(self.security_term_months)
                tranche_payoffs[tranche.name].append(payoff_info['total_payoff'])
                tranche_principal_payoffs[tranche.name].append(payoff_info['principal_payoff'])
                tranche_interest_payoffs[tranche.name].append(payoff_info['interest_payoff'])
            
            monthly_losses.append(sim_monthly_losses)
            
            # Store tranche balance history
            for tranche in self.tranches:
                tranche_balance_history[tranche.name].append(sim_tranche_balances[tranche.name])
        
        # Calculate statistics
        results = {
            'tranche_payoffs': tranche_payoffs,
            'tranche_principal_payoffs': tranche_principal_payoffs,
            'tranche_interest_payoffs': tranche_interest_payoffs,
            'monthly_losses': monthly_losses,
            'tranche_balance_history': tranche_balance_history,
            'statistics': self._calculate_statistics(tranche_payoffs),
            'tranche_info': {
                tranche.name: {
                    'principal': tranche.principal,
                    'annual_rate': tranche.annual_rate,
                    'monthly_rate': tranche.monthly_rate
                } for tranche in self.tranches
            }
        }
        
        self.simulation_results = results
        return results
    
    def _calculate_statistics(self, tranche_payoffs: Dict[str, List[float]]) -> Dict:
        """Calculate statistics for tranche payoffs"""
        statistics = {}
        
        for tranche_name, payoffs in tranche_payoffs.items():
            payoffs_array = np.array(payoffs)
            statistics[tranche_name] = {
                'mean': np.mean(payoffs_array),
                'std': np.std(payoffs_array),
                'min': np.min(payoffs_array),
                'max': np.max(payoffs_array),
                'median': np.median(payoffs_array),
                'var_95': np.percentile(payoffs_array, 5),
                'var_99': np.percentile(payoffs_array, 1),
                'wipeout_rate': np.mean(payoffs_array == 0) * 100
            }
        
        return statistics
    
    def print_tranche_info(self):
        """Print information about tranches"""
        print("🏦 TRANCHE INFORMATION")
        print("=" * 50)
        
        avg_loan_rate = np.mean([loan.annual_rate for loan in self.loans])
        print(f"Average loan rate: {avg_loan_rate:.3%}")
        print()
        
        for tranche in self.tranches:
            print(f"{tranche.name} Tranche:")
            print(f"  Principal: ${tranche.principal:,.0f}")
            print(f"  Annual rate: {tranche.annual_rate:.3%}")
            print(f"  Monthly rate: {tranche.monthly_rate:.4%}")
            print(f"  Rate spread vs avg: {tranche.annual_rate - avg_loan_rate:+.1%}")
            print()

# Example usage
if __name__ == "__main__":
    # Create simulation
    sim = CorrectedMBSSimulation(
        n_loans=100,
        correlation=0.3,
        attachment_point=0.01,  # 1%
        detachment_point=0.05,  # 5%
        security_term_years=7
    )
    
    # Print tranche information
    sim.print_tranche_info()
    
    # Run simulation
    print("Running simulation...")
    results = sim.run_monte_carlo(n_simulations=500, seed=42)
    
    # Print results
    print("\n📊 SIMULATION RESULTS")
    print("=" * 50)
    
    for tranche_name, stats in results['statistics'].items():
        print(f"\n{tranche_name} Tranche:")
        print(f"  Mean payoff: ${stats['mean']:,.0f}")
        print(f"  Std dev: ${stats['std']:,.0f}")
        print(f"  Min payoff: ${stats['min']:,.0f}")
        print(f"  Max payoff: ${stats['max']:,.0f}")
        print(f"  Wipeout rate: {stats['wipeout_rate']:.1f}%")
        print(f"  95% VaR: ${stats['var_95']:,.0f}")
        print(f"  99% VaR: ${stats['var_99']:,.0f}")
