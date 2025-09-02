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
class DetailedLossRecord:
    """Detailed loss record for each simulation path and tranche"""
    simulation_id: int
    tranche_name: str
    total_loss: float
    loss_percentage: float
    tranche_principal: float
    tranche_loss: float
    tranche_payoff: float
    monthly_losses: List[float]
    cumulative_losses: List[float]
    worst_month_loss: float
    worst_month: int
    defaulted_loans: int
    total_loans: int
    default_rate: float

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

@dataclass
class Tranche:
    """Represents a tranche in the MBS structure"""
    name: str
    attachment_point: float  # Loss threshold (as percentage of total pool)
    detachment_point: float  # Loss threshold (as percentage of total pool)
    principal: float  # Principal amount
    
    def get_loss_share(self, total_loss: float, total_principal: float) -> float:
        """Calculate loss share for this tranche"""
        loss_percentage = total_loss / total_principal
        if loss_percentage <= self.attachment_point:
            return 0.0
        elif loss_percentage >= self.detachment_point:
            return self.principal
        else:
            return self.principal * (loss_percentage - self.attachment_point) / (self.detachment_point - self.attachment_point)

class MBSSimulation:
    """Main MBS simulation class"""
    
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
        
        # Create tranches
        self.tranches = self._create_tranches()
        
        # Simulation results
        self.simulation_results = None
        
    def _create_loans(self, term_years: int, amortization_years: int) -> List[Loan]:
        """Create loan objects with realistic assumptions"""
        loans = []
        total_principal = sum(self.loan_sizes)
        
        for i, principal in enumerate(self.loan_sizes):
            # Realistic assumptions for multi-family loans
            annual_rate = np.random.normal(0.045, 0.005)  # 4.5% ± 0.5%
            
            # Prepayment rate: typically 0.5-1.5% annually for multi-family (much lower than residential)
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
        """Create the three tranches: subordinate, mezzanine, and senior"""
        total_principal = sum(loan.principal for loan in self.loans)
        
        # Subordinate tranche (Freddie keeps this)
        subordinate_principal = total_principal * self.attachment_point
        subordinate = Tranche(
            name="Subordinate",
            attachment_point=0.0,
            detachment_point=self.attachment_point,
            principal=subordinate_principal
        )
        
        # Mezzanine tranche (reinsurance layer)
        mezzanine_principal = total_principal * (self.detachment_point - self.attachment_point)
        mezzanine = Tranche(
            name="Mezzanine",
            attachment_point=self.attachment_point,
            detachment_point=self.detachment_point,
            principal=mezzanine_principal
        )
        
        # Senior tranche (top layer)
        senior_principal = total_principal * (1.0 - self.detachment_point)
        senior = Tranche(
            name="Senior",
            attachment_point=self.detachment_point,
            detachment_point=1.0,
            principal=senior_principal
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
    
    def run_monte_carlo(self, n_simulations: int = 1000, seed: int = 42, use_parallel: bool = True, n_jobs: Optional[int] = None, detailed_tracking: bool = False) -> Dict:
        """Run Monte Carlo simulation with optional parallel processing and detailed loss tracking"""
        if detailed_tracking:
            return self._run_monte_carlo_with_detailed_tracking(n_simulations, seed, use_parallel, n_jobs)
        elif use_parallel and n_simulations > 100:  # Only use parallel for larger simulations
            return self._run_monte_carlo_parallel(n_simulations, seed, n_jobs)
        else:
            return self._run_monte_carlo_sequential(n_simulations, seed)
    
    def _run_monte_carlo_sequential(self, n_simulations: int, seed: int) -> Dict:
        """Run Monte Carlo simulation sequentially (original method)"""
        np.random.seed(seed)
        
        # Generate correlated random variables for all simulations
        correlated_uniforms = self._simulate_correlated_defaults(n_simulations)
        
        # Storage for results
        tranche_payoffs = {tranche.name: [] for tranche in self.tranches}
        monthly_cashflows = []
        monthly_losses = []
        
        total_principal = sum(loan.principal for loan in self.loans)
        
        for sim in range(n_simulations):
            # Reset loans for this simulation
            for loan in self.loans:
                loan.remaining_balance = loan.principal
            
            # Track monthly results for this simulation
            sim_monthly_cashflow = []
            sim_monthly_loss = []
            
            for month in range(self.security_term_months):
                month_cashflow = 0.0
                month_loss = 0.0
                
                # Simulate each loan for this month
                for i, loan in enumerate(self.loans):
                    if loan.remaining_balance > 0:
                        # Use correlated random number for this loan/simulation
                        random_state = np.random.RandomState()
                        random_state.seed(int(correlated_uniforms[sim, i] * 1e9 + month))
                        
                        result = self._simulate_loan_month(loan, random_state)
                        
                        month_cashflow += result['payment']
                        month_loss += result['loss']
                        loan.remaining_balance = result['remaining_balance']
                
                sim_monthly_cashflow.append(month_cashflow)
                sim_monthly_loss.append(month_loss)
            
            # Calculate cumulative losses and tranche payoffs
            cumulative_loss = sum(sim_monthly_loss)
            
            for tranche in self.tranches:
                tranche_loss = tranche.get_loss_share(cumulative_loss, total_principal)
                tranche_payoff = tranche.principal - tranche_loss
                tranche_payoffs[tranche.name].append(tranche_payoff)
            
            monthly_cashflows.append(sim_monthly_cashflow)
            monthly_losses.append(sim_monthly_loss)
        
        # Calculate statistics
        results = {
            'tranche_payoffs': tranche_payoffs,
            'monthly_cashflows': monthly_cashflows,
            'monthly_losses': monthly_losses,
            'statistics': self._calculate_statistics(tranche_payoffs),
            'stress_scenarios': self._calculate_stress_scenarios(tranche_payoffs)
        }
        
        self.simulation_results = results
        return results
    
    def _run_monte_carlo_with_detailed_tracking(self, n_simulations: int, seed: int, use_parallel: bool, n_jobs: Optional[int] = None) -> Dict:
        """Run Monte Carlo simulation with detailed loss tracking for tail risk analysis"""
        np.random.seed(seed)
        
        # Generate correlated random variables for all simulations
        correlated_uniforms = self._simulate_correlated_defaults(n_simulations)
        
        # Storage for detailed results
        detailed_loss_records = []
        tranche_payoffs = {tranche.name: [] for tranche in self.tranches}
        monthly_cashflows = []
        monthly_losses = []
        
        total_principal = sum(loan.principal for loan in self.loans)
        
        for sim in range(n_simulations):
            # Reset loans for this simulation
            for loan in self.loans:
                loan.remaining_balance = loan.principal
            
            # Track detailed results for this simulation
            sim_monthly_cashflow = []
            sim_monthly_loss = []
            sim_cumulative_losses = []
            defaulted_loans = 0
            
            for month in range(self.security_term_months):
                month_cashflow = 0.0
                month_loss = 0.0
                
                # Simulate each loan for this month
                for i, loan in enumerate(self.loans):
                    if loan.remaining_balance > 0:
                        # Use correlated random number for this loan/simulation
                        random_state = np.random.RandomState()
                        random_state.seed(int(correlated_uniforms[sim, i] * 1e9 + month))
                        
                        result = self._simulate_loan_month(loan, random_state)
                        
                        month_cashflow += result['payment']
                        month_loss += result['loss']
                        loan.remaining_balance = result['remaining_balance']
                        
                        # Track defaults
                        if result['defaulted']:
                            defaulted_loans += 1
                
                sim_monthly_cashflow.append(month_cashflow)
                sim_monthly_loss.append(month_loss)
                sim_cumulative_losses.append(sum(sim_monthly_loss))
            
            # Calculate cumulative losses and tranche payoffs
            cumulative_loss = sum(sim_monthly_loss)
            loss_percentage = cumulative_loss / total_principal
            
            # Find worst month
            worst_month = np.argmax(sim_monthly_loss)
            worst_month_loss = sim_monthly_loss[worst_month]
            
            # Calculate default rate
            default_rate = defaulted_loans / self.n_loans
            
            # Create detailed loss records for each tranche
            for tranche in self.tranches:
                tranche_loss = tranche.get_loss_share(cumulative_loss, total_principal)
                tranche_payoff = tranche.principal - tranche_loss
                
                # Create detailed loss record
                loss_record = DetailedLossRecord(
                    simulation_id=sim,
                    tranche_name=tranche.name,
                    total_loss=cumulative_loss,
                    loss_percentage=loss_percentage,
                    tranche_principal=tranche.principal,
                    tranche_loss=tranche_loss,
                    tranche_payoff=tranche_payoff,
                    monthly_losses=sim_monthly_loss.copy(),
                    cumulative_losses=sim_cumulative_losses.copy(),
                    worst_month_loss=worst_month_loss,
                    worst_month=worst_month,
                    defaulted_loans=defaulted_loans,
                    total_loans=self.n_loans,
                    default_rate=default_rate
                )
                
                detailed_loss_records.append(loss_record)
                tranche_payoffs[tranche.name].append(tranche_payoff)
            
            monthly_cashflows.append(sim_monthly_cashflow)
            monthly_losses.append(sim_monthly_loss)
        
        # Calculate statistics including tail risk metrics
        results = {
            'tranche_payoffs': tranche_payoffs,
            'monthly_cashflows': monthly_cashflows,
            'monthly_losses': monthly_losses,
            'detailed_loss_records': detailed_loss_records,
            'statistics': self._calculate_statistics(tranche_payoffs),
            'stress_scenarios': self._calculate_stress_scenarios(tranche_payoffs),
            'tail_risk_analysis': self._calculate_tail_risk_metrics(detailed_loss_records)
        }
        
        self.simulation_results = results
        return results
    
    def _run_monte_carlo_parallel(self, n_simulations: int, seed: int, n_jobs: Optional[int] = None) -> Dict:
        """Run Monte Carlo simulation using parallel processing"""
        if n_jobs is None:
            n_jobs = max(1, min(mp.cpu_count()-2, n_simulations // 100))  # Use even fewer cores
        
        # Generate correlated random variables for all simulations
        np.random.seed(seed)
        correlated_uniforms = self._simulate_correlated_defaults(n_simulations)
        
        # Split simulations across processes
        chunk_size = max(1, n_simulations // n_jobs)
        print(f"Chunk size: {chunk_size}, n_jobs: {n_jobs}")

        simulation_chunks = []
        
        for i in range(0, n_simulations, chunk_size):
            end_idx = min(i + chunk_size, n_simulations)
            simulation_chunks.append((i, end_idx, correlated_uniforms[i:end_idx]))
        
        # Prepare simulation parameters for parallel processing
        sim_params = {
            'n_loans': self.n_loans,
            'loan_sizes': [loan.principal for loan in self.loans],
            'correlation': self.correlation,
            'attachment_point': self.attachment_point,
            'detachment_point': self.detachment_point,
            'security_term_months': self.security_term_months,
            'loan_term_years': self.loans[0].term_years,
            'amortization_years': self.loans[0].amortization_years
        }
        
        # Run parallel simulations
        with mp.Pool(processes=n_jobs) as pool:
            chunk_results = pool.map(
                partial(self._run_simulation_chunk, sim_params=sim_params),
                simulation_chunks
            )
        
        # Combine results from all chunks
        tranche_payoffs = {tranche.name: [] for tranche in self.tranches}
        monthly_cashflows = []
        monthly_losses = []
        
        for chunk_result in chunk_results:
            for tranche_name, payoffs in chunk_result['tranche_payoffs'].items():
                tranche_payoffs[tranche_name].extend(payoffs)
            monthly_cashflows.extend(chunk_result['monthly_cashflows'])
            monthly_losses.extend(chunk_result['monthly_losses'])
        
        # Calculate statistics
        results = {
            'tranche_payoffs': tranche_payoffs,
            'monthly_cashflows': monthly_cashflows,
            'monthly_losses': monthly_losses,
            'statistics': self._calculate_statistics(tranche_payoffs),
            'stress_scenarios': self._calculate_stress_scenarios(tranche_payoffs)
        }
        
        self.simulation_results = results
        return results
    
    def _run_simulation_chunk(self, chunk_info: Tuple, sim_params: Dict) -> Dict:
        """Run a chunk of simulations (for parallel processing)"""
        start_idx, end_idx, chunk_uniforms = chunk_info
        
        # Create a temporary simulation object for this chunk
        temp_sim = MBSSimulation(
            n_loans=sim_params['n_loans'],
            loan_sizes=sim_params['loan_sizes'],
            correlation=sim_params['correlation'],
            attachment_point=sim_params['attachment_point'],
            detachment_point=sim_params['detachment_point'],
            security_term_years=sim_params['security_term_months'] // 12,
            loan_term_years=sim_params['loan_term_years'],
            amortization_years=sim_params['amortization_years']
        )
        
        # Storage for results
        tranche_payoffs = {tranche.name: [] for tranche in temp_sim.tranches}
        monthly_cashflows = []
        monthly_losses = []
        
        total_principal = sum(loan.principal for loan in temp_sim.loans)
        
        for sim_idx, sim_uniforms in enumerate(chunk_uniforms):
            # Reset loans for this simulation
            for loan in temp_sim.loans:
                loan.remaining_balance = loan.principal
            
            # Track monthly results for this simulation
            sim_monthly_cashflow = []
            sim_monthly_loss = []
            
            for month in range(sim_params['security_term_months']):
                month_cashflow = 0.0
                month_loss = 0.0
                
                # Simulate each loan for this month
                for i, loan in enumerate(temp_sim.loans):
                    if loan.remaining_balance > 0:
                        # Use correlated random number for this loan/simulation
                        random_state = np.random.RandomState()
                        random_state.seed(int(sim_uniforms[i] * 1e9 + month))
                        
                        result = temp_sim._simulate_loan_month(loan, random_state)
                        
                        month_cashflow += result['payment']
                        month_loss += result['loss']
                        loan.remaining_balance = result['remaining_balance']
                
                sim_monthly_cashflow.append(month_cashflow)
                sim_monthly_loss.append(month_loss)
            
            # Calculate cumulative losses and tranche payoffs
            cumulative_loss = sum(sim_monthly_loss)
            
            for tranche in temp_sim.tranches:
                tranche_loss = tranche.get_loss_share(cumulative_loss, total_principal)
                tranche_payoff = tranche.principal - tranche_loss
                tranche_payoffs[tranche.name].append(tranche_payoff)
            
            monthly_cashflows.append(sim_monthly_cashflow)
            monthly_losses.append(sim_monthly_loss)
        
        return {
            'tranche_payoffs': tranche_payoffs,
            'monthly_cashflows': monthly_cashflows,
            'monthly_losses': monthly_losses
        }
    
    def _calculate_statistics(self, tranche_payoffs: Dict) -> Dict:
        """Calculate summary statistics for each tranche"""
        stats_dict = {}
        
        for tranche_name, payoffs in tranche_payoffs.items():
            payoffs_array = np.array(payoffs)
            stats_dict[tranche_name] = {
                'mean': np.mean(payoffs_array),
                'std': np.std(payoffs_array),
                'min': np.min(payoffs_array),
                'max': np.max(payoffs_array),
                'median': np.median(payoffs_array),
                'var_95': np.percentile(payoffs_array, 5),  # 95% VaR
                'var_99': np.percentile(payoffs_array, 1),  # 99% VaR
                'expected_shortfall_95': np.mean(payoffs_array[payoffs_array <= np.percentile(payoffs_array, 5)]),
                'expected_shortfall_99': np.mean(payoffs_array[payoffs_array <= np.percentile(payoffs_array, 1)])
            }
        
        return stats_dict
    
    def _calculate_stress_scenarios(self, tranche_payoffs: Dict) -> Dict:
        """Calculate stress scenario results"""
        stress_dict = {}
        
        for tranche_name, payoffs in tranche_payoffs.items():
            payoffs_array = np.array(payoffs)
            stress_dict[tranche_name] = {
                'stress_5pct': np.percentile(payoffs_array, 5),
                'stress_1pct': np.percentile(payoffs_array, 1),
                'worst_case': np.min(payoffs_array)
            }
        
        return stress_dict
    
    def _calculate_tail_risk_metrics(self, detailed_loss_records: List[DetailedLossRecord]) -> Dict:
        """Calculate comprehensive tail risk metrics"""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'simulation_id': record.simulation_id,
                'tranche_name': record.tranche_name,
                'total_loss': record.total_loss,
                'loss_percentage': record.loss_percentage,
                'tranche_loss': record.tranche_loss,
                'tranche_payoff': record.tranche_payoff,
                'worst_month_loss': record.worst_month_loss,
                'worst_month': record.worst_month,
                'defaulted_loans': record.defaulted_loans,
                'default_rate': record.default_rate
            }
            for record in detailed_loss_records
        ])
        
        tail_risk_metrics = {}
        
        for tranche_name in df['tranche_name'].unique():
            tranche_data = df[df['tranche_name'] == tranche_name]
            
            # Sort by tranche payoff (worst to best)
            tranche_data_sorted = tranche_data.sort_values('tranche_payoff')
            
            # Calculate percentiles for worst cases
            n_simulations = len(tranche_data)
            worst_5pct_idx = int(0.05 * n_simulations)
            worst_1pct_idx = int(0.01 * n_simulations)
            
            # Worst 5% scenarios
            worst_5pct = tranche_data_sorted.iloc[:worst_5pct_idx]
            worst_1pct = tranche_data_sorted.iloc[:worst_1pct_idx]
            
            tail_risk_metrics[tranche_name] = {
                # Worst 5% metrics
                'worst_5pct_count': len(worst_5pct),
                'worst_5pct_avg_payoff': worst_5pct['tranche_payoff'].mean(),
                'worst_5pct_avg_loss': worst_5pct['tranche_loss'].mean(),
                'worst_5pct_avg_total_loss': worst_5pct['total_loss'].mean(),
                'worst_5pct_avg_loss_pct': worst_5pct['loss_percentage'].mean(),
                'worst_5pct_avg_default_rate': worst_5pct['default_rate'].mean(),
                'worst_5pct_avg_worst_month_loss': worst_5pct['worst_month_loss'].mean(),
                
                # Worst 1% metrics
                'worst_1pct_count': len(worst_1pct),
                'worst_1pct_avg_payoff': worst_1pct['tranche_payoff'].mean(),
                'worst_1pct_avg_loss': worst_1pct['tranche_loss'].mean(),
                'worst_1pct_avg_total_loss': worst_1pct['total_loss'].mean(),
                'worst_1pct_avg_loss_pct': worst_1pct['loss_percentage'].mean(),
                'worst_1pct_avg_default_rate': worst_1pct['default_rate'].mean(),
                'worst_1pct_avg_worst_month_loss': worst_1pct['worst_month_loss'].mean(),
                
                # Overall statistics
                'total_simulations': n_simulations,
                'avg_payoff': tranche_data['tranche_payoff'].mean(),
                'avg_loss': tranche_data['tranche_loss'].mean(),
                'avg_total_loss': tranche_data['total_loss'].mean(),
                'avg_loss_pct': tranche_data['loss_percentage'].mean(),
                'avg_default_rate': tranche_data['default_rate'].mean(),
                
                # Risk metrics
                'var_95': np.percentile(tranche_data['tranche_payoff'], 5),
                'var_99': np.percentile(tranche_data['tranche_payoff'], 1),
                'expected_shortfall_95': tranche_data[tranche_data['tranche_payoff'] <= np.percentile(tranche_data['tranche_payoff'], 5)]['tranche_payoff'].mean(),
                'expected_shortfall_99': tranche_data[tranche_data['tranche_payoff'] <= np.percentile(tranche_data['tranche_payoff'], 1)]['tranche_payoff'].mean(),
                
                # Worst case details
                'worst_case_payoff': tranche_data['tranche_payoff'].min(),
                'worst_case_loss': tranche_data['tranche_loss'].max(),
                'worst_case_total_loss': tranche_data['total_loss'].max(),
                'worst_case_loss_pct': tranche_data['loss_percentage'].max(),
                'worst_case_default_rate': tranche_data['default_rate'].max(),
                'worst_case_worst_month_loss': tranche_data['worst_month_loss'].max()
            }
        
        return tail_risk_metrics
    
    def get_worst_case_scenarios(self, percentile: float = 5.0) -> Dict:
        """Get detailed information about worst case scenarios"""
        if not hasattr(self, 'simulation_results') or 'detailed_loss_records' not in self.simulation_results:
            raise ValueError("Detailed loss tracking not enabled. Run simulation with detailed_tracking=True")
        
        detailed_records = self.simulation_results['detailed_loss_records']
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'simulation_id': record.simulation_id,
                'tranche_name': record.tranche_name,
                'total_loss': record.total_loss,
                'loss_percentage': record.loss_percentage,
                'tranche_loss': record.tranche_loss,
                'tranche_payoff': record.tranche_payoff,
                'worst_month_loss': record.worst_month_loss,
                'worst_month': record.worst_month,
                'defaulted_loans': record.defaulted_loans,
                'default_rate': record.default_rate
            }
            for record in detailed_records
        ])
        
        worst_scenarios = {}
        
        for tranche_name in df['tranche_name'].unique():
            tranche_data = df[df['tranche_name'] == tranche_name]
            
            # Sort by tranche payoff (worst to best)
            tranche_data_sorted = tranche_data.sort_values('tranche_payoff')
            
            # Calculate threshold for worst cases
            n_simulations = len(tranche_data)
            worst_count = int((percentile / 100.0) * n_simulations)
            
            # Get worst scenarios
            worst_scenarios_data = tranche_data_sorted.iloc[:worst_count]
            
            worst_scenarios[tranche_name] = {
                'scenarios': worst_scenarios_data.to_dict('records'),
                'summary': {
                    'count': len(worst_scenarios_data),
                    'percentile': percentile,
                    'avg_payoff': worst_scenarios_data['tranche_payoff'].mean(),
                    'avg_loss': worst_scenarios_data['tranche_loss'].mean(),
                    'avg_total_loss': worst_scenarios_data['total_loss'].mean(),
                    'avg_loss_pct': worst_scenarios_data['loss_percentage'].mean(),
                    'avg_default_rate': worst_scenarios_data['default_rate'].mean(),
                    'min_payoff': worst_scenarios_data['tranche_payoff'].min(),
                    'max_loss': worst_scenarios_data['tranche_loss'].max()
                }
            }
        
        return worst_scenarios
    
    def export_detailed_loss_data(self, filename: str = "detailed_loss_data.csv") -> str:
        """Export detailed loss data to CSV for further analysis"""
        if not hasattr(self, 'simulation_results') or 'detailed_loss_records' not in self.simulation_results:
            raise ValueError("Detailed loss tracking not enabled. Run simulation with detailed_tracking=True")
        
        detailed_records = self.simulation_results['detailed_loss_records']
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'simulation_id': record.simulation_id,
                'tranche_name': record.tranche_name,
                'total_loss': record.total_loss,
                'loss_percentage': record.loss_percentage,
                'tranche_principal': record.tranche_principal,
                'tranche_loss': record.tranche_loss,
                'tranche_payoff': record.tranche_payoff,
                'worst_month_loss': record.worst_month_loss,
                'worst_month': record.worst_month,
                'defaulted_loans': record.defaulted_loans,
                'total_loans': record.total_loans,
                'default_rate': record.default_rate
            }
            for record in detailed_records
        ])
        
        # Export to CSV
        df.to_csv(filename, index=False)
        return filename

    def dump_loan_information(self) -> Dict:
        """Dump detailed loan information including all assumptions"""
        loan_info = {
            'summary': {
                'total_loans': self.n_loans,
                'total_principal': sum(loan.principal for loan in self.loans),
                'average_loan_size': np.mean([loan.principal for loan in self.loans]),
                'min_loan_size': min(loan.principal for loan in self.loans),
                'max_loan_size': max(loan.principal for loan in self.loans),
                'correlation': self.correlation,
                'attachment_point': self.attachment_point,
                'detachment_point': self.detachment_point,
                'security_term_years': self.security_term_years,
                'security_term_months': self.security_term_months
            },
            'loan_details': [],
            'tranche_details': []
        }
        
        # Individual loan details
        for loan in self.loans:
            loan_detail = {
                'loan_id': loan.id,
                'principal': loan.principal,
                'remaining_balance': loan.remaining_balance,
                'term_years': loan.term_years,
                'amortization_years': loan.amortization_years,
                'annual_rate': loan.annual_rate,
                'monthly_rate': loan.monthly_rate,
                'annual_prepay_rate': loan.annual_prepay_rate,
                'monthly_prepay_rate': loan.monthly_prepay_rate,
                'annual_default_rate': loan.annual_default_rate,
                'monthly_default_rate': loan.monthly_default_rate,
                'annual_lgd_rate': loan.annual_lgd_rate,
                'monthly_lgd_rate': loan.monthly_lgd_rate
            }
            loan_info['loan_details'].append(loan_detail)
        
        # Tranche details
        for tranche in self.tranches:
            tranche_detail = {
                'name': tranche.name,
                'attachment_point': tranche.attachment_point,
                'detachment_point': tranche.detachment_point,
                'principal': tranche.principal,
                'attachment_point_pct': f"{tranche.attachment_point * 100:.1f}%",
                'detachment_point_pct': f"{tranche.detachment_point * 100:.1f}%"
            }
            loan_info['tranche_details'].append(tranche_detail)
        
        return loan_info
    
    def print_loan_summary(self):
        """Print a formatted summary of loan information"""
        info = self.dump_loan_information()
        
        print("=== MBS LOAN INFORMATION DUMP ===")
        print(f"Total Loans: {info['summary']['total_loans']}")
        print(f"Total Principal: ${info['summary']['total_principal']:,.2f}")
        print(f"Average Loan Size: ${info['summary']['average_loan_size']:,.2f}")
        print(f"Loan Size Range: ${info['summary']['min_loan_size']:,.2f} - ${info['summary']['max_loan_size']:,.2f}")
        print(f"Correlation: {info['summary']['correlation']:.3f}")
        print(f"Security Term: {info['summary']['security_term_years']} years ({info['summary']['security_term_months']} months)")
        print()
        
        print("=== TRANCHE STRUCTURE ===")
        for tranche in info['tranche_details']:
            print(f"{tranche['name']}: {tranche['attachment_point_pct']} - {tranche['detachment_point_pct']} (${tranche['principal']:,.2f})")
        print()
        
        print("=== LOAN ASSUMPTIONS SUMMARY ===")
        rates = [loan['annual_rate'] for loan in info['loan_details']]
        prepay_rates = [loan['annual_prepay_rate'] for loan in info['loan_details']]
        default_rates = [loan['annual_default_rate'] for loan in info['loan_details']]
        lgd_rates = [loan['annual_lgd_rate'] for loan in info['loan_details']]
        
        print(f"Interest Rates: {np.mean(rates):.3f} avg ({np.min(rates):.3f} - {np.max(rates):.3f})")
        print(f"Prepayment Rates: {np.mean(prepay_rates):.3f} avg ({np.min(prepay_rates):.3f} - {np.max(prepay_rates):.3f})")
        print(f"Default Rates: {np.mean(default_rates):.3f} avg ({np.min(default_rates):.3f} - {np.max(default_rates):.3f})")
        print(f"Loss Given Default: {np.mean(lgd_rates):.3f} avg ({np.min(lgd_rates):.3f} - {np.max(lgd_rates):.3f})")
        print()
        
        print("=== INDIVIDUAL LOAN DETAILS ===")
        print("ID | Principal | Rate | Prepay | Default | LGD")
        print("---|-----------|------|--------|---------|-----")
        for loan in info['loan_details'][:10]:  # Show first 10 loans
            print(f"{loan['loan_id']:2d} | ${loan['principal']:8.1f}M | {loan['annual_rate']:.3f} | {loan['annual_prepay_rate']:.3f} | {loan['annual_default_rate']:.3f} | {loan['annual_lgd_rate']:.3f}")
        if len(info['loan_details']) > 10:
            print(f"... and {len(info['loan_details']) - 10} more loans")
        print()

class MBSVisualizer:
    """Visualization class for MBS simulation results"""
    
    def __init__(self, simulation: MBSSimulation):
        self.simulation = simulation
        self.results = simulation.simulation_results
        
    def plot_tranche_payoff_distributions(self, figsize=(15, 10)):
        """Plot payoff distributions for all tranches"""
        # Clear any existing figures
        plt.close('all')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        tranche_names = list(self.results['tranche_payoffs'].keys())
        
        for i, tranche_name in enumerate(tranche_names):
            payoffs = self.results['tranche_payoffs'][tranche_name]
            
            # Histogram
            axes[i].hist(payoffs, bins=50, alpha=0.7, density=True, color=f'C{i}')
            axes[i].axvline(np.mean(payoffs), color='red', linestyle='--', label='Mean')
            axes[i].axvline(np.percentile(payoffs, 5), color='orange', linestyle='--', label='5% VaR')
            axes[i].axvline(np.percentile(payoffs, 1), color='darkred', linestyle='--', label='1% VaR')
            
            axes[i].set_title(f'{tranche_name} Tranche Payoff Distribution')
            axes[i].set_xlabel('Payoff Amount')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_cashflow_evolution(self, n_simulations_to_show=50, figsize=(15, 8)):
        """Plot monthly cashflow evolution over time"""
        # Clear any existing figures
        plt.close('all')
        
        monthly_cashflows = np.array(self.results['monthly_cashflows'])
        
        # Calculate percentiles
        p5 = np.percentile(monthly_cashflows, 5, axis=0)
        p25 = np.percentile(monthly_cashflows, 25, axis=0)
        p50 = np.percentile(monthly_cashflows, 50, axis=0)
        p75 = np.percentile(monthly_cashflows, 75, axis=0)
        p95 = np.percentile(monthly_cashflows, 95, axis=0)
        
        months = range(1, len(p50) + 1)
        
        plt.figure(figsize=figsize)
        
        # Plot individual simulations
        for i in range(min(n_simulations_to_show, len(monthly_cashflows))):
            plt.plot(months, monthly_cashflows[i], alpha=0.1, color='gray')
        
        # Plot percentiles
        plt.fill_between(months, p5, p95, alpha=0.3, color='blue', label='5th-95th percentile')
        plt.fill_between(months, p25, p75, alpha=0.5, color='blue', label='25th-75th percentile')
        plt.plot(months, p50, color='red', linewidth=2, label='Median')
        
        plt.title('Monthly Cashflow Evolution')
        plt.xlabel('Month')
        plt.ylabel('Cashflow Amount')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_tranche_comparison(self, figsize=(12, 8)):
        """Compare tranche payoffs side by side"""
        # Clear any existing figures
        plt.close('all')
        
        tranche_names = list(self.results['tranche_payoffs'].keys())
        payoffs = [self.results['tranche_payoffs'][name] for name in tranche_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        ax1.boxplot(payoffs, labels=tranche_names)
        ax1.set_title('Tranche Payoff Comparison (Box Plot)')
        ax1.set_ylabel('Payoff Amount')
        ax1.grid(True, alpha=0.3)
        
        # Violin plot
        ax2.violinplot(payoffs, positions=range(len(tranche_names)))
        ax2.set_xticks(range(len(tranche_names)))
        ax2.set_xticklabels(tranche_names)
        ax2.set_title('Tranche Payoff Comparison (Violin Plot)')
        ax2.set_ylabel('Payoff Amount')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_loss_distribution(self, figsize=(12, 8)):
        """Plot loss distribution analysis"""
        # Clear any existing figures
        plt.close('all')
        
        monthly_losses = np.array(self.results['monthly_losses'])
        cumulative_losses = np.sum(monthly_losses, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Cumulative loss distribution
        ax1.hist(cumulative_losses, bins=50, alpha=0.7, density=True)
        ax1.axvline(np.mean(cumulative_losses), color='red', linestyle='--', label='Mean')
        ax1.axvline(np.percentile(cumulative_losses, 95), color='orange', linestyle='--', label='95% VaR')
        ax1.axvline(np.percentile(cumulative_losses, 99), color='darkred', linestyle='--', label='99% VaR')
        ax1.set_title('Cumulative Loss Distribution')
        ax1.set_xlabel('Cumulative Loss')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Monthly loss heatmap
        monthly_loss_percentiles = np.percentile(monthly_losses, [5, 25, 50, 75, 95], axis=0)
        im = ax2.imshow(monthly_loss_percentiles, aspect='auto', cmap='Reds')
        ax2.set_title('Monthly Loss Percentiles Over Time')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Percentile')
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(['5th', '25th', '50th', '75th', '95th'])
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("=" * 80)
        print("MBS SIMULATION SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nSimulation Parameters:")
        print(f"  Number of loans: {self.simulation.n_loans}")
        print(f"  Correlation: {self.simulation.correlation}")
        print(f"  Attachment point: {self.simulation.attachment_point:.1%}")
        print(f"  Detachment point: {self.simulation.detachment_point:.1%}")
        print(f"  Security term: {self.simulation.security_term_years} years")
        
        total_principal = sum(loan.principal for loan in self.simulation.loans)
        print(f"  Total pool principal: ${total_principal:,.0f}")
        
        print(f"\nTranche Structure:")
        for tranche in self.simulation.tranches:
            print(f"  {tranche.name}: ${tranche.principal:,.0f} ({tranche.principal/total_principal:.1%})")
        
        print(f"\nTranche Performance Statistics:")
        stats = self.results['statistics']
        for tranche_name, stat in stats.items():
            print(f"\n  {tranche_name} Tranche:")
            print(f"    Mean payoff: ${stat['mean']:,.0f}")
            print(f"    Standard deviation: ${stat['std']:,.0f}")
            print(f"    Min payoff: ${stat['min']:,.0f}")
            print(f"    Max payoff: ${stat['max']:,.0f}")
            print(f"    95% VaR: ${stat['var_95']:,.0f}")
            print(f"    99% VaR: ${stat['var_99']:,.0f}")
        
        print(f"\nStress Scenarios:")
        stress = self.results['stress_scenarios']
        for tranche_name, stress_stat in stress.items():
            print(f"\n  {tranche_name} Tranche:")
            print(f"    5% stress scenario: ${stress_stat['stress_5pct']:,.0f}")
            print(f"    1% stress scenario: ${stress_stat['stress_1pct']:,.0f}")
            print(f"    Worst case: ${stress_stat['worst_case']:,.0f}")
        
        print("=" * 80)

def run_example_simulation():
    """Run an example simulation with visualization"""
    print("Running MBS Simulation Example...")
    
    # Create simulation
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,  # 1%
        detachment_point=0.05,  # 5%
        security_term_years=7,
        loan_term_years=10,
        amortization_years=30
    )
    
    # Run Monte Carlo simulation
    results = sim.run_monte_carlo(n_simulations=1000)
    
    # Create visualizer and generate plots
    viz = MBSVisualizer(sim)
    
    # Generate summary report
    viz.create_summary_report()
    
    # Generate visualizations
    viz.plot_tranche_payoff_distributions()
    viz.plot_monthly_cashflow_evolution()
    viz.plot_tranche_comparison()
    viz.plot_loss_distribution()
    
    return sim, results, viz

if __name__ == "__main__":
    # Run example simulation
    sim, results, viz = run_example_simulation()
