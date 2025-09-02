#!/usr/bin/env python3
"""
Parallel Performance Test
This script compares sequential vs parallel Monte Carlo simulation performance
"""

import time
import numpy as np
from mbs_simulation import MBSSimulation
import multiprocessing as mp

def test_performance_comparison():
    """Compare sequential vs parallel performance"""
    print("PARALLEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create simulation
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    print(f"System info:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Number of loans: {sim.n_loans}")
    print(f"  Security term: {sim.security_term_years} years")
    
    # Test different simulation sizes
    simulation_sizes = [100, 500, 1000, 2000]
    
    for n_sims in simulation_sizes:
        print(f"\n{'='*40}")
        print(f"Testing with {n_sims} simulations")
        print(f"{'='*40}")
        
        # Test sequential
        print("Running sequential simulation...")
        start_time = time.time()
        results_seq = sim.run_monte_carlo(n_simulations=n_sims, use_parallel=False)
        seq_time = time.time() - start_time
        
        # Test parallel
        print("Running parallel simulation...")
        start_time = time.time()
        results_par = sim.run_monte_carlo(n_simulations=n_sims, use_parallel=True)
        par_time = time.time() - start_time
        
        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else float('inf')
        
        print(f"\nResults for {n_sims} simulations:")
        print(f"  Sequential time: {seq_time:.2f} seconds")
        print(f"  Parallel time:   {par_time:.2f} seconds")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Efficiency:     {(speedup/mp.cpu_count())*100:.1f}%")
        
        # Verify results are identical
        for tranche_name in results_seq['tranche_payoffs'].keys():
            seq_mean = np.mean(results_seq['tranche_payoffs'][tranche_name])
            par_mean = np.mean(results_par['tranche_payoffs'][tranche_name])
            diff = abs(seq_mean - par_mean)
            print(f"  {tranche_name} mean difference: {diff:.6f}")

def test_parallel_scaling():
    """Test how performance scales with different numbers of processes"""
    print("\n" + "=" * 60)
    print("PARALLEL SCALING TEST")
    print("=" * 60)
    
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    n_simulations = 2000
    max_cores = min(mp.cpu_count(), 8)  # Test up to 8 cores
    
    print(f"Testing with {n_simulations} simulations")
    print(f"Testing up to {max_cores} cores")
    
    # Test sequential first
    print("\nRunning sequential baseline...")
    start_time = time.time()
    sim.run_monte_carlo(n_simulations=n_simulations, use_parallel=False)
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f} seconds")
    
    # Test different numbers of processes
    for n_jobs in range(2, max_cores + 1):
        print(f"\nTesting with {n_jobs} processes...")
        start_time = time.time()
        sim.run_monte_carlo(n_simulations=n_simulations, use_parallel=True, n_jobs=n_jobs)
        par_time = time.time() - start_time
        
        speedup = seq_time / par_time
        efficiency = (speedup / n_jobs) * 100
        
        print(f"  Time: {par_time:.2f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.1f}%")

def test_memory_usage():
    """Test memory usage comparison"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE COMPARISON")
    print("=" * 60)
    
    import psutil
    import os
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    sim = MBSSimulation(
        n_loans=50,
        correlation=0.3,
        attachment_point=0.01,
        detachment_point=0.05,
        security_term_years=7
    )
    
    n_simulations = 1000
    
    # Test sequential memory usage
    print("Testing sequential memory usage...")
    initial_memory = get_memory_usage()
    results_seq = sim.run_monte_carlo(n_simulations=n_simulations, use_parallel=False)
    seq_memory = get_memory_usage() - initial_memory
    
    # Test parallel memory usage
    print("Testing parallel memory usage...")
    initial_memory = get_memory_usage()
    results_par = sim.run_monte_carlo(n_simulations=n_simulations, use_parallel=True)
    par_memory = get_memory_usage() - initial_memory
    
    print(f"\nMemory usage comparison:")
    print(f"  Sequential: {seq_memory:.1f} MB")
    print(f"  Parallel:   {par_memory:.1f} MB")
    print(f"  Ratio:      {par_memory/seq_memory:.2f}x")

def main():
    """Run all performance tests"""
    try:
        # Test basic performance comparison
        test_performance_comparison()
        
        # Test parallel scaling
        test_parallel_scaling()
        
        # Test memory usage (if psutil is available)
        try:
            import psutil
            test_memory_usage()
        except ImportError:
            print("\nSkipping memory usage test (psutil not available)")
        
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        raise

if __name__ == "__main__":
    main()
