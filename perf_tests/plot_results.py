import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Load data
results_file = 'build/perf_results.csv'
if not os.path.exists(results_file):
    print(f"Error: {results_file} not found. Please run the benchmark first.")
    exit(1)

df = pd.read_csv(results_file)

# Calculate derived metrics
df['n_pairs'] = (df['n_genes'] * (df['n_genes'] - 1)) / 2
df['throughput'] = df['n_pairs'] / df['time_sec']

# Separate backends
backends = df['backend'].unique()
colors = {
    'CPU_Serial': '#d62728', # Red
    'CPU_OpenMP': '#ff7f0e', # Orange
    'GPU_CUDA': '#1f77b4',   # Blue
    'GPU_HIP': '#9467bd',    # Purple
    'GPU_OneAPI': '#2ca02c'  # Green
}
markers = {
    'CPU_Serial': 'o',
    'CPU_OpenMP': 's',
    'GPU_CUDA': '^',
    'GPU_HIP': 'v',
    'GPU_OneAPI': 'D'
}

# --- Plot 1: Execution Time (Log Scale) ---
plt.figure(figsize=(10, 6))

for backend in backends:
    data = df[df['backend'] == backend]
    plt.plot(data['n_genes'], data['time_sec'], 
             marker=markers.get(backend, 'o'), 
             label=backend.replace('_', ' '), 
             color=colors.get(backend, 'black'),
             linewidth=2, markersize=8)

plt.yscale('log')
plt.xlabel('Number of Genes ($N$)', fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.title('Execution Time vs Dataset Size', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('execution_time_measured.png', dpi=300)
print("Saved execution_time_measured.png")

# --- Plot 2: Speedup vs CPU_Serial ---
plt.figure(figsize=(10, 6))

# Find baseline (CPU_Serial)
baseline = df[df['backend'] == 'CPU_Serial'].set_index('n_genes')['time_sec']

if not baseline.empty:
    for backend in backends:
        if backend == 'CPU_Serial': continue
        
        data = df[df['backend'] == backend].set_index('n_genes')
        # Calculate speedup relative to baseline for matching N
        speedup = baseline / data['time_sec']
        
        plt.plot(speedup.index, speedup, 
                 marker=markers.get(backend, 'o'), 
                 label=f"{backend.replace('_', ' ')} Speedup", 
                 color=colors.get(backend, 'black'),
                 linewidth=2, markersize=8)

    plt.xlabel('Number of Genes ($N$)', fontsize=14)
    plt.ylabel('Speedup Factor ($T_{Serial} / T_{Backend}$)', fontsize=14)
    plt.title('Speedup Relative to Serial CPU', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('speedup_measured.png', dpi=300)
    print("Saved speedup_measured.png")
else:
    print("Warning: CPU_Serial backend not found in results, skipping speedup plot.")

# --- Plot 3: Throughput ---
plt.figure(figsize=(10, 6))

for backend in backends:
    data = df[df['backend'] == backend]
    # Throughput in Billions of pairs per second
    plt.plot(data['n_genes'], data['throughput'] / 1e9, 
             marker=markers.get(backend, 'o'), 
             label=backend.replace('_', ' '), 
             color=colors.get(backend, 'black'),
             linewidth=2, markersize=8)

plt.xlabel('Number of Genes ($N$)', fontsize=14)
plt.ylabel('Throughput ($10^9$ Pairs/sec)', fontsize=14)
plt.title('Computational Throughput', fontsize=16)
plt.legend(fontsize=12)
plt.yscale('log') # Log scale for throughput often helps visualize GPU vs CPU gap
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('throughput_measured.png', dpi=300)
print("Saved throughput_measured.png")
