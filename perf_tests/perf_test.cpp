#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <cuda_runtime.h>

#include "BooleanNetCUDA.cuh"

// Utility to round up division
uint64_t round_div_up (uint64_t a, uint64_t b){
    return (a + b - 1)/b;
}

// Data generation
void generate_random_data(std::vector<uint64_t>& expr, std::vector<uint64_t>& zero, uint64_t n_genes, uint64_t n_samples) {
    uint64_t nbits = 64;
    uint64_t nslots = round_div_up(n_samples, nbits);
    uint64_t size = n_genes * nslots;
    
    expr.resize(size);
    zero.resize(size);

    std::mt19937_64 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist;

    for(size_t i=0; i<size; ++i) {
        expr[i] = dist(rng);
        zero[i] = dist(rng); 
    }
}

double run_benchmark(uint64_t n_genes, uint64_t n_samples, 
                     const std::vector<uint64_t>& h_expr_vec, const std::vector<uint64_t>& h_zero_vec) {

    float statThresh = 3.0f;
    float pvalThresh = 0.1f;
    
    uint64_t nbits = 64;
    uint64_t nslots = round_div_up(n_samples, nbits);
    
    // Allocate Device Buffers
    uint64_t* d_expr;
    uint64_t* d_zero;
    uint64_t* d_impl_len;
    uint64_t* d_symm_impl_len;
    impl* d_implications;
    symm_impl* d_symm_implications;

    size_t data_size = n_genes * nslots * sizeof(uint64_t);
    cudaMalloc(&d_expr, data_size);
    cudaMalloc(&d_zero, data_size);
    
    cudaMalloc(&d_impl_len, sizeof(uint64_t));
    cudaMalloc(&d_symm_impl_len, sizeof(uint64_t));
    cudaMalloc(&d_implications, MAX_N_IMP * sizeof(impl));
    cudaMalloc(&d_symm_implications, MAX_N_SYM_IMP * sizeof(symm_impl));

    // Copy Host -> Device
    cudaMemcpy(d_expr, h_expr_vec.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zero, h_zero_vec.data(), data_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_impl_len, 0, sizeof(uint64_t));
    cudaMemset(d_symm_impl_len, 0, sizeof(uint64_t));

    // Execute and Time
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    launch_cuda_kernel(d_expr, d_zero, n_genes, n_samples, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Free memory
    cudaFree(d_expr);
    cudaFree(d_zero);
    cudaFree(d_impl_len);
    cudaFree(d_symm_impl_len);
    cudaFree(d_implications);
    cudaFree(d_symm_implications);
    
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main() {
    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    std::vector<uint64_t> gene_counts = {1000, 5000, 10000, 15000, 20000, 25000};
    uint64_t n_samples = 500;
    
    std::ofstream csv("perf_results.csv");
    csv << "backend,n_genes,time_sec\n";
    
    std::string backendName = "GPU_CUDA";

    for (auto n : gene_counts) {
        std::cout << "Benchmarking N=" << n << "..." << std::endl;
        
        std::vector<uint64_t> h_expr, h_zero;
        generate_random_data(h_expr, h_zero, n, n_samples);
        
        std::cout << "  Running on " << backendName << "..." << std::flush;
        
        double time = run_benchmark(n, n_samples, h_expr, h_zero);
        
        std::cout << " " << time << "s" << std::endl;
        
        csv << backendName << "," << n << "," << time << "\n";
        csv.flush();
    }
    
    csv.close();
    std::cout << "Results written to perf_results.csv" << std::endl;
    return 0;
}