#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include "BooleanNet.hpp"

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

// Generic Benchmark Runner
// Runs the kernel for a specific backend (device + executor)
template<typename Backend>
double run_benchmark(Backend const& backend, uint64_t n_genes, uint64_t n_samples, 
                     const std::vector<uint64_t>& h_expr_vec, const std::vector<uint64_t>& h_zero_vec) {

    using namespace alpaka;
    using Idx = uint64_t;
    
    // Extract device and executor from the backend tuple
    auto const& deviceSpec = backend[alpaka::object::deviceSpec];
    auto const& exec = backend[alpaka::object::exec];

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    // Use the first available device
    if (devSelector.getDeviceCount() == 0) {
        throw std::runtime_error("No device found for this backend");
    }
    auto devAcc = devSelector.makeDevice(0);
    auto queue = devAcc.makeQueue();

    float statThresh = 3.0f;
    float pvalThresh = 0.1f;
    
    uint64_t nbits = 64;
    uint64_t nslots = round_div_up(n_samples, nbits);
    
    // 1. Allocate Host Buffers (Pinned/Page-locked if supported)
    // We copy from std::vector to these Alpaka buffers to ensure efficient transfer
    using IdxVec = Vec<Idx, 1u>;
    auto h_expr_buf = onHost::allocHost<uint64_t>(IdxVec{n_genes * nslots});
    auto h_zero_buf = onHost::allocHost<uint64_t>(IdxVec{n_genes * nslots});
    
    // Populate host buffers
    // Use pointers to copy
    std::copy(h_expr_vec.begin(), h_expr_vec.end(), &h_expr_buf[0]);
    std::copy(h_zero_vec.begin(), h_zero_vec.end(), &h_zero_buf[0]);

    // 2. Allocate Device Buffers
    auto d_expr = onHost::alloc<uint64_t>(devAcc, IdxVec{n_genes * nslots});
    auto d_zero = onHost::alloc<uint64_t>(devAcc, IdxVec{n_genes * nslots});
    
    // Output buffers
    uint64_t max_imp = MAX_N_IMP; 
    auto d_impl_len = onHost::alloc<uint64_t>(devAcc, IdxVec{1});
    auto d_symm_impl_len = onHost::alloc<uint64_t>(devAcc, IdxVec{1});
    auto d_implications = onHost::alloc<impl>(devAcc, IdxVec{max_imp});
    auto d_symm_implications = onHost::alloc<symm_impl>(devAcc, IdxVec{MAX_N_SYM_IMP});

    // 3. Copy Host -> Device
    memcpy(queue, d_expr, h_expr_buf);
    memcpy(queue, d_zero, h_zero_buf);
    
    memset(queue, d_impl_len, 0);
    memset(queue, d_symm_impl_len, 0);

    // 4. Define Grid/Block
    Vec<Idx, 2u> const lws{BLOCK_SIZE, BLOCK_SIZE};
    Vec<Idx, 2u> const gws{
        round_div_up(n_genes, lws[0]),
        round_div_up(n_genes, lws[1])
    };
    auto frameSpec = onHost::FrameSpec{gws, lws};

    // 5. Kernel Setup
    BooleanNet::getImplication kernel;
    auto const taskKernel = KernelBundle{
        kernel, 
        d_expr, d_zero, 
        n_genes, n_samples, 
        statThresh, pvalThresh, 
        d_impl_len, d_implications, 
        d_symm_impl_len, d_symm_implications
    };

    // 6. Execute and Time
    wait(queue); // Ensure copies are done
    auto start = std::chrono::high_resolution_clock::now();
    
    queue.enqueue(exec, frameSpec, taskKernel);
    wait(queue); 
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main() {
    std::vector<uint64_t> gene_counts = {1000, 5000, 10000, 15000, 20000, 25000};
    uint64_t n_samples = 500;
    
    std::ofstream csv("perf_results.csv");
    csv << "backend,n_genes,time_sec\n";
    
    // Iterate over gene counts
    for (auto n : gene_counts) {
        std::cout << "Benchmarking N=" << n << "..." << std::endl;
        
        std::vector<uint64_t> h_expr, h_zero;
        generate_random_data(h_expr, h_zero, n, n_samples);
        
        // Iterate over all available backends using Alpaka helper
        alpaka::onHost::executeForEachIfHasDevice(
            [&](auto const& backend) {
                try {
                    auto const& deviceSpec = backend[alpaka::object::deviceSpec];
                    auto const& exec = backend[alpaka::object::exec];
                    
                    std::string backendName = alpaka::onHost::demangledName(exec);
                    // Simplify name for CSV
                    if (backendName.find("CpuSerial") != std::string::npos) backendName = "CPU_Serial";
                    else if (backendName.find("GpuCuda") != std::string::npos) backendName = "GPU_CUDA";
                    else if (backendName.find("GpuHip") != std::string::npos) backendName = "GPU_HIP";
                    else if (backendName.find("CpuOmp") != std::string::npos) backendName = "CPU_OpenMP";
                    
                    std::cout << "  Running on " << backendName << "..." << std::flush;
                    
                    double time = run_benchmark(backend, n, n_samples, h_expr, h_zero);
                    
                    std::cout << " " << time << "s" << std::endl;
                    
                    csv << backendName << "," << n << "," << time << "\n";
                    csv.flush();
                } catch (const std::exception& e) {
                    std::cerr << "    Failed: " << e.what() << std::endl;
                }
            },
            alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors)
        );
    }
    
    csv.close();
    std::cout << "Results written to perf_results.csv" << std::endl;
    return 0;
}
