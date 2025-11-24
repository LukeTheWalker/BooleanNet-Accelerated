#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "FileManager.hpp"
#include "BooleanNet.hpp"
#include "InputParser.hpp"
#include "util.hpp"

using namespace std;
using namespace alpaka;

uint32_t round_div_up (uint32_t a, uint32_t b){
    return (a + b - 1)/b;
}

void StepMinerCompression (char * expression_values_char, vector<uint32_t> expr_values, vector<uint32_t> zero_flags, uint32_t ngenes, int nsamples){
    int nbits = sizeof(*expr_values.data()) * 8;
    int nslots = round_div_up(nsamples, nbits);
    for (uint32_t i = 0; i < ngenes; i++){
        for (int j = 0; j < nsamples; j++){
            int byte_to_access = j / nbits;
            int bit_to_access = j % nbits;
            int zero_flags_idx = i * nslots + byte_to_access;
            int expr_values_idx = i * nslots + byte_to_access;
            if (expression_values_char[i * nsamples + j] == -1){
            bit_ops::set(zero_flags[zero_flags_idx], bit_to_access);
            bit_ops::clear(expr_values[expr_values_idx], bit_to_access);
            }
            else if (expression_values_char[i * nsamples + j] == 1){
            bit_ops::set(zero_flags[zero_flags_idx], bit_to_access);
            bit_ops::set(expr_values[expr_values_idx], bit_to_access);
            }
            else {
            bit_ops::clear(zero_flags[zero_flags_idx], bit_to_access);
            bit_ops::clear(expr_values[expr_values_idx], bit_to_access);
            }
        }
    }
}

void launch_kernel (auto exec, auto queue, concepts::View auto d_expr_values, concepts::View auto d_zero_flags, uint32_t ngenes, int nsamples, float statThresh, float pvalThresh,  concepts::View auto d_impl_len, concepts::View auto d_implications, concepts::View auto d_symm_impl_len, concepts::View auto d_symm_implications, bool full_precision){
    using namespace alpaka;
    using Idx = uint32_t;

    int nbits = sizeof(*d_zero_flags) * 8;
    int nslots = round_div_up(nsamples, nbits);

    Vec<Idx, 2u> const lws{BLOCK_SIZE, BLOCK_SIZE};
    Vec<Idx, 2u> const gws{
        round_div_up(ngenes, lws[0]),
        round_div_up(ngenes, lws[1])
    };
    auto frameSpec = onHost::FrameSpec{gws, lws};

    cerr << "Launching kernel with " << gws[0] * gws[1] << " work-groups and " << lws[0] * lws[1] << " work-items per group" << endl;

    if (full_precision){
        BooleanNet::getImplication<double> kernel;
        auto const taskKernel = KernelBundle{kernel, d_expr_values, d_zero_flags, ngenes, nsamples, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications};

        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        queue.enqueue(exec, frameSpec, taskKernel);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    }
    else{ 
        BooleanNet::getImplication<float> kernel;
        auto const taskKernel = KernelBundle{kernel, d_expr_values, d_zero_flags, ngenes, nsamples, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications};
        
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        queue.enqueue(exec, frameSpec, taskKernel);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    }
}

void parse_arguments(int argc, char * argv[], string & expression_file, string & implication_file, float & statThresh, float & pvalThresh, bool & full_precision){
    InputParser pars(argc, argv);

    expression_file = "";
    implication_file = "";

    statThresh = 3.0;
    pvalThresh = 0.1;

    full_precision = false;

    if (pars.cmdOptionExists("-h")) {cerr << "Usage: " << argv[0] << " -i <expression_file> -s <statistic_threshold> -p <p-value_threshold> -o <implication_file>" << endl; exit(0);}
    if (pars.cmdOptionExists("-i")) expression_file = pars.getCmdOption("-i");  else cerr << "Warning: no expression file specified, using default: " << expression_file << endl;
    if (pars.cmdOptionExists("-s")) statThresh = stod(pars.getCmdOption("-s")); else cerr << "Warning: no statistic threshold specified, using default: " << statThresh << endl;
    if (pars.cmdOptionExists("-p")) pvalThresh = stod(pars.getCmdOption("-p")); else cerr << "Warning: no p-value threshold specified, using default: " << pvalThresh << endl;
    if (pars.cmdOptionExists("-o")) implication_file = pars.getCmdOption("-o"); else cerr << "Warning: no implication file specified, using default: " << implication_file << endl;
    if (pars.cmdOptionExists("-fp64")) full_precision = true; else cerr << "Warning: using compressed representation" << endl;
}

int main(int argc, char * argv[]){
    FileManager fm;

    string expression_file, implication_file;
    float statThresh, pvalThresh;

    bool full_precision;

    parse_arguments(argc, argv, expression_file, implication_file, statThresh, pvalThresh, full_precision);

    vector<string> genes;
    char * expr_values_char;
    int n_rows, n_cols;
    fm.readFile(expression_file);
   
    genes = fm.getListGenes();
    expr_values_char = fm.getMatrix();
    n_rows = fm.getNumberOfRows();
    n_cols = fm.getNumberOfColumns();

    using IdxVec = Vec<std::size_t, 1u>;
    IdxVec const extent(n_rows * n_cols);

    vector<uint32_t> expr_values(n_rows * n_cols);
    vector<uint32_t> zero_flags(n_rows * n_cols);

    StepMinerCompression(expr_values_char, expr_values, zero_flags, n_rows, n_cols);

    int nbits = sizeof(*expr_values.data()) * 8;
    int nslots = round_div_up(n_cols, nbits);

    cerr << "Expression Matrix shape: " << n_rows << " x " << n_cols << endl;
    cerr << "Number of genes: " << genes.size() << endl;

    // device initialization ----------------------------------
    auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto exec = exec::gpuCuda;

    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for " << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Mallocs --------------------------------------------

    auto impl_len   = onHost::allocHost<uint32_t>(IdxVec{1});
    auto d_impl_len = onHost::alloc<uint32_t>(devAcc, IdxVec{1});

    auto d_implications = onHost::alloc<impl>(devAcc, IdxVec{MAX_N_IMP});

    auto symm_impl_len   = onHost::allocHost<uint32_t>(IdxVec{1});
    auto d_symm_impl_len = onHost::alloc<uint32_t>(devAcc, IdxVec{1});

    auto d_symm_implications = onHost::alloc<symm_impl>(devAcc, IdxVec{MAX_N_SYM_IMP});

    auto d_zero_flags  = onHost::alloc<uint32_t>(devAcc, IdxVec{n_rows * nslots});
    auto d_expr_values = onHost::alloc<uint32_t>(devAcc, IdxVec{n_rows * nslots});


    // Memcpy --------------------------------------------

    onHost::memset(queue, d_impl_len, 0);
    onHost::memset(queue, d_symm_impl_len, 0);
    onHost::memcpy(queue, d_zero_flags, zero_flags);
    onHost::memcpy(queue, d_expr_values, expr_values);

    // // Launch kernel ------------------------------------------

    launch_kernel(exec, queue, d_expr_values, d_zero_flags, n_rows, n_cols, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications, full_precision);

    // // Copy back results --------------------------------------

    cerr << "Kernel execution completed" << endl;

    onHost::memcpy(queue, impl_len, d_impl_len);
    onHost::memcpy(queue, symm_impl_len, d_symm_impl_len);

    cerr << "Number of asymmetric implications: " << *impl_len.data() << endl;
    cerr << "Number of symmetric implications:  " << *symm_impl_len.data() << endl;

    
    if (*impl_len.data() > MAX_N_IMP || *symm_impl_len.data() > MAX_N_SYM_IMP){
        cerr << "Error! Too many implications!" << endl;
        exit(1);
    }

    // // Copy back results --------------------------------------

    auto impl_len_val = *impl_len.data();
    auto implications   = onHost::allocHost<impl>(IdxVec{impl_len_val});

    onHost::memcpy(queue, implications, d_implications, impl_len_val);

    auto symm_impl_len_val = *symm_impl_len.data();
    auto symm_implications   = onHost::allocHost<symm_impl>(IdxVec{symm_impl_len_val});
    onHost::memcpy(queue, symm_implications, d_symm_implications, symm_impl_len_val);

    // // Print results ------------------------------------------

    fm.writeImplications(implication_file, genes, *impl_len.data(), implications.data(), *symm_impl_len.data(), symm_implications.data());

    // // Free memory --------------------------------------------

    return 0;
}
