#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>
#include <hip/hip_runtime.h>

#include "FileManager.hpp"
#include "BooleanNetHIP.hpp"
#include "StepMiner.hpp"
#include "InputParser.hpp"
#include "util.hpp"

using namespace std;

uint64_t round_div_up (uint64_t a, uint64_t b){
    return (a + b - 1)/b;
}

void parse_arguments(int argc, char * argv[], string & expression_file, string & implication_file, float & statThresh, float & pvalThresh, float & SMgap){
    InputParser pars(argc, argv);

    expression_file = "";
    implication_file = "";

    SMgap = -1.0;
    statThresh = 3.0;
    pvalThresh = 0.1;

    if (pars.cmdOptionExists("-h")) {cerr << "Usage: " << argv[0] << " -i <expression_file> -s <statistic_threshold> -p <p-value_threshold> -o <implication_file>" << endl; exit(0);}
    if (pars.cmdOptionExists("-i")) expression_file = pars.getCmdOption("-i");  else cerr << "Warning: no expression file specified, using default: " << expression_file << endl;
    if (pars.cmdOptionExists("-s")) statThresh = stod(pars.getCmdOption("-s")); else cerr << "Warning: no statistic threshold specified, using default: " << statThresh << endl;
    if (pars.cmdOptionExists("-p")) pvalThresh = stod(pars.getCmdOption("-p")); else cerr << "Warning: no p-value threshold specified, using default: " << pvalThresh << endl;
    if (pars.cmdOptionExists("-g")) SMgap = stof(pars.getCmdOption("-g"));  else cerr << "Warning: no SMgap specified, assuming discretized file in input" << endl;
    if (pars.cmdOptionExists("-o")) implication_file = pars.getCmdOption("-o"); else cerr << "Warning: no implication file specified, using default: " << implication_file << endl;
}

int main(int argc, char * argv[]){
    FileManager fm;

    string expression_file, implication_file;
    float statThresh, pvalThresh, SMgap;

    parse_arguments(argc, argv, expression_file, implication_file, statThresh, pvalThresh, SMgap);

    vector<string> genes;
    std::unique_ptr<char[]> expr_values_char;
    uint64_t n_rows, n_cols;

    if (SMgap < 0.0) fm.readDiscretizedFile(expression_file);
    else             fm.readRawFile(expression_file, SMgap);
   
    genes = fm.getListGenes();
    expr_values_char = fm.getMatrix();
    n_rows = fm.getNumberOfRows();
    n_cols = fm.getNumberOfColumns();

    vector<uint64_t> expr_values(n_rows * n_cols);
    vector<uint64_t> zero_flags (n_rows * n_cols);

    StepMiner::compressTernaryMatrixToBitsets(expr_values_char.get(), n_rows, n_cols, expr_values.data(), zero_flags.data());

    uint64_t nbits = sizeof(*expr_values.data()) * 8;
    uint64_t nslots = round_div_up(n_cols, nbits);

    cerr << "Expression Matrix shape: " << n_rows << " x " << n_cols << endl;
    cerr << "Number of genes: " << genes.size() << endl;

    // device initialization ----------------------------------
    int deviceCount;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "Error: No HIP devices found." << endl;
        return 1;
    }
    CHECK_HIP(hipSetDevice(0));
    
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using HIP device: " << prop.name << std::endl;

    // Mallocs --------------------------------------------
    uint64_t* d_impl_len;
    impl* d_implications;
    uint64_t* d_symm_impl_len;
    symm_impl* d_symm_implications;
    uint64_t* d_zero_flags;
    uint64_t* d_expr_values;

    CHECK_HIP(hipMalloc(&d_impl_len, sizeof(uint64_t)));
    CHECK_HIP(hipMalloc(&d_implications, MAX_N_IMP * sizeof(impl)));
    CHECK_HIP(hipMalloc(&d_symm_impl_len, sizeof(uint64_t)));
    CHECK_HIP(hipMalloc(&d_symm_implications, MAX_N_SYM_IMP * sizeof(symm_impl)));
    
    // expr_values and zero_flags size calculation
    // In original code: n_rows * nslots elements of uint64_t.
    size_t data_size_bytes = n_rows * nslots * sizeof(uint64_t);
    CHECK_HIP(hipMalloc(&d_zero_flags, data_size_bytes));
    CHECK_HIP(hipMalloc(&d_expr_values, data_size_bytes));

    // Memcpy --------------------------------------------

    CHECK_HIP(hipMemset(d_impl_len, 0, sizeof(uint64_t)));
    CHECK_HIP(hipMemset(d_symm_impl_len, 0, sizeof(uint64_t)));
    CHECK_HIP(hipMemcpy(d_zero_flags, zero_flags.data(), data_size_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_expr_values, expr_values.data(), data_size_bytes, hipMemcpyHostToDevice));

    // // Launch kernel ------------------------------------------

    auto const beginT = std::chrono::high_resolution_clock::now();
    launch_hip_kernel(d_expr_values, d_zero_flags, n_rows, n_cols, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications);
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;

    // // Copy back results --------------------------------------

    cerr << "Kernel execution completed" << endl;

    uint64_t impl_len_val;
    uint64_t symm_impl_len_val;

    CHECK_HIP(hipMemcpy(&impl_len_val, d_impl_len, sizeof(uint64_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&symm_impl_len_val, d_symm_impl_len, sizeof(uint64_t), hipMemcpyDeviceToHost));

    cerr << "Number of asymmetric implications: " << impl_len_val << endl;
    cerr << "Number of symmetric implications:  " << symm_impl_len_val << endl;

    if (impl_len_val > MAX_N_IMP || symm_impl_len_val > MAX_N_SYM_IMP){
        cerr << "Error! Too many implications!" << endl;
        exit(1);
    }

    // // Copy back results --------------------------------------

    std::vector<impl> implications(impl_len_val);
    CHECK_HIP(hipMemcpy(implications.data(), d_implications, impl_len_val * sizeof(impl), hipMemcpyDeviceToHost));

    std::vector<symm_impl> symm_implications(symm_impl_len_val);
    CHECK_HIP(hipMemcpy(symm_implications.data(), d_symm_implications, symm_impl_len_val * sizeof(symm_impl), hipMemcpyDeviceToHost));

    // // Print results ------------------------------------------

    fm.writeImplications(implication_file, genes, impl_len_val, implications.data(), symm_impl_len_val, symm_implications.data());

    // // Free memory --------------------------------------------
    CHECK_HIP(hipFree(d_impl_len));
    CHECK_HIP(hipFree(d_implications));
    CHECK_HIP(hipFree(d_symm_impl_len));
    CHECK_HIP(hipFree(d_symm_implications));
    CHECK_HIP(hipFree(d_zero_flags));
    CHECK_HIP(hipFree(d_expr_values));

    return 0;
}