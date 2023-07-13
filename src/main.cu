#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "FileManager.cuh"
#include "BooleanNet.cuh"
#include "InputParser.hpp"
#include "util.cuh"

using namespace std;

uint32_t round_div_up (uint32_t a, uint32_t b){
    return (a + b - 1)/b;
}

void StepMinerCompression (char * expression_values_char, uint32_t *expr_values, uint32_t * zero_flags, uint32_t ngenes, int nsamples){
    int nbits = sizeof(*zero_flags) * 8;
    int nslots = round_div_up(nsamples, nbits);
    for (uint32_t i = 0; i < ngenes; i++){
        uint32_t * zero_flags_row = zero_flags + i * nslots;
        uint32_t * discretizedValues_row = expr_values + i * nslots;
        for (int j = 0; j < nsamples; j++){
            int byte_to_access = j / nbits;
            int bit_to_access = j % nbits;
            if (expression_values_char[i * nsamples + j] == -1){
                BIT_SET(*(zero_flags_row + byte_to_access), bit_to_access);
                BIT_CLEAR(*(discretizedValues_row + byte_to_access), bit_to_access);
            }
            else if (expression_values_char[i * nsamples + j] == 1){
                BIT_SET(*(zero_flags_row + byte_to_access), bit_to_access);
                BIT_SET(*(discretizedValues_row + byte_to_access), bit_to_access);

            }
            else {
                BIT_CLEAR(*(zero_flags_row + byte_to_access), bit_to_access);
                BIT_CLEAR(*(discretizedValues_row + byte_to_access), bit_to_access);
            }
        }
    }
}


void launch_kernel (uint32_t *d_expr_values, uint32_t * d_zero_flags, uint32_t ngenes, int nsamples, float statThresh, float pvalThresh, uint32_t * d_impl_len, impl * d_implications, uint32_t * d_symm_impl_len, symm_impl * d_symm_implications, bool full_precision){
    // int lws = 256;
    int nbits = sizeof(*d_zero_flags) * 8;
    int nslots = round_div_up(nsamples, nbits);
    dim3 lws(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gws(round_div_up(ngenes, lws.x), round_div_up(ngenes, lws.y), 1);
    cerr << "Launching kernel with " << gws.x << " x " << gws.y << " work-groups and " << lws.x << " x " << lws.y << " work-items per group" << endl;

    cudaError_t err;
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventRecord(start); cuda_err_check(err, __FILE__, __LINE__);

    if (full_precision)
    BooleanNet::getImplication<double><<<gws, lws>>>(d_expr_values, d_zero_flags, ngenes, nsamples, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications);
    else 
    BooleanNet::getImplication<float><<<gws, lws>>>(d_expr_values, d_zero_flags, ngenes, nsamples, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications);
    err = cudaEventRecord(stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cerr << "Kernel execution time: " << milliseconds << " ms" << endl;

    err = cudaEventDestroy(start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventDestroy(stop); cuda_err_check(err, __FILE__, __LINE__);
}

void parse_arguments(int argc, char * argv[], string & expression_file, string & implication_file, float & statThresh, float & pvalThresh, bool & full_precision){
    InputParser pars(argc, argv);

    expression_file = "/home/luca/Development/IDM/Tesi/expr_discrete/expr_big.txt";
    implication_file = "/home/luca/Development/IDM/Tesi/impl.txt";

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
    FileManager fm; cudaError_t err;

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

    uint32_t * expr_values;
    uint32_t * zero_flags;
    err = cudaMallocHost(&expr_values, sizeof(uint32_t) * n_rows * n_cols); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMallocHost(&zero_flags, sizeof(uint32_t) * n_rows * n_cols); cuda_err_check(err, __FILE__, __LINE__);

    StepMinerCompression(expr_values_char, expr_values, zero_flags, n_rows, n_cols);

    int nbits = sizeof(*zero_flags) * 8;
    int nslots = round_div_up(n_cols, nbits);

    cerr << "Expression Matrix shape: " << n_rows << " x " << n_cols << endl;
    cerr << "Number of genes: " << genes.size() << endl;

    // cuda Malloc --------------------------------------------

    uint32_t impl_len;
    uint32_t * d_impl_len;
    err = cudaMalloc(&d_impl_len, sizeof(uint32_t)); cuda_err_check(err, __FILE__, __LINE__);

    impl * d_implications;
    err = cudaMalloc(&d_implications, sizeof(impl) * MAX_N_IMP); cuda_err_check(err, __FILE__, __LINE__);
    cerr << "Max number of implications acceptable: " << MAX_N_IMP << endl;

    uint32_t symm_impl_len;
    uint32_t * d_symm_impl_len;
    err = cudaMalloc(&d_symm_impl_len, sizeof(uint32_t)); cuda_err_check(err, __FILE__, __LINE__);

    symm_impl * d_symm_implications;
    err = cudaMalloc(&d_symm_implications, sizeof(symm_impl) * MAX_N_SYM_IMP); cuda_err_check(err, __FILE__, __LINE__);
    cerr << "Max number of symmetric implications acceptable: " << MAX_N_SYM_IMP << endl;

    uint32_t * d_zero_flags;
    err = cudaMalloc(&d_zero_flags, sizeof(uint32_t) * n_rows * nslots); cuda_err_check(err, __FILE__, __LINE__);

    uint32_t * d_expr_values;
    err = cudaMalloc(&d_expr_values, sizeof(uint32_t) * n_rows * nslots); cuda_err_check(err, __FILE__, __LINE__);


    // cuda Memcpy --------------------------------------------

    err = cudaMemset(d_impl_len, 0, sizeof(uint32_t)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_symm_impl_len, 0, sizeof(uint32_t)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_zero_flags, zero_flags, sizeof(uint32_t) * n_rows * nslots, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_expr_values, expr_values, sizeof(uint32_t) * n_rows * nslots, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    // Launch kernel ------------------------------------------

    launch_kernel(d_expr_values, d_zero_flags, n_rows, n_cols, statThresh, pvalThresh, d_impl_len, d_implications, d_symm_impl_len, d_symm_implications, full_precision);

    cudaDeviceSynchronize();

    // Copy back results --------------------------------------

    cerr << "Kernel execution completed" << endl;

    err = cudaMemcpy(&impl_len, d_impl_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    cerr << "Number of asymmetric implications: " << impl_len << endl;

    err = cudaMemcpy(&symm_impl_len, d_symm_impl_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    cerr << "Number of symmetric implications:  " << symm_impl_len << endl;

    if (impl_len > MAX_N_IMP || symm_impl_len > MAX_N_SYM_IMP){
        cerr << "Error! Too many implications!" << endl;
        exit(1);
    }

    impl * implications;
    err = cudaMallocHost(&implications, sizeof(impl) * impl_len);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(implications, d_implications, sizeof(impl) * impl_len, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    symm_impl * symm_implications;
    err = cudaMallocHost(&symm_implications, sizeof(symm_impl) * symm_impl_len);

    err = cudaMemcpy(symm_implications, d_symm_implications, sizeof(symm_impl) * symm_impl_len, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    // Print results ------------------------------------------

    fm.writeImplications(implication_file, genes, impl_len, implications, symm_impl_len, symm_implications);

    // Free memory --------------------------------------------

    err = cudaFree(d_expr_values); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_impl_len); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_implications); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_symm_impl_len); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_symm_implications); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(implications); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(symm_implications); cuda_err_check(err, __FILE__, __LINE__);

    return 0;
}
