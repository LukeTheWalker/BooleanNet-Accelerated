#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "FileManager.cuh"
#include "BooleanNet.cuh"
#include "InputParser.hpp"
#include "util.cuh"

#define MAX_N_IMP 25000000

using namespace std;

uint64_t round_div_up (uint64_t a, uint64_t b){
    return (a + b - 1)/b;
}

void launch_kernel (char * d_expr_values, uint64_t ngenes, int nsamples, BooleanNet * d_net, float statThresh, float pvalThresh, uint32_t * d_impl_len, implication * d_implications){
    int lws = 256;
    uint64_t gws = round_div_up(ngenes * ngenes, lws);
    cerr << "Launching kernel with " << gws << " work-groups and " << lws << " work-items per group" << " for " << ngenes*ngenes << " items" << endl;
    getImplication<<<gws, lws>>>(d_expr_values, ngenes, nsamples, d_net, statThresh, pvalThresh, d_impl_len, d_implications);
    cudaError_t err = cudaGetLastError();
    cuda_err_check(err, __FILE__, __LINE__);
}

int main(int argc, char * argv[]){
    FileManager fm; cudaError_t err; InputParser pars(argc, argv);


    string expression_file = "/home/luca/Development/IDM/Tesi/expr_discrete/expr_big.txt";
    string implication_file = "/home/luca/Development/IDM/Tesi/impl.txt";

    float statThresh = 3.9;
    float pvalThresh = 0.01;

    if (pars.cmdOptionExists("-h")) {cerr << "Usage: " << argv[0] << " -i <expression_file> -s <statistic_threshold> -p <p-value_threshold>" << endl; exit(0);}
    if (pars.cmdOptionExists("-i")) expression_file = pars.getCmdOption("-i");  else cerr << "Warning: no expression file specified, using default: " << expression_file << endl;
    if (pars.cmdOptionExists("-s")) statThresh = stod(pars.getCmdOption("-s")); else cerr << "Warning: no statistic threshold specified, using default: " << statThresh << endl;
    if (pars.cmdOptionExists("-p")) pvalThresh = stod(pars.getCmdOption("-p")); else cerr << "Warning: no p-value threshold specified, using default: " << pvalThresh << endl;

    vector<string> genes;
    char * expr_values;
    int n_rows, n_cols;
    fm.readFile(expression_file);
   
    genes = fm.getListGenes();
    expr_values = fm.getMatrix();
    n_rows = fm.getNumberOfRows();
    n_cols = fm.getNumberOfColumns();

    cerr << "Expression Matrix shape: " << n_rows << " x " << n_cols << endl;
    cerr << "Number of genes: " << genes.size() << endl;

    // cuda Malloc --------------------------------------------

    // float * implications;
    // err = cudaMallocHost(&implications, sizeof(float) * genes.size() * genes.size() * 4 * 5);
    // cuda_err_check(err, __FILE__, __LINE__);

    BooleanNet net;
    BooleanNet * d_net;
    err = cudaMalloc(&d_net, sizeof(BooleanNet));
    cuda_err_check(err, __FILE__, __LINE__);

    char * d_expr_values;
    err = cudaMalloc(&d_expr_values, sizeof(char) * n_rows * n_cols);
    cuda_err_check(err, __FILE__, __LINE__);

    uint32_t impl_len;
    uint32_t * d_impl_len;
    err = cudaMalloc(&d_impl_len, sizeof(uint32_t));
    cuda_err_check(err, __FILE__, __LINE__);

    implication * d_implications;
    err = cudaMalloc(&d_implications, sizeof(implication) * MAX_N_IMP);
    cuda_err_check(err, __FILE__, __LINE__);

    // cuda Memcpy --------------------------------------------

    err = cudaMemcpy(d_expr_values, expr_values, sizeof(char) * n_rows * n_cols, cudaMemcpyHostToDevice);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_net, &net, sizeof(BooleanNet), cudaMemcpyHostToDevice);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemset(d_impl_len, 0, sizeof(uint32_t));
    cuda_err_check(err, __FILE__, __LINE__);

    cerr << "Instantiated Implications Matrix of size: " << genes.size() * genes.size() /** 4 * 5*/ << endl;

    launch_kernel(d_expr_values, n_rows, n_cols, d_net, statThresh, pvalThresh, d_impl_len, d_implications);

    // net.get_all_implications(genes, expr_values, n_cols, statThresh, pvalThresh, implications);

    // err = cudaFreeHost(implications);
    // cuda_err_check(err, __FILE__, __LINE__);

    cudaDeviceSynchronize();

    cerr << "Kernel execution completed" << endl;

    err = cudaMemcpy(&impl_len, d_impl_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    cerr << "Number of implications: " << impl_len << endl;

    if (impl_len > MAX_N_IMP){
        cerr << "Error! Too many implications!" << endl;
        exit(1);
    }

    implication * implications;
    err = cudaMallocHost(&implications, sizeof(implication) * impl_len);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(implications, d_implications, sizeof(implication) * impl_len, cudaMemcpyDeviceToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    for (int i = 0; i < impl_len; i++){
        cout << genes[implications[i].gene1] << " " << genes[implications[i].gene2] << " " << (int)implications[i].impl_type << " " << implications[i].statistic << " " << implications[i].pval << endl;
    }

    err = cudaFree(d_expr_values);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_net);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_impl_len);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_implications);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(implications);
    cuda_err_check(err, __FILE__, __LINE__);

    return 0;
}
