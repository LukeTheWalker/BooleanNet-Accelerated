#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "FileManager.cuh"
#include "BooleanNet.cuh"
#include "util.cuh"

using namespace std;

uint64_t round_div_up (uint64_t a, uint64_t b){
    return (a + b - 1)/b;
}

void launch_kernel (char * d_expr_values, uint64_t ngenes, int nsamples, BooleanNet * d_net, float statThresh, float pvalThresh){
    int lws = 256;
    uint64_t gws = round_div_up(ngenes * ngenes, lws);
    cout << "Launching kernel with " << gws << " work-groups and " << lws << " work-items per group" << " for " << ngenes*ngenes << " items" << endl;
    getImplication<<<gws, lws>>>(d_expr_values, ngenes, nsamples, d_net, statThresh, pvalThresh);
    cudaError_t err = cudaGetLastError();
    cuda_err_check(err, __FILE__, __LINE__);
}

int main(){
    FileManager fm;
    string expression_file = "/home/luca/Development/IDM/Tesi/expr_big.txt";
    string implication_file = "/home/luca/Development/IDM/Tesi/impl.txt";
    string gene1 = "";
    string gene2 = "";
    // char type = -1;
    double statThresh = 4.0;
    double pvalThresh = 0.01;
    cudaError_t err;

    if(expression_file == ""){
        cout << "Error! No expression matrix has been specified!" << endl;
        exit(1);
    }

    vector<string> genes;
    char * expr_values;
    int n_rows, n_cols;
    fm.readFile(expression_file);
   
    genes = fm.getListGenes();
    expr_values = fm.getMatrix();
    n_rows = fm.getNumberOfRows();
    n_cols = fm.getNumberOfColumns();

    cout << "Expression Matrix shape: " << n_rows << " x " << n_cols << endl;
    cout << "Number of genes: " << genes.size() << endl;

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

    err = cudaMemcpy(d_expr_values, expr_values, sizeof(char) * n_rows * n_cols, cudaMemcpyHostToDevice);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_net, &net, sizeof(BooleanNet), cudaMemcpyHostToHost);
    cuda_err_check(err, __FILE__, __LINE__);

    cout << "Instantiated Implications Matrix of size: " << genes.size() * genes.size() /** 4 * 5*/ << endl;

    launch_kernel(d_expr_values, n_rows, n_cols, d_net, statThresh, pvalThresh);

    // net.get_all_implications(genes, expr_values, n_cols, statThresh, pvalThresh, implications);

    // err = cudaFreeHost(implications);
    // cuda_err_check(err, __FILE__, __LINE__);

    cudaDeviceSynchronize();

    err = cudaFree(d_expr_values);
    cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_net);
    cuda_err_check(err, __FILE__, __LINE__);

    return 0;
}
