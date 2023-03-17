#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "FileManager.cuh"
#include "BooleanNet.cuh"
#include "util.cuh"

using namespace std;

int main(){
    FileManager fm;
    string expression_file = "/home/luca/Development/IDM/Tesi/expr_small.txt";
    string implication_file = "/home/luca/Development/IDM/Tesi/impl.txt";
    string gene1 = "";
    string gene2 = "";
    char type = -1;
    double statThresh = 6.0;
    double pvalThresh = 0.01;

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

    BooleanNet net;

    float * implications;
    // cudaError_t err = cudaMallocHost(&implications, sizeof(float) * genes.size() * genes.size() * 4 * 5);
    // cuda_err_check(err, __FILE__, __LINE__);

    cout << "Instantiated Implications Matrix of size: " << genes.size() * genes.size() * 4 * 5<< endl;

    net.get_all_implications(genes, expr_values, n_cols, statThresh, pvalThresh, implications);

    // err = cudaFreeHost(implications);
    // cuda_err_check(err, __FILE__, __LINE__);

    return 0;
}
