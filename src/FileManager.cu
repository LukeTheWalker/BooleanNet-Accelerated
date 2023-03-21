#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#include "FileManager.cuh"
#include "util.cuh"
#include "BooleanNet.cuh"

using namespace std;

FileManager::FileManager(){
    std::cerr << "FileManager created" << std::endl;
}

FileManager::~FileManager(){
    cudaError_t err = cudaFreeHost(matrix);
    cuda_err_check(err, __FILE__, __LINE__);

    std::cerr << "FileManager destroyed" << std::endl;
}

int FileManager::getNumberOfColumns(string file){
    ifstream in(file);
    string line;
    string delimiter = "\t";

    getline(in, line);

    size_t pos = 0;
    string token;
    int i = 0;
    while ((pos = line.find(delimiter)) != string::npos) {
        token = line.substr(0, pos);
        line.erase(0, pos + delimiter.length());
        i++;
    }
    in.close();
    return i + 1; // accounting for last column which does not have a delimiter at the end of the line
}

int FileManager::getNumberOfRows(string file){
    ifstream in(file);
    string line;
    int i = 0;
    while(getline(in, line)){
        i++;
    }
    in.close();
    return i;
}

void FileManager::readFile(string file){
    string line;
    int i = 0;
    string delimiter = "\t";

    n_columns = getNumberOfColumns(file);
    n_rows = getNumberOfRows(file) - 1;

    ifstream in(file);

    std::cerr << "Reading file with " << n_rows << " rows and " << n_columns << " columns" << std::endl;

    cudaError_t err = cudaMallocHost(&matrix, (n_rows) * (n_columns) * sizeof(char));
    cuda_err_check(err, __FILE__, __LINE__);

    getline(in, line); // get rid of headers
    while(getline(in, line)){
        size_t pos = 0;
        string token;
        int j = -1;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            if(j != -1){
                matrix[i*n_columns+j] = stoi(token);
            }else{
                listGenes.push_back(token);
            }
            line.erase(0, pos + delimiter.length());
            j++;
        }
        matrix[i*n_columns+j] = stoi(line);
        i++;
    }
    in.close();
}

void FileManager::initImplicationFile(string file){
    ofstream out(file);
    out << "Implication\tStatistic(s)\tP-value(s)" << endl;
}

void FileManager::writeImplications(string file, vector<string> genes, uint32_t impl_len, impl * implications, uint32_t symm_impl_len, symm_impl * symm_implications){
    ofstream out(file);
    out << "Gene1\tGene2\tImplication\tStatistic\tP-value" << endl;
    for(int i = 0; i < impl_len; i++){
        out << 
            genes[implications[i].gene1] << "\t" << 
            genes[implications[i].gene2] << "\t" << 
            get_impl_string(implications[i].impl_type) << "\t" << 
            implications[i].statistic << "\t" << 
            implications[i].pval << endl;
    }
    out.close();
    string symm_file = file.substr(0, file.length() - 4) + "_symm.txt";
    ofstream out_symm(symm_file);
    out_symm << "Gene1\tGene2\tImplication\tStatistic1\tStatistic2\tP-value1\tP-value2" << endl;
    for(int i = 0; i < symm_impl_len; i++){
        out_symm << 
            genes[symm_implications[i].gene1] << "\t" << 
            genes[symm_implications[i].gene2] << "\t" << 
            get_impl_string(symm_implications[i].impl_type) << "\t" << 
            symm_implications[i].statistic[0] << "\t" << 
            symm_implications[i].statistic[1] << "\t" <<
            symm_implications[i].pval[0] << "\t" <<
            symm_implications[i].pval[1] << endl;
    }
    out_symm.close();
}

vector<string> FileManager::getListGenes(){
    return listGenes;
}

char * FileManager::getMatrix(){
    return matrix;
}

int FileManager::getNumberOfRows(){
    return n_rows;
}

int FileManager::getNumberOfColumns(){
    return n_columns;
}