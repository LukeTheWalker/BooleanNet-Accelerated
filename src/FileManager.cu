#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#include "FileManager.cuh"
#include "util.cuh"

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

    n_columns = getNumberOfColumns(file) - 1;
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
        i++;
    }
}

void FileManager::initImplicationFile(string file){
    ofstream out(file);
    out << "Implication\tStatistic(s)\tP-value(s)" << endl;
}

void FileManager::writeImplications(vector<vector<string>> listImplications, string file){
    ofstream out(file);
    for(int i = 0; i < listImplications.size(); i++){
        for(int j = 0; j < listImplications[i].size(); j++){
            out << listImplications[i][j] << "\t";
        }
        out << endl;
    }
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