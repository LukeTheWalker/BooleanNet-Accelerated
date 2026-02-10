#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cstring>

#include "FileManager.hpp"
#include "util.hpp"
#include "StepMiner.hpp"

using namespace std;

FileManager::FileManager(){
    std::cerr << "FileManager created" << std::endl;
}

FileManager::~FileManager(){
    std::cerr << "FileManager destroyed" << std::endl;
}

uint64_t FileManager::getNumberOfColumns(string file){
    ifstream in(file);
    string line;
    string delimiter = "\t";

    getline(in, line);

    size_t pos = 0;
    string token;
    uint64_t i = 0;
    while ((pos = line.find(delimiter)) != string::npos) {
        token = line.substr(0, pos);
        line.erase(0, pos + delimiter.length());
        i++;
    }
    in.close();
    return i + 1; // accounting for last column which does not have a delimiter at the end of the line
}

uint64_t FileManager::getNumberOfRows(string file){
    ifstream in(file);
    string line;
    uint64_t i = 0;
    while(getline(in, line)){
        i++;
    }
    in.close();
    return i;
}

void FileManager::readDiscretizedFile(string file){
    string line;
    int64_t i = 0;
    string delimiter = "\t";

    n_columns = getNumberOfColumns(file);
    n_rows = getNumberOfRows(file) - 1;

    ifstream in(file);

    std::cerr << "Reading discretized file with " << n_rows << " rows and " << n_columns << " columns" << std::endl;

    discretizedMatrix = std::make_unique<char[]>(n_rows * n_columns);

    getline(in, line); // get rid of headers
    while(getline(in, line)){
        size_t pos = 0;
        string token;
        int64_t j = -1;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            if(j != -1){
                discretizedMatrix[i*n_columns+j] = stoi(token);
            }else{
                listGenes.push_back(token);
            }
            line.erase(0, pos + delimiter.length());
            j++;
        }
        discretizedMatrix[i*n_columns+j] = stoi(line);
        i++;
    }
    in.close();    
}

void FileManager::readRawFile(string file, float SMgap){
    string line;
    int64_t i = 0;
    string delimiter = "\t";

    n_columns = getNumberOfColumns(file);
    n_rows = getNumberOfRows(file) - 1;

    ifstream in(file);

    std::cerr << "Reading raw file with " << n_rows << " rows and " << n_columns << " columns" << std::endl;

    vector<double> rawMatrix(n_rows * n_columns);

    getline(in, line); // get rid of headers
    while(getline(in, line)){
        size_t pos = 0;
        string token;
        int64_t j = -1;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            if(j != -1){
                rawMatrix[i*n_columns+j] = stod(token);
            }else{
                listGenes.push_back(token);
            }
            line.erase(0, pos + delimiter.length());
            j++;
        }
        rawMatrix[i*n_columns+j] = stod(line);
        i++;
    }
    in.close();
    // Discretization
    std::cerr << "Discretizing raw file with " << n_rows << " rows and " << n_columns << " columns" << std::endl;

    discretizedMatrix = std::make_unique<char[]>(n_rows * n_columns);
    #pragma omp parallel for
    for(uint64_t r = 0; r < n_rows; r++){
        std::vector<char> discretizedRow = StepMiner::discretizeRow(&rawMatrix[r*n_columns], n_columns, SMgap);
        std::memcpy(&discretizedMatrix[r*n_columns], discretizedRow.data(), n_columns * sizeof(char));
    }
}

void FileManager::initImplicationFile(string file){
    ofstream out(file);
    out << "Implication\tStatistic(s)\tP-value(s)" << endl;
}

void FileManager::writeImplications(string file, vector<string> genes, uint64_t impl_len, impl * implications, uint64_t symm_impl_len, symm_impl * symm_implications){
    ofstream out(file);
    out << "Gene1\tGene2\tImplication\tStatistic\tP-value" << endl;
    for(uint64_t i = 0; i < impl_len; i++){
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
    for(uint64_t i = 0; i < symm_impl_len; i++){
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

unique_ptr<char[]> FileManager::getMatrix(){
    return std::move(discretizedMatrix);
}

uint64_t FileManager::getNumberOfRows(){
    return n_rows;
}

uint64_t FileManager::getNumberOfColumns(){
    return n_columns;
}