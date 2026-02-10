#pragma once

#include <vector>
#include <string>

#include "BooleanNetCUDA.cuh"

class FileManager{
    public:
        FileManager();
        ~FileManager();
        void readDiscretizedFile(std::string file);
        void readRawFile(std::string file, float SMgap);
        void initImplicationFile(std::string file);
        void writeImplications(std::string file, std::vector<std::string> genes, uint64_t impl_len, impl * implications, uint64_t symm_impl_len, symm_impl * symm_implications);
        std::vector<std::string> getListGenes();
        std::unique_ptr<char[]> getMatrix();
        uint64_t getNumberOfRows();
        uint64_t getNumberOfColumns();
    private:
        std::vector<std::string> listGenes = {};
        std::unique_ptr<char[]> discretizedMatrix;
        uint64_t n_rows = 0;
        uint64_t n_columns = 0;
        uint64_t getNumberOfColumns(std::string);
        uint64_t getNumberOfRows(std::string);
};