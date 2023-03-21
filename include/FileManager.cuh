#ifndef FILEMANAGER_H
#define FILEMANAGER_H

#include <vector>
#include <string>

#include "BooleanNet.cuh"

class FileManager{
    public:
        FileManager();
        ~FileManager();
        void readFile(std::string file);
        void initImplicationFile(std::string file);
        void writeImplications(std::string file, std::vector<std::string> genes, uint32_t impl_len, impl * implications, uint32_t symm_impl_len, symm_impl * symm_implications);
        std::vector<std::string> getListGenes();
        char * getMatrix();
        int getNumberOfRows();
        int getNumberOfColumns();
    private:
        std::vector<std::string> listGenes = {};
        char * matrix;
        int n_rows = -1;
        int n_columns = -1;
        int getNumberOfColumns(std::string);
        int getNumberOfRows(std::string);
};

#endif
