#ifndef FILEMANAGER_H
#define FILEMANAGER_H

#include <vector>
#include <string>

class FileManager{
    public:
        FileManager();
        ~FileManager();
        void readFile(std::string file);
        void initImplicationFile(std::string file);
        void writeImplications(std::vector<std::vector<std::string>> listImplications, std::string file);
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
