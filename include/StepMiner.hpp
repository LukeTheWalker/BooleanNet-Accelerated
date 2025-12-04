#pragma once

#include <vector>
#include <cstdint>

namespace StepMiner {
        void discretizeMatrix(double * listValues, int n_rows, int n_columns, float gap, uint64_t* discretizedValues, uint64_t* zero_flags);
        void compressTernaryToBitsets(const char* expr_vals, int n_samples, uint64_t* discretizedValues_row, uint64_t* zero_flags_row);
        void compressTernaryMatrixToBitsets(const char* expr_matrix, int n_genes, int n_samples, uint64_t* discretizedValues, uint64_t* zero_flags);
        std::vector<char> discretizeRow(const double *originalValues, int n_samples, float gap);
};