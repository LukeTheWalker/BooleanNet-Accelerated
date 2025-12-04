#include "StepMiner.hpp"
#include "util.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

std::vector<char> StepMiner::discretizeRow(const double* originalValues, int n_samples, float gap) {
    std::vector<double> values(n_samples);
    std::copy(originalValues, originalValues + n_samples, values.begin());
    std::sort(values.begin(), values.end());

    int bestPos = -1;
    double minSSE = std::numeric_limits<double>::max();
    for (int i = 0; i < n_samples; i++) {
        double currentSSE = 0.0;
        double leftSum = 0.0;
        for (int j = 0; j <= i; j++)
            leftSum += values[j];
        leftSum /= (i + 1);
        for (int j = 0; j <= i; j++)
            currentSSE += pow(values[i] - leftSum, 2);
        double rightSum = 0.0;
        for (int j = i + 1; j < n_samples; j++)
            rightSum += values[j];
        rightSum /= (n_samples - i - 1);
        for (int j = i + 1; j < n_samples; j++)
            currentSSE += pow(values[i] - rightSum, 2);
        if (currentSSE < minSSE) {
            minSSE = currentSSE;
            bestPos = i;
        }
    }

    double threshold = values[bestPos];
    double lowerBound = threshold - gap;
    double upperBound = threshold + gap;

    std::vector<char> expr_vals(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        if (originalValues[i] < lowerBound) expr_vals[i] = -1;
        else if (originalValues[i] > upperBound) expr_vals[i] = 1;
        else expr_vals[i] = 0;
    }
    return expr_vals;
}

void StepMiner::compressTernaryToBitsets(const char* expr_vals, int n_samples, uint64_t* discretizedValues_row, uint64_t* zero_flags_row) {
    uint64_t n_bytes = (n_samples + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8);
    // initialize rows to zero to avoid leftover bits
    for (uint64_t b = 0; b < n_bytes; ++b) {
        discretizedValues_row[b] = 0;
        zero_flags_row[b] = 0;
    }

    for (uint64_t j = 0; j < static_cast<uint64_t>(n_samples); ++j) {
        uint64_t byte_to_access = j / (sizeof(uint64_t) * 8);
        uint64_t bit_to_access  = j % (sizeof(uint64_t) * 8);
        if (expr_vals[j] == -1) {
            bit_ops::set(zero_flags_row[byte_to_access], bit_to_access);
            bit_ops::clear(discretizedValues_row[byte_to_access], bit_to_access);
        }
        else if (expr_vals[j] == 1) {
            bit_ops::set(zero_flags_row[byte_to_access], bit_to_access);
            bit_ops::set(discretizedValues_row[byte_to_access], bit_to_access);
        }
        else {
            bit_ops::clear(zero_flags_row[byte_to_access], bit_to_access);
            bit_ops::clear(discretizedValues_row[byte_to_access], bit_to_access);
        }
    }
}

void StepMiner::compressTernaryMatrixToBitsets(const char* expr_matrix, int n_genes, int n_samples, uint64_t* discretizedValues, uint64_t* zero_flags) {
    uint64_t n_bytes = (n_samples + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8);
    for (int row = 0; row < n_genes; ++row) {
        const char* row_ptr = expr_matrix + static_cast<uint64_t>(row) * static_cast<uint64_t>(n_samples);
        uint64_t* disc_row = discretizedValues + static_cast<uint64_t>(row) * n_bytes;
        uint64_t* zero_row = zero_flags + static_cast<uint64_t>(row) * n_bytes;
        compressTernaryToBitsets(row_ptr, n_samples, disc_row, zero_row);
    }
}

void StepMiner::discretizeMatrix(double* listValues, int n_genes, int n_samples, float gap, uint64_t* discretizedValues, uint64_t* zero_flags) {
    int n_bytes = (n_samples + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8);
    for (int row = 0; row < n_genes; row++) {
        double* originalValues = listValues + row * n_samples;
        int row_offset = row * n_bytes;
        uint64_t* zero_flags_row = zero_flags + row_offset;
        uint64_t* discretizedValues_row = discretizedValues + row_offset;

        std::vector<char> expr_vals = discretizeRow(originalValues, n_samples, gap);
        compressTernaryToBitsets(expr_vals.data(), n_samples, discretizedValues_row, zero_flags_row);
    }
    return;
}