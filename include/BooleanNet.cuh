#ifndef BOOLEANNET_H
#define BOOLEANNET_H

#include <string>
#include <vector>

class BooleanNet{
public:
    __host__   void get_all_implications(std::vector<std::string> genes, char* expr_values, int nsamples, float statThresh, float pvalThresh, float * implication_matrix);
    __host__ __device__ void getQuadrantCounts (int gene1, int gene2, char* expr_values, int nsamples, int * quadrant_counts);
    __host__ __device__ char is_zero (int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type);
    __host__ __device__ void getSingleImplication(int * quadrant_counts, int n_total, int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type, float * statistic, float * pval);
};

__global__ void getImplication(char * expr_values, int genes, int samples);

#endif