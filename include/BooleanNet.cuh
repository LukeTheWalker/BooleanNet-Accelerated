#ifndef BOOLEANNET_H
#define BOOLEANNET_H

#include <string>
#include <vector>


// where implication is a number between 0 and 5
// 0: gene1 low  => gene2 low
// 1: gene1 low  => gene2 high
// 2: gene1 high => gene2 low
// 3: gene1 high => gene2 high
// 4: equivalence (gene1 low <=> gene2 low)  && (gene1 high <=> gene2 high) => 0 && 3
// 5: opposite    (gene1 low <=> gene2 high) && (gene1 high <=> gene2 low)  => 1 && 2

typedef struct impl_t{
    int gene1;
    int gene2;
    char impl_type;
    float statistic;
    float pval;
} impl;

typedef struct symm_impl_t{
    int gene1;
    int gene2;
    char impl_type;
    float statistic[2];
    float pval[2];
} symm_impl;


class BooleanNet{
public:
    __host__   void get_all_implications(std::vector<std::string> genes, char* expr_values, int nsamples, float statThresh, float pvalThresh, float * impl_matrix);
    __host__ __device__ void getQuadrantCounts (int gene1, int gene2, char* expr_values, int nsamples, int * quadrant_counts);
    __host__ __device__ char is_zero (int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type);
    __host__ __device__ void getSingleImplication(int * quadrant_counts, int n_total, int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type, float * statistic, float * pval);
};

__global__ void getImplication(char * expr_values, uint64_t ngenes, int nsamples, BooleanNet * net, float statThresh, float pvalThresh, uint32_t * impl_len, impl * implications, uint32_t * symm_impl_len, symm_impl * symm_impls);

#endif