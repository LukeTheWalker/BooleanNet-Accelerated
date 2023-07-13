#ifndef BOOLEANNET_H
#define BOOLEANNET_H

#include <string>
#include <vector>

const int BLOCK_SIZE = 16;


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


namespace BooleanNet{
    template<typename T> __global__ void getImplication(uint32_t * expr_values, uint32_t * zero_flags, uint32_t ngenes, int nsamples, float statThresh, float pvalThresh, uint32_t * impl_len, impl * implications, uint32_t * symm_impl_len, symm_impl * symm_implications);
};

// 4GB of memory divided by the size of a single implication
const uint32_t MAX_N_IMP = (uint32_t)4e9 / (uint32_t)sizeof(impl);
const uint32_t MAX_N_SYM_IMP = (uint32_t)1e9 / (uint32_t)sizeof(symm_impl);

#endif