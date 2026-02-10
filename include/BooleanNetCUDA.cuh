#pragma once

#include <cstdint>
#include <vector>

const uint64_t BLOCK_SIZE = 16;

// where implication is a number between 0 and 5
// 0: gene1 low  => gene2 low
// 1: gene1 low  => gene2 high
// 2: gene1 high => gene2 low
// 3: gene1 high => gene2 high
// 4: equivalence (gene1 low <=> gene2 low)  && (gene1 high <=> gene2 high) => 0 && 3
// 5: opposite    (gene1 low <=> gene2 high) && (gene1 high <=> gene2 low)  => 1 && 2

typedef struct impl_t{
    uint64_t gene1;
    uint64_t gene2;
    char impl_type;
    float statistic;
    float pval;
} impl;

typedef struct symm_impl_t{
    uint64_t gene1;
    uint64_t gene2;
    char impl_type;
    float statistic[2];
    float pval[2];
} symm_impl;

// 4GB of memory divided by the size of a single implication
constexpr uint64_t MAX_N_IMP = (uint64_t)4e9 / (uint64_t)sizeof(impl);
constexpr uint64_t MAX_N_SYM_IMP = (uint64_t)1e9 / (uint64_t)sizeof(symm_impl);

void launch_cuda_kernel(
    const uint64_t* d_expr_values,
    const uint64_t* d_zero_flags,
    uint64_t ngenes,
    int nsamples,
    float statThresh,
    float pvalThresh,
    uint64_t* d_impl_len,
    impl* d_implications,
    uint64_t* d_symm_impl_len,
    symm_impl* d_symm_implications
);
