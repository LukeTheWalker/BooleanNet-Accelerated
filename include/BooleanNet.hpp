#pragma once

#include <string>
#include <vector>
#include <alpaka/alpaka.hpp>
#include <assert.h>
#include <bit>

const unsigned int BLOCK_SIZE = 16;

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

inline ALPAKA_FN_ACC char get_inverse_implication(char impl_type){
    if (impl_type == 0){
        return 3;
    }
    else if (impl_type == 1){
        return 1;
    }
    else if (impl_type == 2){
        return 2;
    }
    else if (impl_type == 3){
        return 0;
    }
    return -1;
}

inline ALPAKA_FN_ACC void getQuadrantCounts(uint32_t gene1, uint32_t gene2,  alpaka::concepts::MdSpan auto const expr_values, alpaka::concepts::MdSpan auto const zero_flags, int nsamples, int* quadrant_counts){
    for (int i = 0; i < 4; i++){
        quadrant_counts[i] = 0;
    }
    const int nbits = sizeof(*zero_flags) * 8;
    const int nslots = (nsamples + nbits - 1) / nbits;
    for (int i = 0; i < nslots; i++){
        const uint32_t gene1_slot = expr_values[gene1 * nslots + i];
        const uint32_t gene2_slot = expr_values[gene2 * nslots + i];
        const uint32_t zero_slot = zero_flags[gene1 * nslots + i] & zero_flags[gene2 * nslots + i];
        const uint32_t gene1_slot_low = ~gene1_slot & zero_slot;
        const uint32_t gene1_slot_high = gene1_slot & zero_slot;
        const uint32_t gene2_slot_low = ~gene2_slot & zero_slot;
        const uint32_t gene2_slot_high = gene2_slot & zero_slot;
        quadrant_counts[0] += std::popcount(gene1_slot_low & gene2_slot_low);
        quadrant_counts[1] += std::popcount(gene1_slot_low & gene2_slot_high);
        quadrant_counts[2] += std::popcount(gene1_slot_high & gene2_slot_low);
        quadrant_counts[3] += std::popcount(gene1_slot_high & gene2_slot_high);
    }
}

inline ALPAKA_FN_ACC char is_zero(int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type){
    if (impl_type == 0){
        if (n_first_low > 0 && n_second_high > 0)
            return 0;
    }
    else if (impl_type == 1){
        if (n_first_low > 0 && n_second_low > 0)
            return 0;
    }
    else if (impl_type == 2){
        if (n_first_high > 0 && n_second_high > 0)
            return 0;
    }
    else if (impl_type == 3){
        if (n_first_high > 0 && n_second_low > 0)
            return 0;
    }
    else {
        printf("Invalid impl_type in is_zero\n");
    }
    return 1;
}

template<typename T>
ALPAKA_FN_ACC void getSingleImplication(int* quadrant_counts, int n_total, int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type, float* statistic, float* pval){
    if (is_zero(n_first_low, n_first_high, n_second_low, n_second_high, impl_type)){
        *statistic = 0.0;
        *pval = 1.0;
        return;
    }

    if (impl_type == 0){
        T n_expected = (T)(n_first_low * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[1]) / __fsqrt_rn(n_expected);
        *pval = ((((T)quadrant_counts[1] / n_first_low) + ((T)quadrant_counts[1] / n_second_high)) / 2);
    }
    else if (impl_type == 1){
        T n_expected = (T)(n_first_low * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[0]) / __fsqrt_rn(n_expected);
        *pval = ((((T)quadrant_counts[0] / n_first_low) + ((T)quadrant_counts[0] / n_second_low)) / 2);
    }
    else if (impl_type == 2){
        T n_expected = (T)(n_first_high * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[3]) / __fsqrt_rn(n_expected);
        *pval = ((((T)quadrant_counts[3] / n_first_high) + ((T)quadrant_counts[3] / n_second_high)) / 2);
    }
    else if (impl_type == 3){
        T n_expected = (T)(n_first_high * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[2]) / __fsqrt_rn(n_expected);
        *pval = ((((T)quadrant_counts[2] / n_first_high) + ((T)quadrant_counts[2] / n_second_low)) / 2);
    }
}

namespace BooleanNet{
    template<typename T> 
    struct getImplication {
        ALPAKA_FN_ACC auto operator() (
            alpaka::onAcc::concepts::Acc auto const & acc,
            alpaka::concepts::MdSpan auto const expr_values, alpaka::concepts::MdSpan auto const zero_flags, 
            uint32_t ngenes, int nsamples, 
            float statThresh, float pvalThresh,
            alpaka::concepts::IMdSpan auto impl_len, alpaka::concepts::IMdSpan auto implications, 
            alpaka::concepts::IMdSpan auto symm_impl_len, alpaka::concepts::IMdSpan auto symm_implications) const -> void {

            auto threadIndexMD = acc[alpaka::layer::thread].idx();
            auto blockDimensionMD = acc[alpaka::layer::thread].count();
            auto blockIndexMD = acc[alpaka::layer::block].idx();
            auto gridDimensionMD = acc[alpaka::layer::block].count();

            // The global thread index is calculated in 2D
            auto gridThreadIndexMD = blockDimensionMD * blockIndexMD + threadIndexMD;
            // The total number of threads in the grid is also calculated in 2D
            auto gridSizeMD = gridDimensionMD * blockDimensionMD;

            // Loop over the 2D data grid
            for (auto gene1 = gridThreadIndexMD.y(); gene1 < ngenes; gene1 += gridSizeMD.y())
            {
                for (auto gene2 = gridThreadIndexMD.x(); gene2 < ngenes; gene2 += gridSizeMD.x())
                {

                    if (gene1 >= ngenes || gene2 >= ngenes || gene2 <= gene1){
                        return;
                    }

                    int n_first_low, n_first_high, n_second_high, n_second_low, n_total;
                    float all_statistic[4], all_pval[4];

                    int quadrant_counts[4];
                    getQuadrantCounts(gene1, gene2, expr_values, zero_flags, nsamples, quadrant_counts);

                    n_first_low = quadrant_counts[0] + quadrant_counts[1];
                    n_first_high = quadrant_counts[2] + quadrant_counts[3];
                    n_second_high = quadrant_counts[1] + quadrant_counts[3];
                    n_second_low = quadrant_counts[0] + quadrant_counts[2];

                    n_total = n_first_low + n_first_high;

                    for (char impl_type = 0; impl_type < 4; impl_type++){
                        float * statistic = all_statistic + impl_type;
                        float * pval = all_pval + impl_type;
                        getSingleImplication<T>(quadrant_counts, n_total, n_first_low, n_first_high, n_second_low, n_second_high, impl_type, statistic, pval);
                        if (*statistic >= statThresh && *pval <= pvalThresh){
                            int idx = alpaka::onAcc::atomicAdd(acc, impl_len.data(), (uint32_t)2);
                            assert(idx < MAX_N_IMP);
                            implications[idx] = {(int)gene1, (int)gene2, impl_type, *statistic, *pval};
                            implications[idx + 1] = {(int)gene2, (int)gene1, get_inverse_implication(impl_type), *statistic, *pval};
                        }
                    }
                    if (all_statistic[0] >= statThresh && all_pval[0] <= pvalThresh && all_statistic[3] >= statThresh && all_pval[3] <= pvalThresh){
                        int idx = alpaka::onAcc::atomicAdd(acc, symm_impl_len.data(), (uint32_t)2);
                        assert(idx < MAX_N_SYM_IMP);
                        symm_implications[idx] = {(int)gene1, (int)gene2, 4, all_statistic[0], all_statistic[3], all_pval[0], all_pval[3]};
                        symm_implications[idx + 1] = {(int)gene2, (int)gene1, 4, all_statistic[3], all_statistic[0], all_pval[3], all_pval[0]};
                    }
                    else if (all_statistic[1] >= statThresh && all_pval[1] <= pvalThresh && all_statistic[2] >= statThresh && all_pval[2] <= pvalThresh){
                        int idx = alpaka::onAcc::atomicAdd(acc, symm_impl_len.data(), (uint32_t)2);
                        assert(idx < MAX_N_SYM_IMP);
                        symm_implications[idx] = {(int)gene1, (int)gene2, 5, all_statistic[1], all_statistic[2], all_pval[1], all_pval[2]};
                        symm_implications[idx + 1] = {(int)gene2, (int)gene1, 5, all_statistic[2], all_statistic[1], all_pval[2], all_pval[1]};
                    }
                }
            }
        }
    };
};

// 4GB of memory divided by the size of a single implication
constexpr uint32_t MAX_N_IMP = (uint32_t)4e9 / (uint32_t)sizeof(impl);
constexpr uint32_t MAX_N_SYM_IMP = (uint32_t)1e9 / (uint32_t)sizeof(symm_impl);
