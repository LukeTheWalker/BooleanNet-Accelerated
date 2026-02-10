#include "BooleanNetHIP.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ inline char get_inverse_implication(char impl_type){
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

__device__ inline void getQuadrantCounts(uint64_t gene1, uint64_t gene2, const uint64_t* expr_values, const uint64_t* zero_flags, uint64_t nsamples, int32_t* quadrant_counts){
    for (uint64_t i = 0; i < 4; i++){
        quadrant_counts[i] = 0;
    }
    const uint64_t nbits = 64;
    const uint64_t nslots = (nsamples + nbits - 1) / nbits;
    for (uint64_t i = 0; i < nslots; i++){
        const uint64_t gene1_slot = expr_values[gene1 * nslots + i];
        const uint64_t gene2_slot = expr_values[gene2 * nslots + i];
        const uint64_t zero_slot = zero_flags[gene1 * nslots + i] & zero_flags[gene2 * nslots + i];
        const uint64_t gene1_slot_low = ~gene1_slot & zero_slot;
        const uint64_t gene1_slot_high = gene1_slot & zero_slot;
        const uint64_t gene2_slot_low = ~gene2_slot & zero_slot;
        const uint64_t gene2_slot_high = gene2_slot & zero_slot;

        quadrant_counts[0] += __builtin_popcountll(gene1_slot_low & gene2_slot_low);
        quadrant_counts[1] += __builtin_popcountll(gene1_slot_low & gene2_slot_high);
        quadrant_counts[2] += __builtin_popcountll(gene1_slot_high & gene2_slot_low);
        quadrant_counts[3] += __builtin_popcountll(gene1_slot_high & gene2_slot_high);
    }
}

__device__ inline char is_zero(uint64_t n_first_low, uint64_t n_first_high, uint64_t n_second_low, uint64_t n_second_high, char impl_type){
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
    return 1;
}

__device__ inline void getSingleImplication(int32_t* quadrant_counts, uint64_t n_total, uint64_t n_first_low, uint64_t n_first_high, uint64_t n_second_low, uint64_t n_second_high, char impl_type, float* statistic, float* pval){
    if (is_zero(n_first_low, n_first_high, n_second_low, n_second_high, impl_type)){
        *statistic = 0.0;
        *pval = 1.0;
        return;
    }

    if (impl_type == 0){
        float n_expected = (float)(n_first_low * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[1]) / sqrtf(n_expected);
        *pval = ((((float)quadrant_counts[1] / n_first_low) + ((float)quadrant_counts[1] / n_second_high)) / 2);
    }
    else if (impl_type == 1){
        float n_expected = (float)(n_first_low * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[0]) / sqrtf(n_expected);
        *pval = ((((float)quadrant_counts[0] / n_first_low) + ((float)quadrant_counts[0] / n_second_low)) / 2);
    }
    else if (impl_type == 2){
        float n_expected = (float)(n_first_high * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[3]) / sqrtf(n_expected);
        *pval = ((((float)quadrant_counts[3] / n_first_high) + ((float)quadrant_counts[3] / n_second_high)) / 2);
    }
    else if (impl_type == 3){
        float n_expected = (float)(n_first_high * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[2]) / sqrtf(n_expected);
        *pval = ((((float)quadrant_counts[2] / n_first_high) + ((float)quadrant_counts[2] / n_second_low)) / 2);
    }
}

__global__ void getImplicationKernel(
    const uint64_t* expr_values, const uint64_t* zero_flags, 
    uint64_t ngenes, uint64_t nsamples, 
    float statThresh, float pvalThresh,
    uint64_t* impl_len, impl* implications, 
    uint64_t* symm_impl_len, symm_impl* symm_implications) {

    uint64_t gene1_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t gene2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint64_t stride_y = gridDim.y * blockDim.y;
    uint64_t stride_x = gridDim.x * blockDim.x;

    for (uint64_t gene1 = gene1_idx; gene1 < ngenes; gene1 += stride_y)
    {
        for (uint64_t gene2 = gene2_idx; gene2 < ngenes; gene2 += stride_x)
        {

            if (gene1 >= ngenes || gene2 >= ngenes || gene2 <= gene1){
                continue;
            }

            uint64_t n_first_low, n_first_high, n_second_high, n_second_low, n_total;
            float all_statistic[4], all_pval[4];

            int32_t quadrant_counts[4];
            getQuadrantCounts(gene1, gene2, expr_values, zero_flags, nsamples, quadrant_counts);

            n_first_low = quadrant_counts[0] + quadrant_counts[1];
            n_first_high = quadrant_counts[2] + quadrant_counts[3];
            n_second_high = quadrant_counts[1] + quadrant_counts[3];
            n_second_low = quadrant_counts[0] + quadrant_counts[2];

            n_total = n_first_low + n_first_high;

            for (char impl_type = 0; impl_type < 4; impl_type++){
                float * statistic = all_statistic + impl_type;
                float * pval = all_pval + impl_type;
                getSingleImplication(quadrant_counts, n_total, n_first_low, n_first_high, n_second_low, n_second_high, impl_type, statistic, pval);
                if (*statistic >= statThresh && *pval <= pvalThresh){
                    // atomicAdd returns the old value
                    unsigned long long int idx = atomicAdd((unsigned long long int*)impl_len, 2);
                    if (idx < MAX_N_IMP) {
                        implications[idx] = {gene1, gene2, impl_type, *statistic, *pval};
                        implications[idx + 1] = {gene2, gene1, get_inverse_implication(impl_type), *statistic, *pval};
                    }
                }
            }
            if (all_statistic[0] >= statThresh && all_pval[0] <= pvalThresh && all_statistic[3] >= statThresh && all_pval[3] <= pvalThresh){
                unsigned long long int idx = atomicAdd((unsigned long long int*)symm_impl_len, 2);
                if (idx < MAX_N_SYM_IMP) {
                    symm_implications[idx] = {gene1, gene2, 4, all_statistic[0], all_statistic[3], all_pval[0], all_pval[3]};
                    symm_implications[idx + 1] = {gene2, gene1, 4, all_statistic[3], all_statistic[0], all_pval[3], all_pval[0]};
                }
            }
            else if (all_statistic[1] >= statThresh && all_pval[1] <= pvalThresh && all_statistic[2] >= statThresh && all_pval[2] <= pvalThresh){
                unsigned long long int idx = atomicAdd((unsigned long long int*)symm_impl_len, 2);
                if (idx < MAX_N_SYM_IMP) {
                    symm_implications[idx] = {gene1, gene2, 5, all_statistic[1], all_statistic[2], all_pval[1], all_pval[2]};
                    symm_implications[idx + 1] = {gene2, gene1, 5, all_statistic[2], all_statistic[1], all_pval[2], all_pval[1]};
                }
            }
        }
    }
}

void launch_hip_kernel(
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
) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (ngenes + threads.x - 1) / threads.x,
        (ngenes + threads.y - 1) / threads.y
    );
    
    printf("Launching kernel with (%d, %d) blocks and (%d, %d) threads\n", blocks.x, blocks.y, threads.x, threads.y);
    
    hipLaunchKernelGGL(getImplicationKernel, blocks, threads, 0, 0, 
        d_expr_values, d_zero_flags, ngenes, nsamples, statThresh, pvalThresh,
        d_impl_len, d_implications, d_symm_impl_len, d_symm_implications
    );
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP Error: %s\n", hipGetErrorString(err));
    }
    hipDeviceSynchronize();
}