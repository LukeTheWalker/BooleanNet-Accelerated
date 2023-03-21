#include "BooleanNet.cuh"

__host__ void BooleanNet::get_all_implications(std::vector<std::string> genes, char* expr_values, int nsamples, float statThresh, float pvalThresh, float * implication_matrix){
#if 1
    int gene1, gene2;
    int n_first_low, n_first_high, n_second_high, n_second_low, n_total;
    float statistic, pval;
    int i = 0;
    for (gene1 = 0; gene1 < genes.size(); gene1++){
        for (gene2 = 0; gene2 < genes.size(); gene2++){
            if (gene1 != gene2){
                int quadrant_counts[4];
                getQuadrantCounts(gene1, gene2, expr_values, nsamples, quadrant_counts);

                // for (int i = 0; i < 4; i++){
                //     if (i == 2) printf("\n");
                //     printf("%d\t", quadrant_counts[i]);
                // }
                // printf("\n");

                n_first_low = quadrant_counts[0] + quadrant_counts[1];
                n_first_high = quadrant_counts[2] + quadrant_counts[3];
                n_second_high = quadrant_counts[1] + quadrant_counts[3];
                n_second_low = quadrant_counts[0] + quadrant_counts[2];

                n_total = n_first_low + n_first_high;

                for (char impl_type = 0; impl_type < 4; impl_type++){
                    getSingleImplication(quadrant_counts, n_total, n_first_low, n_first_high, n_second_low, n_second_high, impl_type, &statistic, &pval);
                    if (statistic > statThresh && pval < pvalThresh){
                        // implication_matrix[i] = gene1;
                        // implication_matrix[i+1] = gene2;
                        // implication_matrix[i+2] = impl_type;
                        // implication_matrix[i+3] = statistic;
                        // implication_matrix[i+4] = pval;
                        printf("%s\t%s\t%d\t%f\t%f\t\n", genes[gene1].c_str(), genes[gene2].c_str(), impl_type, statistic, pval);
                    }
                    i += 5;
                }
            }
        }
    }
#endif
}

__host__ __device__ void BooleanNet::getQuadrantCounts(int gene1, int gene2, char* expr_values, int nsamples, int* quadrant_counts){
    for (int i = 0; i < 4; i++){
        quadrant_counts[i] = 0;
    }
    // for (int i = 0; i < nsamples; i++){
    //     printf("%d\t", expr_values[gene1 * nsamples + i]);
    // }
    // printf("\n");
    // for (int i = 0; i < nsamples; i++){
    //     printf("%d\t", expr_values[gene2 * nsamples + i]);
    // }
    // printf("\n");
    for (int i = 0; i < nsamples; i++){
        if (expr_values[gene1 * nsamples + i] == -1){
            if (expr_values[gene2 * nsamples + i] == -1){
                quadrant_counts[0]++;
            }
            else if (expr_values[gene2 * nsamples + i] == 1){
                quadrant_counts[1]++;
            }
        }
        else if (expr_values[gene1 * nsamples + i] == 1){
            if (expr_values[gene2 * nsamples + i] == -1){
                quadrant_counts[2]++;
            }
            else if (expr_values[gene2 * nsamples + i] == 1){
                quadrant_counts[3]++;
            }
        }
    }
}

__host__ __device__ void BooleanNet::getSingleImplication(int* quadrant_counts, int n_total, int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type, float* statistic, float* pval){
    if (is_zero(n_first_low, n_first_high, n_second_low, n_second_high, impl_type)){
        *statistic = 0.0;
        *pval = 1.0;
        return;
    }

    if (impl_type == 0){
        double n_expected = (double)(n_first_low * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[1]) / sqrt(n_expected);
        *pval = ((((double)quadrant_counts[1] / n_first_low) + ((double)quadrant_counts[1] / n_second_high)) / 2);
    }
    else if (impl_type == 1){
        double n_expected = (double)(n_first_low * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[0]) / sqrt(n_expected);
        *pval = ((((double)quadrant_counts[0] / n_first_low) + ((double)quadrant_counts[0] / n_second_low)) / 2);
    }
    else if (impl_type == 2){
        double n_expected = (double)(n_first_high * n_second_high) / n_total;
        *statistic = (n_expected - quadrant_counts[3]) / sqrt(n_expected);
        *pval = ((((double)quadrant_counts[3] / n_first_high) + ((double)quadrant_counts[3] / n_second_high)) / 2);
    }
    else if (impl_type == 3){
        double n_expected = (double)(n_first_high * n_second_low) / n_total;
        *statistic = (n_expected - quadrant_counts[2]) / sqrt(n_expected);
        *pval = ((((double)quadrant_counts[2] / n_first_high) + ((double)quadrant_counts[2] / n_second_low)) / 2);
    }
}
__host__ __device__ char BooleanNet::is_zero(int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type){
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

__global__ void getImplication(char * expr_values, uint64_t ngenes, int nsamples, BooleanNet * net, float statThresh, float pvalThresh, uint32_t * impl_len, impl * d_implications, uint32_t * d_symm_impl_len, symm_impl * d_symm_implications){
    uint64_t gi = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;

    uint64_t gene1 = gi / ngenes;
    uint64_t gene2 = gi % ngenes;

    uint64_t nels = ngenes * ngenes;

    // if (gi % (nels / 100) == 0)
    //     printf("Processed %ld%% of %ld total genes, index: %ld\n", gi / (nels / 100), nels, gi);
    
    if (gene1 == gene2 || gi >= nels){
        return;
    }

    int n_first_low, n_first_high, n_second_high, n_second_low, n_total;
    float all_statistic[4], all_pval[4];

    int quadrant_counts[4];
    net->getQuadrantCounts(gene1, gene2, expr_values, nsamples, quadrant_counts);

    n_first_low = quadrant_counts[0] + quadrant_counts[1];
    n_first_high = quadrant_counts[2] + quadrant_counts[3];
    n_second_high = quadrant_counts[1] + quadrant_counts[3];
    n_second_low = quadrant_counts[0] + quadrant_counts[2];

    n_total = n_first_low + n_first_high;

    for (char impl_type = 0; impl_type < 4; impl_type++){
        float * statistic = all_statistic + impl_type;
        float * pval = all_pval + impl_type;
        net->getSingleImplication(quadrant_counts, n_total, n_first_low, n_first_high, n_second_low, n_second_high, impl_type, statistic, pval);
        if (*statistic >= statThresh && *pval <= pvalThresh){
            int idx = atomicAdd(impl_len, 1);
            d_implications[idx] = {(int)gene1, (int)gene2, impl_type, *statistic, *pval};
        }
    }
    if (all_statistic[0] >= statThresh && all_pval[0] <= pvalThresh && all_statistic[3] >= statThresh && all_pval[3] <= pvalThresh){
        int idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene1, (int)gene2, 4, all_statistic[0], all_statistic[3], all_pval[0], all_pval[3]};
    }
    else if (all_statistic[1] >= statThresh && all_pval[1] <= pvalThresh && all_statistic[2] >= statThresh && all_pval[2] <= pvalThresh){
        int idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene1, (int)gene2, 5, all_statistic[1], all_statistic[2], all_pval[1], all_pval[2]};
    }
}

