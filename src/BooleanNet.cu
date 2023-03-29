#include "BooleanNet.cuh"

// NOTE: strict mapping of this function for impl_type in [0,3].
__device__ char get_inverse_implication(char impl_type){
    // Not using swith case because of problems with some gpu's and compiler, but still better than 4 if-else
    return (3 - impl_type) - (impl_type == 1) + (impl_type == 2);
}

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
    quadrant_counts[0] = quadrant_counts[1] = quadrant_counts[2] = quadrant_counts[3] = 0;
    int g1_ns = gene1 * nsamples;
    int g2_ns = gene2 * nsamples;
    for (int i = 0; i < nsamples; i++){
        bool k1p = g1_ns + i == 1;
        bool k1n = g1_ns + i == -1;
        bool k2p = g2_ns + i == 1;
        bool k2n = g2_ns + i == -1;
        quadrant_counts[0] += k2n && k1n;
        quadrant_counts[1] += k2p && k1n;
        quadrant_counts[2] += k2n && k1p;
        quadrant_counts[3] += k2p && k1p;
    }
}

__host__ __device__ void BooleanNet::getSingleImplication(int* quadrant_counts, int n_total, int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type, float* statistic, float* pval){
    if (is_zero(n_first_low, n_first_high, n_second_low, n_second_high, impl_type)){
        *statistic = 0.0;
        *pval = 1.0;
        return;
    }

    int n1 = (impl_type > 1) * n_first_high + (impl_type <= 1) * n_first_low;
    int n2 = (impl_type & 1) * n_second_low + (1 - (impl_type & 1)) * n_second_high;
    int np = n1 * n2;
    int q_index = impl_type ^ 1;
    double n_expected = (double)np / n_total;
    *statistic = (n_expected - quadrant_counts[q_index] / sqrt(n_expected));
    *pval = ((double)(n1 + n2)) / (2 * np) * quadrant_counts[q_index];

}
__host__ __device__ char BooleanNet::is_zero(int n_first_low, int n_first_high, int n_second_low, int n_second_high, char impl_type){
#ifdef DEBUG
    if (impl_type & 0xfffffffc) printf("Invalid impl_type in is_zero\n");
#endif

    // For an explanation of the below method, see also:
    // https://en.wikipedia.org/wiki/Karnaugh_map
    // https://en.wikipedia.org/wiki/Quine%E2%80%93McCluskey_algorithm

    bool ih = impl_type & 2;        // a
    bool il = impl_type & 1;        // b
    bool n1l = n_first_low > 0;     // c
    bool n1h = n_first_high > 0;    // d
    bool n2l = n_second_low > 0;    // e
    bool n2h = n_second_high > 0;   // f

    // SOP form
    // Calculated at: http://www.32x8.com/qmm6_____A-B-C-D-E-F_____m_0-1-2-3-4-5-6-7-8-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-27-28-29-30-31-32-33-34-35-36-38-39-40-41-42-43-44-45-46-47-48-49-50-51-52-53-55-56-57-58-59-60-61-62-63___________option-4_____899788965371824592779
    return  (!ih && !n1l) || (!il && !n2h) || (n2l && n2h) ||
            (n1l && n1h)  || (il && !n2h)  || (ih && !n1h);
}

__global__ void getImplication(char * expr_values, uint64_t ngenes, int nsamples, BooleanNet * net, float statThresh, float pvalThresh, uint32_t * impl_len, impl * d_implications, uint32_t * d_symm_impl_len, symm_impl * d_symm_implications){
    uint64_t gi = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + (uint64_t) threadIdx.x;

    // uint64_t gene1 = gi / ngenes;
    // uint64_t gene2 = gi % ngenes;

    uint64_t gene1 = ngenes - 2 - floor(sqrt((double)-8*gi + 4*ngenes*(ngenes-1)-7)/2.0 - 0.5);
    uint64_t gene2 = gi + gene1 + 1 - ngenes*(ngenes-1)/2 + (ngenes-gene1)*((ngenes-gene1)-1)/2;

    uint64_t nels = (ngenes * (ngenes - 1)) / 2;

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
            idx = atomicAdd(impl_len, 1);
            d_implications[idx] = {(int)gene2, (int)gene1, get_inverse_implication(impl_type), *statistic, *pval};
        }
    }
    if (all_statistic[0] >= statThresh && all_pval[0] <= pvalThresh && all_statistic[3] >= statThresh && all_pval[3] <= pvalThresh){
        int idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene1, (int)gene2, 4, all_statistic[0], all_statistic[3], all_pval[0], all_pval[3]};
        idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene2, (int)gene1, 4, all_statistic[3], all_statistic[0], all_pval[3], all_pval[0]};
    }
    else if (all_statistic[1] >= statThresh && all_pval[1] <= pvalThresh && all_statistic[2] >= statThresh && all_pval[2] <= pvalThresh){
        int idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene1, (int)gene2, 5, all_statistic[1], all_statistic[2], all_pval[1], all_pval[2]};
        idx = atomicAdd(d_symm_impl_len, 1);
        d_symm_implications[idx] = {(int)gene2, (int)gene1, 5, all_statistic[2], all_statistic[1], all_pval[2], all_pval[1]};
    }
}

