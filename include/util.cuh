#ifndef UTILITY_H
#define UTILITY_H

void cuda_err_check(cudaError_t err, const char *file, int line);
std::string get_impl_string(int impl);

#endif