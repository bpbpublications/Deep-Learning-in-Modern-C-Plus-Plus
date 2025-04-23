#include <iostream>

#include "functions.hpp"

void printCudaVersion()
{
    std::cout
        << "__CUDACC_VER_MAJOR__: " << __CUDACC_VER_MAJOR__ << "\n"
        << "__CUDACC_VER_MINOR__" << __CUDACC_VER_MINOR__ << "\n"
        << "__CUDACC_VER_BUILD__: " << __CUDACC_VER_BUILD__ << "\n\n";

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << "\n";

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << "\n";
}

Tensor<float, 2> run_gpu_contraction(const Tensor<float, 2> &A, const Tensor<float, 2> &B)
{
    const int M = A.dimension(0);
    const int K = A.dimension(1); // for production, assert K == B.dimension(0)
    const int N = B.dimension(1);

    // allocating data on GPU

    std::size_t A_bytes = M * K * sizeof(float);
    std::size_t B_bytes = K * N * sizeof(float);

    float *A_data;
    float *B_data;

    gpuMalloc((void **)(&A_data), A_bytes);
    gpuMalloc((void **)(&B_data), B_bytes);

    gpuMemcpy(A_data, A.data(), A_bytes, gpuMemcpyHostToDevice);
    gpuMemcpy(B_data, B.data(), B_bytes, gpuMemcpyHostToDevice);

    auto M_array = Eigen::array<int, 2>{M, K};
    auto K_array = Eigen::array<int, 2>{K, N};
    Eigen::TensorMap<Eigen::Tensor<float, 2>> A_on_GPU(A_data, M_array);
    Eigen::TensorMap<Eigen::Tensor<float, 2>> B_on_GPU(B_data, K_array);

    std::size_t result_bytes = M * N * sizeof(float);
    float *result_data;
    gpuMalloc((void **)(&result_data), result_bytes);
    auto n_array = Eigen::array<int, 2>{M, N};
    Eigen::TensorMap<Eigen::Tensor<float, 2>> gpu_result(result_data, n_array);

    // running contraction on GPU
    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);
    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
    gpu_result.device(gpu_device) = A_on_GPU.contract(B_on_GPU, dims);

    // copying the result data to CPU
    Tensor<float, 2> result(M, N);
    gpuMemcpy(result.data(), result_data, result_bytes, gpuMemcpyDeviceToHost);

    // freed GPU memory
    gpuFree((void *)A_data);
    gpuFree((void *)B_data);
    gpuFree((void *)result_data);

    return result;
}
