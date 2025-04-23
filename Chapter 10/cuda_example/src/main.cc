#include <iostream>
#include <chrono>

using namespace std::chrono;

#include "functions.hpp"

Tensor<float, 2> run_cpu_contraction(const Tensor<float, 2> &A, const Tensor<float, 2> &B)
{
    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
    Tensor<float, 2> expected = A.contract(B, dims);
    return expected;
}

int main()
{
    srand((unsigned int) time(0));

    printCudaVersion();

    const int S = 256;

    Tensor<float, 2> A(S, S);
    Tensor<float, 2> B(S, S);
    A.setRandom();
    B.setRandom();

    auto expected = run_cpu_contraction(A, B);

    auto tensor = run_gpu_contraction(A, B);

    Tensor<float, 2> diff = (expected - tensor).abs();

    Tensor<float, 0> max_diff = diff.maximum();
    Tensor<float, 0> mean_diff = diff.mean();

    std::cout << "\nmax_diff is " << max_diff << "\n\n";
    std::cout << "mean_diff is " << mean_diff << "\n\n";

    auto start_cpu = high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        run_cpu_contraction(A, B);
    }

    auto stop_cpu = high_resolution_clock::now();

    auto duration_cpu = duration_cast<milliseconds>(stop_cpu - start_cpu);

    std::cout << "Tensor size is (" << S << "x" << S << ")\n\n";
    std::cout << "Running on CPU took " << duration_cpu.count() << " milliseconds.\n";

    auto start_cuda = high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        run_gpu_contraction(A, B);
    }

    auto stop_cuda = high_resolution_clock::now();

    auto duration_cuda = duration_cast<milliseconds>(stop_cuda - start_cuda);

    std::cout << "Running on GPU took " << duration_cuda.count() << " milliseconds.\n";

    return 0;
}