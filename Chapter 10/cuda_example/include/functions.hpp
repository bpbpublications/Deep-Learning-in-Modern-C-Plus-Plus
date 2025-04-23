// see https://eigen.tuxfamily.org/dox/TopicCUDA.html
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

using Eigen::Tensor;

void printCudaVersion();

Tensor<float, 2> run_gpu_contraction(const Tensor<float, 2> &A, const Tensor<float, 2> &B);