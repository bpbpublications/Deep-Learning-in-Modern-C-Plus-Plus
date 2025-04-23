![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 10
Coding the Backpropagation Algorithm

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- OpenCV: used to load images and apply image transformations (blur, rotations, zoom, etc)
- Eigen: the core C++ linear algebra library, extensively used over the text
- TBB: The multi-thread API used to speed up the execution of algorithms

Instructions for installing these dependencies can be found on the root page of this repository.

The CUDA example requires:
- a computer with a CUDA-compatible GPU
- GPU NVidia card drives
- CUDA toolkit

Check the instructions in [this link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install the CUDA tookit on linux.

### Building the code

The experiments 1 & 3 can be built by the following commands
```
chapter_10 $ mkdir build
chapter_10 $ cd build
chapter_10/build $ cmake ..
chapter_10/build $ make
```

The CUDA experiment is built by:
```
chapter_10 $ cd cuda_example
chapter_10/cuda_example $ mkdir build
chapter_10/cuda_example $ cd build
chapter_10/cuda_example/build $ cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
chapter_10/cuda_example/build $ make
```

PS.: Depending on the version of the CUDA tookit, you may need to run the following commands BEFORE invoking cmake:

```
$ export CUDA_HOME=/usr/local/cuda
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
$ export PATH=$PATH:$CUDA_HOME/bin
```

### Running the experiments

The first experiment can be executed by

```
chapter_10/build $ ./naive_backpropagation
```

The second experiment can be executed by

```
chapter_10/build $ ./improving_backpropagation
```

The CUDA experiment can be executed by

```
chapter_10/cuda_example/build $ ./cuda_helloworld
```

### Dataset

This chapter uses the MNIST dataset. The MNIST dataset is provided as four files files:

- [Training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- [Training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
- [Testing images](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
- [Testing labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

Under the directory `data/` (at same level of the `chapter_10` folder) create a folder `mnist`.

Download the files anywhere and uncompress their contents into the mnist directory.

You end up with the following structure:

```
- chapter_10
- data
  - mnist
    - t10k-images.idx3-ubyte
    - t10k-labels.idx1-ubyte
    - train-images.idx3-ubyte
    - train-labels.idx1-ubyte
```

More detail can be found in [this link](https://github.com/cvdfoundation/mnist).