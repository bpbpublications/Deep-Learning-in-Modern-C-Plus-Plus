![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 9
Coding the Gradient Descent algorithm

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- OpenCV: used to load images and apply image transformations (blur, rotations, zoom, etc)
- Eigen: the core C++ linear algebra library, extensively used over the text

### Building the code

The experiments can be built by the following commands
```
chapter_9 $ mkdir build
chapter_9 $ cd build
chapter_9/build $ cmake ..
chapter_9/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_9/build $ ./gradient_descent_example
```

The second experiment can be executed by

```
chapter_9/build $ ./conv_grad_example
```

The autograd experiment can be executed by

```
chapter_9/cuda_example/build $ ./conv_autograd
```
