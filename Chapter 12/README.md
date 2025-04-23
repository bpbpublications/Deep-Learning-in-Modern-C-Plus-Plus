![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 12
Implementing Cross-validation, mini batching, and model performance metrics

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- OpenCV: used to load images and apply image transformations (blur, rotations, zoom, etc)
- Eigen: the core C++ linear algebra library, extensively used over the text
- TBB: The multi-thread API used to speed up the execution of algorithms

PS: These are the same pre-requisites used in the previous chapters.

Instructions for installing these dependencies can be found on the root page of this repository.

### Building the code

```
chapter_12 $ mkdir build
chapter_12 $ cd build
chapter_12/build $ cmake ..
chapter_12/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_12/build $ ./using_metrics
```

The second experiment can be executed by

```
chapter_12/build $ ./stratification
```

The third experiment can be executed by

```
chapter_12/build $ ./minibatch
```

The fourth experiment can be executed by

```
chapter_12/build $ ./sgd_example
```

### Dataset

This chapter uses the MNIST dataset. Check Chapter 10 for the intructions to download MNIST
