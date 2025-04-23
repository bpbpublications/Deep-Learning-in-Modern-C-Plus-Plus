![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 11
Underfitting, Overfitting, and Regularization 

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
chapter_11 $ mkdir build
chapter_11 $ cd build
chapter_11/build $ cmake ..
chapter_11/build $ make
```

### Running the experiments

The polybomial regression experiment can be executed by

```
chapter_11/build $ ./fitting_polynoms
```

The overtiffing experiment can be executed by

```
chapter_11/build $ ./overfitting_example_SOURCES_LIB
```

The third experiment can be executed by

```
chapter_11/build $ ./penalization
```

The fourth experiment can be executed by

```
chapter_11/build $ ./data_augmentation
```

The last experiment can be executed by

```
chapter_11/build $ ./dropout_example
```

### Dataset

This chapter uses the MNIST dataset. Check Chapter 10 for the intructions to download MNIST
