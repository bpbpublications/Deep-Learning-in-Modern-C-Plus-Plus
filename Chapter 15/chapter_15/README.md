![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 15
Developing an Image Classifier

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
chapter_15 $ mkdir build
chapter_15 $ cd build
chapter_15/build $ cmake ..
chapter_15/build $ make
```

### Running the experiments

The experiment 1 can be executed by invoking:

```
chapter_15/build $ ./3conv_net
```

The experiment 2 can be executed by:

```
chapter_15/build $ ./3conv_dropout_net
```

The experiment 3 can be executed by:

```
chapter_15/build $ ./3conv_data_augmentation
```
PS: This chapter uses the same dataset used in Chapter 14, the Dogs x Cats dataset.
Check the instructions on Chapter 14 to know how to download the Dogs x Cats dataset.
