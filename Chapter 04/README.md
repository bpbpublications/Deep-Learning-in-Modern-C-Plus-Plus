![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 4
Implementing Convolutions

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- Eigen: the core C++ linear algebra library, extensively used over the text
- OpenCV: used to load images and apply image transformations (blur, rotations, zoom, etc)

### Building the code

The experiments can be built by the following commands
```
chapter_4 $ mkdir build
chapter_4 $ cd build
chapter_4/build $ cmake ..
chapter_4/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_4/build $ ./convolution_2d_example
```

The second experiment can be executed by

```
chapter_4/build $ ./same_padding_example
```

The OpenCV experiment can be executed by

```
chapter_4/build $ ./using_opencv
```

The Sobel experiment can be executed by

```
chapter_4/build $ ./applying_sobel
```

The tensor examples experiment can be executed by

```
chapter_4/build $ ./using_tensors
```

The tensor convolutions experiment can be executed by

```
chapter_4/build $ ./tensor_convolutions
```

The Strides experiment can be executed by

```
chapter_4/build $ ./using_strides
```

The padding using Eigen experiment can be executed by

```
chapter_4/build $ ./using_padding
```