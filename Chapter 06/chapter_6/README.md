![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 6
Learning by Minimizing Cost Functions

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- Eigen: the core C++ linear algebra library, extensively used over the text

### Building the code

The experiments can be built by the following commands
```
chapter_6 $ mkdir build
chapter_6 $ cd build
chapter_6/build $ cmake ..
chapter_6/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_6/build $ ./h0_h1_example
```

The second experiment can be executed by

```
chapter_6/build $ ./fitting_example
```

The third experiment can be executed by

```
chapter_6/build $ ./using_gradient
```

The BCE experiment can be executed by

```
chapter_6/build $ ./bce_example
```