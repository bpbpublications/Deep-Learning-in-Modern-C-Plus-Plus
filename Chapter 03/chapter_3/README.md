![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 3
Testing Deep Learning Code

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- Eigen: the core C++ linear algebra library, extensively used over the text
- GoogleTest: library for conducting unit testing with C++

### Building the code

The first experiment can be built by the following commands
```
chapter_3 $ cd 1_googletest_setup
chapter_3/1_googletest_setup $ mkdir build
chapter_3/1_googletest_setup $ cd build
chapter_3/1_googletest_setup/build $ cmake ..
chapter_3/1_googletest_setup/build $ make
```

The second experiment can be built by the following commands
```
chapter_3 $ cd 2_stat_tests
chapter_3/2_stat_tests $ mkdir build
chapter_3/2_stat_tests $ cd build
chapter_3/2_stat_tests/build $ cmake ..
chapter_3/2_stat_tests/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_3/1_googletest_setup/build $ ./test_googletest_example
```

The second experiment can be executed by

```
chapter_3/2_stat_tests/build $ ./test_glorot_test_example
```