![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 2
Coding Deep Learning with Modern C++

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- Eigen: the core C++ linear algebra library, extensively used over the text

### Building the code

The first experiment is compiled by invoking the compiler. If using gcc, the command to compile is:
```
chapter_2 $ cd 1_basic_cpp
g++ src/hello_world.cpp -o my_program
```

The second experiment can be built by the following commands
```
chapter_2 $ cd 1_basic_cpp
chapter_2/1_basic_cpp $ mkdir build
chapter_2/1_basic_cpp $ cd build
chapter_2/1_basic_cpp/build $ cmake ..
chapter_2/1_basic_cpp/build $ make
```

The third experiment can be built by the following commands
```
chapter_2 $ cd 2_numeric_functional_algorithms
chapter_2/2_numeric_functional_algorithms $ mkdir build
chapter_2/2_numeric_functional_algorithms $ cd build
chapter_2/2_numeric_functional_algorithms/build $ cmake ..
chapter_2/2_numeric_functional_algorithms/build $ make
```

The fourth experiment can be built by the following commands
```
chapter_2 $ cd 3_lambdas
chapter_2/3_lambdas $ mkdir build
chapter_2/3_lambdas $ cd build
chapter_2/3_lambdas/build $ cmake ..
chapter_2/3_lambdas/build $ make
```

The fifth experiment can be built by the following commands
```
chapter_2 $ cd 4_eigen_example
chapter_2/4_eigen_example $ mkdir build
chapter_2/4_eigen_example $ cd build
chapter_2/4_eigen_example/build $ cmake ..
chapter_2/4_eigen_example/build $ make
```

The sixth experiment can be built by the following commands
```
chapter_2 $ cd 5_gen_random_numbers
chapter_2/5_gen_random_numbers $ mkdir build
chapter_2/5_gen_random_numbers $ cd build
chapter_2/5_gen_random_numbers/build $ cmake ..
chapter_2/5_gen_random_numbers/build $ make
```

### Running the experiments

The second experiment can be executed by the following command:
```
chapter_2/1_basic_cpp $ ./build/bin/hello_world
```

The third experiment can be executed by the following commands:

```
chapter_2/2_numeric_functional_algorithms/build $ ./using_algorithms
```
```
chapter_2/2_numeric_functional_algorithms/build $ ./using_functional
```
and 
```
chapter_2/2_numeric_functional_algorithms/build $ ./using_numeric
```

The fourth experiment can be executed by the following command:

```
chapter_2/3_lambdas/build $ ./using_lambdas
```

The fifth experiment can be executed by the following command:

```
chapter_2/4_eigen_example/build $ ./eigen_premier
```

The sixth experiment can be executed by the following command:

```
chapter_2/5_gen_random_numbers/build $ ./stats_cpp
```