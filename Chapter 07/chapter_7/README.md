![book image](../../media/book-image.png)

a ![bpb logo](../../media/bpb.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 7
Defining Activation Functions

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- Eigen: the core C++ linear algebra library, extensively used over the text

### Building the code

The experiments can be built by the following commands
```
chapter_7 $ mkdir build
chapter_7 $ cd build
chapter_7/build $ cmake ..
chapter_7/build $ make
```

### Running the experiments

The first experiment can be executed by

```
chapter_7/build $ ./using_softmax
```

The second experiment can be executed by

```
chapter_7/build $ ./batched_softmax
```

The Iris training can be executed by

```
chapter_7/build $ ./iris_example
```

## Downloading & storing the data

The Iris dataset is distributed in two files:

- iris.zip, at https://archive.ics.uci.edu/static/public/53/iris.zip

Download and save them anywhere on your system.

Uncompress the contents of file iris.zip and extract the file iris.data and save it in the directory `data` with name iris.csv

You end up with the following structure:

```
- chapter_7
- data
  - iris.csv
```

For more information about the dataset, including licensing, check the dataset homepage at https://archive.ics.uci.edu/dataset/53/iris