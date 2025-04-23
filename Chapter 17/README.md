![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 17
Developing an Object Localization System

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
chapter_17 $ mkdir build
chapter_17 $ cd build
chapter_17/build $ cmake ..
chapter_17/build $ make
```

### Running the experiments

The experiment 1 can be executed by invoking:

```
chapter_17/build $ ./regression_model
```

The experiment 2 can be executed by:

```
chapter_17/build $ ./net_training_oxford_pets
```

The experiment 3 can be executed by:

```
chapter_17/build $ ./fine_tuning_vgg_net_training_oxford_pets
```
PS: Please do not forget to download and store the Oxford Pets III dataset before running the experiments

## Downloading & storing the data

The Oxford Pets III dataset is distributed in two files:

- images.tar.gz, at https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
- annotations.tar.gz, at https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

Download and save them anywhere on your system.

Under the directory `data/` (at same level of the `chapter_17` folder) create a folder `Oxford-IIIT_Pet_Dataset`.

Uncompress the contents of file images.tar.gz and file annotations.tar.gz into `data/Oxford-IIIT_Pet_Dataset`.

You end up with the following structure:

```
- chapter_17
- data
  - Oxford-IIIT_Pet_Dataset
    - annotations
      - trimaps
      - xmls
    - images
      - *.jpg files
```

For more information about the dataset, including licensing, check the dataset homepage at https://www.robots.ox.ac.uk/~vgg/data/pets/