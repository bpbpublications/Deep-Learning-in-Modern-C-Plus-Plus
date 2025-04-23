![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 14
Developing an Image Classifier

## Instructions to build & run the experiments

### Pre-requisites

The following packages are required:

- OpenCV: used to load images and apply image transformations (blur, rotations, zoom, etc)
- Eigen: the core C++ linear algebra library, extensively used over the text

PS: These are the same pre-requisites used in the previous chapters.

Instructions for installing these dependencies can be found on the root page of this repository.

### Building the code

```
chapter_14 $ mkdir build
chapter_14 $ cd build
chapter_14/build $ cmake ..
chapter_14/build $ make
```

### Running the experiments

The face detector can be executed by invoking:

```
chapter_14/build $ ./face_detection
```

The experiment can be executed by:

```
chapter_14/build $ ./chapter_14
```

PS: This chapter uses the Dogs x Cats dataset. Make sure you have downloaded the dataset before run the experiment

## Downloading & storing the data

The Dogs x Cats dataset is distributed in one file:

- kagglecatsanddogs_5340.zip, at https://www.microsoft.com/en-us/download/details.aspx?id=54765

Download and save the file anywhere on your system.

Under the directory `data/` (at same level of the `chapter_14` folder) create a folder `dogs_x_cats`.

Uncompress the contents of kagglecatsanddogs_5340.zip into `data/dogs_x_cats`.

You end up with the following structure:

```
- chapter_14
- data
  - dogs_x_cats
    - PetImages
    - CDLA-Permissive-2.0.pdf
    - readme[1].txt
```

For more information about the dataset, including licensing, check the dataset homepage at https://www.kaggle.com/c/dogs-vs-cats
