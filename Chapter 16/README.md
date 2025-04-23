![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 16
Leveraging Training Performance with Transfer Learning

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
chapter_16 $ mkdir build
chapter_16 $ cd build
chapter_16/build $ cmake ..
chapter_16/build $ make
```

### Running the experiments

The experiment 1 can be executed by invoking:

```
chapter_16/build $ ./training_tf_flowers
```

For experiment 2, you need to load the weights from the Dogs x Cats model.
You can run the modified version of the Dogs x Cats model training by:

```
chapter_16/build $ ./training_dogs_cats
```

The experiment 2 itself can be executed by:

```
chapter_16/build $ ./training_tf_flowers_with_transfer_learning
```

The experiment 3 can be executed by:

```
chapter_16/build $ ./training_tf_flowers_vgg
```
PS: Please do not forget to download and store the TensorFlow Flowers dataset before running the experiments

## Downloading & storing the data

The TensorFlow Flowers dataset is provided in just one file:

- flower_photos.tgz, at http://download.tensorflow.org/example_images/flower_photos.tgz

Download and save the file anywhere on your system.

Under the directory `data/` (at same level of the `chapter_16` folder) create a folder `flower_photos`.

Uncompress the contents of flower_photos.tgz into `data/flower_photos`.

You end up with following structure:

```
- chapter_16
- data
  - flower_photos
    - daisy
    - dandelion
    - roses
    - sunflowers
    - tulips
    - LICENSE.txt
```

For more information about the dataset, including licensing, check the dataset homepage at https://www.tensorflow.org/datasets/catalog/tf_flowers