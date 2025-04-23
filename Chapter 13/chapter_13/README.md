![book image](../media/book-image.png)

a ![pbp logo](../media/pbp.png) publication

# Deep Learning in Modern C++
Implementing the foundational Deep Learning components in C++ and Eigen

# Chapter 13
Implementing Optimizers

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
chapter_13 $ mkdir build
chapter_13 $ cd build
chapter_13/build $ cmake ..
chapter_13/build $ make
```

### Running the experiment

The example is executed by:

```
chapter_13/build $ ./chapter_13
```
The experiment consists of running the program several times with different configurations.

In the first run, we comment the example lines and uncomment the lines to run the model training for 5 times:

```c++
// Model model(device);
// model.set_momentum();
// training(model, training_images, training_labels, validation_images, validation_labels, device, MAX_EPOCHS, 32, 0.001);

std::vector<int> batch_sizes {16, 512, 4096};

for (int i = 0; i < 5; ++i) {

    std::cout << "\n=== Run #" << i << "\n\n";

    for (int minibatch_size : batch_sizes) {

        Model model(device);
        model.set_momentum();
        // model.set_rmsprop();
        // model.set_adam();

        std::cout << "minibatch_size: " << minibatch_size << "\n";
        training(model, training_images, training_labels, validation_images, validation_labels, device, MAX_EPOCHS, minibatch_size, 0.001);
    }
    
}
```
Repeat the process selecting RMSProp:
```c++
// model.set_momentum();
model.set_rmsprop();
// model.set_adam();
```
Adam:
```c++
// model.set_momentum();
model.set_rmsprop();
// model.set_adam();
```
Comment them all to set no optmizer:
```c++
// model.set_momentum();
// model.set_rmsprop();
// model.set_adam();
```
### Dataset

This chapter uses the MNIST dataset. Check Chapter 10 for the intructions to download MNIST
