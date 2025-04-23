#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std::chrono;

#include <iterator>
#include <cstddef>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/book.hpp"

std::random_device rd{};
const auto seed = rd();
std::mt19937 rng(seed);

template <typename Device>
class Model {

public:

    Model(Device &device) {

        const bool use_bias = false;

        Tensor_2D W1(28*28, 128);
        glorot_uniform_initializer(rng, W1);
        this->dense_1 = new DenseLayer(device, W1, use_bias, new ReLU<Device, 2>(device));

        Tensor_2D W2(128, 64);
        glorot_uniform_initializer(rng, W2);
        this->dense_2 = new DenseLayer(device, W2, use_bias, new ReLU<Device, 2>(device));

        Tensor_2D WC(64, 10);
        glorot_uniform_initializer(rng, WC);
        this->output_layer = new SoftmaxCrossEntropyLayer(device, WC, use_bias);
    }

    ~Model() {
        delete dense_1;
        delete dense_2;
        delete output_layer;
    }

    void forward(const Tensor_2D &input) {
        this->dense_1->forward(input);
        this->dense_2->forward(this->dense_1->get_output()); 
        this->output_layer->forward(this->dense_2->get_output());
    }

    void backward(const Tensor_2D &upstream) {
        this->output_layer->backward(upstream, true);
        this->dense_2->backward(output_layer->get_downstream(), true);
        this->dense_1->backward(this->dense_2->get_downstream(), false);
    }

    void update(const float learning_rate, int epoch) {
        this->dense_1->update(learning_rate, epoch);
        this->dense_2->update(learning_rate, epoch);
        this->output_layer->update(learning_rate, epoch);
    }

    Tensor_2D predict(const Tensor_2D &input) {
        auto y0 = this->dense_1->predict(input);
        auto y1 = this->dense_2->predict(y0);
        auto result = this->output_layer->predict(y1);
        return std::move(result);
    }

private:
    DenseLayer<Device> *dense_1;
    DenseLayer<Device> *dense_2;
    SoftmaxCrossEntropyLayer<Device> *output_layer;
};

template <typename MODEL, typename GEN>
void training(MODEL &model, GEN rng, const Tensor_2D &training_images, const Tensor_2D &training_labels, 
              const Tensor_2D &validation_images, const Tensor_2D &validation_labels,
              const int minibatch_size, bool verbose = false) {

    CategoricalCrossEntropy cost_fn;

    const int MAX_EPOCHS = 20;
    const float learning_rate = 0.1f;

    int epoch = 0;
    while (epoch < MAX_EPOCHS)
    {

        auto begin = high_resolution_clock::now();

        Batches batches(rng, minibatch_size, &training_images, &training_labels);

        Batch<2, 2> * batch = batches.next();

        while (batch) {

            model.forward(batch->X);
            model.backward(batch->T);
            model.update(learning_rate, epoch);

            batch = batches.next();
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto training_pred = model.predict(training_images);
        float training_loss = cost_fn.evaluate(training_labels, training_pred);
        float training_acc = accuracy(training_labels, training_pred);

        auto validation_pred = model.predict(validation_images);
        float validation_acc = accuracy(validation_labels, validation_pred);
        float validation_loss = cost_fn.evaluate(validation_labels, validation_pred);

        if (verbose) {
            std::cout 
                    << "epoch:\t" << epoch << "\t"
                    << "took:\t" << duration.count() << " mills\t"
                    << "\ttraining_loss:\t" << training_loss
                    << "\ttraining_acc:\t" << training_acc
                    << "\tvalidation_loss:\t" << validation_loss
                    << "\tvalidation_acc:\t" << validation_acc
                    << "\n";
        }
        
        epoch++;

    }

}

int main(int, char **)
{

    const auto [training_images, training_labels, validation_images, validation_labels] = load_mnist("../../data/mnist");
    std::cout << "Data loaded!\n";
    std::cout << "training_images: " << training_images.dimensions() << "\n";
    std::cout << "training_labels: " << training_labels.dimensions() << "\n";
    std::cout << "validation_images: " << validation_images.dimensions() << "\n";
    std::cout << "validation_labels: " << validation_labels.dimensions() << "\n\n";

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    Eigen::ThreadPoolDevice device(&tp, threads);

    std::cout << std::setprecision(4);

    Model model(device);
    int minibatch_size = 500;
    training(model, rng, training_images, training_labels, validation_images, validation_labels, minibatch_size, true);

    // std::vector<int> configurations {60'000, 10'000, 1'000, 100};

    // for (int minibatch_size : configurations) {
    //     std::cout << "Using mini-batch size: " << minibatch_size << "\n\n";
    //     Model model(device);
    //     training(model, rng, training_images, training_labels, validation_images, validation_labels, minibatch_size, true);
    //     std::cout << "\n\n";
    // }

    return 0;
}