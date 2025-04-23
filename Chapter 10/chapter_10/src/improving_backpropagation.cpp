#include <iostream>
#include <chrono>
using namespace std::chrono;

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
        bool use_bias = false;

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

template <typename MODEL>
void training(MODEL &model, const Tensor_2D &training_images, const Tensor_2D &training_labels, 
              const int MAX_EPOCHS, float learning_rate) {

    CategoricalCrossEntropy cost_fn;

    int epoch = 0;

    while (epoch < MAX_EPOCHS)
    {

        auto begin = high_resolution_clock::now();

        model.forward(training_images);
        model.backward(training_labels);
        model.update(learning_rate, epoch);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto training_pred = model.predict(training_images);
        float training_loss = cost_fn.evaluate(training_labels, training_pred);
        float training_acc = accuracy(training_labels, training_pred);

        std::cout 
                << "epoch:\t" << epoch << "\t"
                << "took:\t" << duration.count() << " mills\t"
                << "\ttraining_loss:\t" << training_loss
                << "\ttraining_acc:\t" << training_acc
                << "\n";

        epoch++;

    }

}

int main(int, char **)
{

    auto [training_images, training_labels] = load_mnist("../../data/mnist", rng);
    std::cout << "Data loaded!\n";

    std::cout << "training_images dims: " << training_images.dimensions() << "\n";
    std::cout << "training_labels dims: " << training_labels.dimensions() << "\n";

    const int MAX_EPOCHS = 20;
    const float learning_rate = 0.1f;

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    Eigen::ThreadPoolDevice device(&tp, threads);

    Model model(device);
    training(model, training_images, training_labels, MAX_EPOCHS, learning_rate);

    return 0;
}