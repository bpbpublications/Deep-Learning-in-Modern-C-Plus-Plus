#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std::chrono;

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/book.hpp"

using DEVICE = Eigen::ThreadPoolDevice;
using GEN = std::mt19937;

std::random_device rd{};
const auto seed = rd();
GEN rng(seed);

template <typename Device>
class Model {

public:

    Model(Device &device) {
        const int x_width = 28;
        const int x_height = 28;

        const int kernel_size = 3;
        const int filters = 32;
        const int channels = 1;

        const int hidden_neurons = 128;
        const int num_classes = 10;

        this->input_layer = new ReshapeLayer<Device, 2, 4>(device, std::vector<Eigen::Index>{x_width, x_height, 1});

        Tensor_4D kernels(kernel_size, kernel_size, channels, filters);
        glorot_uniform_initializer(rng, kernels);
        this->conv2d = new Conv2DLayer(device, kernels, use_bias, new ReLU<Device, 4>(device), std::vector<int>{0, 0, 0, 0});

        this->maxpooling = new MaxPooling(device, 2);

        this->dropout = new DropoutLayer<DEVICE, 4, GEN>(device, 0.2, rng);

        const int half_width = (x_width - kernel_size + 1) / 2;
        const int half_height = (x_height - kernel_size + 1) / 2;
        const int flatten_size = half_width * half_height * filters;

        this->flatten = new FlattenLayer<DEVICE, 4>(device);

        Tensor_2D WD(flatten_size, hidden_neurons);
        glorot_uniform_initializer(rng, WD);
        this->dense = new DenseLayer(device, WD, use_bias, new ReLU<Device, 2>(device));

        Tensor_2D WC(hidden_neurons, num_classes);
        glorot_uniform_initializer(rng, WC);
        this->output_layer = new SoftmaxCrossEntropyLayer(device, WC, use_bias);
    }

    void set_momentum() {
        conv2d->set_kernels_optimizer(Momentum<4>());
        dense->set_weight_optimizer(Momentum<2>());
        output_layer->set_weight_optimizer(Momentum<2>());

        if (use_bias) {
            conv2d->set_bias_optimizer(Momentum<1>());
            dense->set_bias_optimizer(Momentum<1>());
            output_layer->set_bias_optimizer(Momentum<1>());
        }
    }

    void set_rmsprop() {
        conv2d->set_kernels_optimizer(RMSProp<4>());
        dense->set_weight_optimizer(RMSProp<2>());
        output_layer->set_weight_optimizer(RMSProp<2>());

        if (use_bias) {
            conv2d->set_bias_optimizer(RMSProp<1>());
            dense->set_bias_optimizer(RMSProp<1>());
            output_layer->set_bias_optimizer(RMSProp<1>());
        }
    }

    void set_adam() {
        conv2d->set_kernels_optimizer(Adam<4>());
        dense->set_weight_optimizer(Adam<2>());
        output_layer->set_weight_optimizer(Adam<2>());

        if (use_bias) {
            conv2d->set_bias_optimizer(Adam<1>());
            dense->set_bias_optimizer(Adam<1>());
            output_layer->set_bias_optimizer(Adam<1>());
        }
    }

    ~Model() {
        delete input_layer;
        delete conv2d;
        delete maxpooling;
        delete flatten;
        delete dense;
        delete dropout;
        delete output_layer;
    }

    void forward(const Eigen::Tensor<float, 2> &input) {
        this->input_layer->forward(input); // reshape (batch, 784) => (batch, 28, 28, 1)
        this->conv2d->forward(this->input_layer->get_output()); // conv 2d (batch, 28, 28, 1) => (batch, 26, 26, 32)
        this->maxpooling->forward(this->conv2d->get_output()); // max pooling (batch, 26, 26, 32) => (batch, 13, 13, 32)
        this->dropout->forward(this->maxpooling->get_output()); // dropout
        this->flatten->forward(this->dropout->get_output()); // flatten (13,13,32) => (batch, 5408)
        this->dense->forward(this->flatten->get_output()); // dense
        this->output_layer->forward(this->dense->get_output()); // softmax
    }

    void backward(const Eigen::Tensor<float, 2> &upstream) {
        this->output_layer->backward(upstream, true);
        this->dense->backward(output_layer->get_downstream(), true);
        this->flatten->backward(this->dense->get_downstream(), true);
        this->dropout->backward(flatten->get_downstream(), true);
        this->maxpooling->backward(this->dropout->get_downstream(), true);
        this->conv2d->backward(this->maxpooling->get_downstream(), false);
    }

    void update(const float learning_rate, int epoch) {
        this->conv2d->update(learning_rate, epoch);
        this->dense->update(learning_rate, epoch);
        this->output_layer->update(learning_rate, epoch);
    }

    Eigen::Tensor<float, 2> predict(const Eigen::Tensor<float, 2> &input) {
        auto y0 = this->input_layer->predict(input);
        auto y1 = this->conv2d->predict(y0);
        auto y2 = this->maxpooling->predict(y1);
        auto y3 = this->dropout->predict(y2);
        auto y4 = this->flatten->predict(y3);
        auto y5 = this->dense->predict(y4);
        auto result = this->output_layer->predict(y5);
        return std::move(result);
    }
    
private:
    ReshapeLayer<DEVICE, 2, 4> *input_layer;
    Conv2DLayer<Device> *conv2d;
    MaxPooling<DEVICE> *maxpooling;
    DropoutLayer<DEVICE, 4, GEN> *dropout;
    FlattenLayer<DEVICE, 4> *flatten;
    DenseLayer<Device> *dense;
    SoftmaxCrossEntropyLayer<Device> *output_layer;
    const bool use_bias = true;

};

template <typename Device, typename MODEL>
void training(MODEL &model, const Tensor_2D &training_images, const Tensor_2D &training_labels, 
              const Tensor_2D &validation_images, const Tensor_2D &validation_labels, Device &device, 
              const int MAX_EPOCHS, const int minibatch_size, const float learning_rate) {

    CategoricalCrossEntropy cost_fn;

    int epoch = 0;
    while (epoch < MAX_EPOCHS) {

        auto begin = high_resolution_clock::now();

        int steps = 0;

        Batches batches(rng, minibatch_size, &training_images, &training_labels);

        Batch<2, 2>* batch = batches.next();

        while (batch) {

            model.forward(batch->X);
            model.backward(batch->T);
            model.update(learning_rate, epoch + 1);

            batch = batches.next();
            steps++;
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto validation_pred = model.predict(validation_images);
        float validation_acc = accuracy(validation_labels, validation_pred);

        float validation_loss = cost_fn.evaluate(validation_labels, validation_pred);

        std::cout 
                // << "epoch:" << "\t"
                // << epoch << "\t"
                // << "\tvalidation_loss:" << "\t"
                // << validation_loss
                // << "\tvalidation_acc:" << "\t"
                << validation_acc
                << "\n";
        
        epoch++;

    }

}

int main(int, char **)
{
    
    auto [training_images, training_labels, validation_images, validation_labels] = load_mnist("../../data/mnist");
    std::cout << "Data loaded!\n";
    std::cout << "training_images: " << training_images.dimensions() << "\n";
    std::cout << "training_labels: " << training_labels.dimensions() << "\n";
    std::cout << "validation_images: " << validation_images.dimensions() << "\n";
    std::cout << "validation_labels: " << validation_labels.dimensions() << "\n\n";

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    Eigen::ThreadPoolDevice device(&tp, threads);

    std::cout << std::setprecision(4);

    const int MAX_EPOCHS = 10;

    Model model(device);
    model.set_momentum();

    training(model, training_images, training_labels, validation_images, validation_labels, device, MAX_EPOCHS, 32, 0.001);

    

    // Generating registers for the experiment

    // Traning 5 times using Momentum for different mini-batch sizes

    // std::vector<int> batch_sizes {16, 512, 4096};

    // for (int i = 0; i < 5; ++i) {

    //     std::cout << "\n=== Run #" << i << "\n\n";

    //     for (int minibatch_size : batch_sizes) {

    //         Model model(device);
    //         model.set_momentum();
    //         // model.set_rmsprop();
    //         // model.set_adam();

    //         std::cout << "minibatch_size: " << minibatch_size << "\n";
    //         training(model, training_images, training_labels, validation_images, validation_labels, device, MAX_EPOCHS, minibatch_size, 0.001);
    //     }
        
    // }



    // Training five times using Momentum for different learning rates

    // std::vector<float> learning_rates {0.01, 0.001, 0.0001 };

    // for (int i = 0; i < 5; ++i) {

    //     std::cout << "\n=== Run #" << i << "\n\n";

    //     for (float learning_rate : learning_rates) {

    //         Model model(device);
    //         model.set_momentum();
    //         // model.set_rmsprop();
    //         // model.set_adam();

    //         std::cout << "learning_rate: " << learning_rate << "\n";
    //         training(model, training_images, training_labels, validation_images, validation_labels, device, MAX_EPOCHS, 32, learning_rate);
    //     }
        
    // }

    return 0;
}