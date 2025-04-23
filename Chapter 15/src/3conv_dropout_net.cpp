#include <iostream>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "training_dogs_vs_cats.hpp"

template <typename DEVICE, typename GEN>
class Model {

public:

    Model(DEVICE &device, GEN &rng, int img_size): IMG_SIZE(img_size) {

        const int kernel_size = 3;
        const int channels = 3;

        const int hidden_neurons = 512;
        const int num_classes = 2;
        bool use_bias = true;

        Tensor_4D kernels_0(kernel_size, kernel_size, channels, 16);
        glorot_uniform_initializer(rng, kernels_0);
        this->conv2d_0 = new Conv2DLayer(device, kernels_0, use_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});
        this->conv2d_0->set_kernels_optimizer(Adam<4>());
        this->conv2d_0->set_bias_optimizer(Adam<1>());

        this->maxpooling_0 = new MaxPooling(device, 2);

        this->dropout_0 = new DropoutLayer<DEVICE, 4, GEN>(device, 0.2, rng);

        Tensor_4D kernels_1(kernel_size, kernel_size, kernels_0.dimension(3), 32);
        glorot_uniform_initializer(rng, kernels_1);
        this->conv2d_1 = new Conv2DLayer(device, kernels_1, use_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});
        this->conv2d_1->set_kernels_optimizer(Adam<4>());
        this->conv2d_1->set_bias_optimizer(Adam<1>());

        this->maxpooling_1 = new MaxPooling(device, 2);

        this->dropout_1 = new DropoutLayer<DEVICE, 4, GEN>(device, 0.2, rng);

        Tensor_4D kernels_2(kernel_size, kernel_size, kernels_1.dimension(3), 64);
        glorot_uniform_initializer(rng, kernels_2);
        this->conv2d_2 = new Conv2DLayer(device, kernels_2, use_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});
        this->conv2d_2->set_kernels_optimizer(Adam<4>());
        this->conv2d_2->set_bias_optimizer(Adam<1>());

        this->maxpooling_2 = new MaxPooling(device, 2);

        this->dropout_2 = new DropoutLayer<DEVICE, 4, GEN>(device, 0.2, rng);

        this->flatten = new FlattenLayer<DEVICE, 4>(device);

        int reduced_image_size = IMG_SIZE / 8;

        const int flatten_size = reduced_image_size * reduced_image_size * kernels_2.dimension(3);

        Tensor_2D W1(flatten_size, hidden_neurons);
        glorot_uniform_initializer(rng, W1);
        this->dense = new DenseLayer(device, W1, use_bias, new ReLU<DEVICE, 2>(device));
        this->dense->set_weight_optimizer(Adam<2>());
        this->dense->set_bias_optimizer(Adam<1>());

        this->dropout = new DropoutLayer<DEVICE, 2, GEN>(device, 0.8, rng);

        Tensor_2D W2(hidden_neurons, num_classes);
        glorot_uniform_initializer(rng, W2);
        this->output_layer = new SoftmaxCrossEntropyLayer(device, W2, use_bias);
        this->output_layer->set_weight_optimizer(Adam<2>());
        this->output_layer->set_bias_optimizer(Adam<1>());
    }

    ~Model() {
        delete conv2d_0;
        delete maxpooling_0;
        delete dropout_0;

        delete conv2d_1;
        delete maxpooling_1;
        delete dropout_1;

        delete conv2d_2;
        delete maxpooling_2;
        delete dropout_2;

        delete flatten;
        delete dense;
        delete dropout;
        delete output_layer;
    }

    void forward(const Tensor_4D &input) {

        this->conv2d_0->forward(input);
        this->maxpooling_0->forward(this->conv2d_0->get_output()); 
        this->dropout_0->forward(this->maxpooling_0->get_output());

        this->conv2d_1->forward(this->dropout_0->get_output());
        this->maxpooling_1->forward(this->conv2d_1->get_output()); 
        this->dropout_1->forward(this->maxpooling_1->get_output());

        this->conv2d_2->forward(this->dropout_1->get_output());
        this->maxpooling_2->forward(this->conv2d_2->get_output()); 
        this->dropout_2->forward(this->maxpooling_2->get_output());

        this->flatten->forward(this->dropout_2->get_output());
        this->dense->forward(this->flatten->get_output()); 
        this->dropout->forward(this->dense->get_output());
        this->output_layer->forward(this->dropout->get_output()); 

    }

    void backward(const Tensor_2D &upstream) {
        this->output_layer->backward(upstream, true);
        this->dropout->backward(this->output_layer->get_downstream(), true);
        this->dense->backward(this->dropout->get_downstream(), true);
        this->flatten->backward(this->dense->get_downstream(), true);

        this->dropout_2->backward(this->flatten->get_downstream(), true);
        this->maxpooling_2->backward(this->dropout_2->get_downstream(), true);
        this->conv2d_2->backward(this->maxpooling_2->get_downstream(), true);

        this->dropout_1->backward(this->conv2d_2->get_downstream(), true);
        this->maxpooling_1->backward(this->dropout_1->get_downstream(), true);
        this->conv2d_1->backward(this->maxpooling_1->get_downstream(), true);

        this->dropout_0->backward(this->conv2d_1->get_downstream(), true);
        this->maxpooling_0->backward(this->dropout_0->get_downstream(), true);
        this->conv2d_0->backward(this->maxpooling_0->get_downstream(), false);

    }

    void update(const TYPE learning_rate, int epoch) {
        this->conv2d_0->update(learning_rate, epoch);
        this->conv2d_1->update(learning_rate, epoch);
        this->conv2d_2->update(learning_rate, epoch);
        this->dense->update(learning_rate, epoch);
        this->output_layer->update(learning_rate, epoch);
    }

    // On predict do not use dropout
    Tensor_2D predict(const Tensor_4D &input) {
        auto y1 = this->conv2d_0->predict(input);
        auto y2 = this->maxpooling_0->predict(y1);

        auto y3 = this->conv2d_1->predict(y2);
        auto y4 = this->maxpooling_1->predict(y3);

        auto y5 = this->conv2d_2->predict(y4);
        auto y6 = this->maxpooling_2->predict(y5);

        auto y7 = this->flatten->predict(y6);
        auto y8 = this->dense->predict(y7);
        auto result = this->output_layer->predict(y8);
        return std::move(result);
    }

    const Tensor_2D get_output() const {
        return output_layer->get_output();
    }
    
    const int get_IMG_SIZE () const {
        return this->IMG_SIZE;
    }

    
private:
    const int IMG_SIZE;
    Conv2DLayer<DEVICE> *conv2d_0;
    MaxPooling<DEVICE> *maxpooling_0;
    DropoutLayer<DEVICE, 4, GEN> *dropout_0;

    Conv2DLayer<DEVICE> *conv2d_1;
    MaxPooling<DEVICE> *maxpooling_1;
    DropoutLayer<DEVICE, 4, GEN> *dropout_1;

    Conv2DLayer<DEVICE> *conv2d_2;
    MaxPooling<DEVICE> *maxpooling_2;
    DropoutLayer<DEVICE, 4, GEN> *dropout_2;

    FlattenLayer<DEVICE, 4> *flatten;
    DenseLayer<DEVICE> *dense;
    DropoutLayer<DEVICE, 2, GEN> *dropout;
    DenseLayer<DEVICE> *output_layer;

};

int main(int, char **)
{

    using DEVICE = Eigen::ThreadPoolDevice;
    using GEN = std::mt19937;
    using MODEL = Model<DEVICE, GEN>;

    std::random_device rd{};
    const auto seed = rd(); 
    GEN rng(seed);

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    DEVICE device(&tp, threads);

    const int IMAGE_SIZE = 224;

    MODEL model(device, rng, IMAGE_SIZE);

    run_experiment<DEVICE, GEN, MODEL, ParallelBatches<GEN>>(
        "../../data/dogs_x_cats/PetImages",
        model, device, rng, 
        25, // epochs
        32, // minibatch_size
        0.001, // learning_rate
        false // square_images
    );

    std::cout << "success\n";

    return 0;
}