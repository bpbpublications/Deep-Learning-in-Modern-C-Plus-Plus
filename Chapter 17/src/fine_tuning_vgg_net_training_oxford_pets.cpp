#include <iostream>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "vgg_training_oxford_pets.hpp"

template <typename DEVICE, typename GEN>
class Model {

public:

    Model(DEVICE &device, GEN &rng, int img_size): IMG_SIZE(img_size) {

        const int kernel_size = 3;
        const int channels = 3;

        const int hidden_neurons = 512;
        const int num_classes = NUMBER_CLASSES;
        bool use_bias = true;

        Tensor_4D block1_conv1_kernel(kernel_size, kernel_size, 3, 64);
        load_tensor_from_file(block1_conv1_kernel, "../../data/vgg16/imagenet/block1_conv1_kernel.txt");
        Tensor_1D block1_conv1_bias(64);
        load_tensor_from_file(block1_conv1_bias, "../../data/vgg16/imagenet/block1_conv1_bias.txt");
        this->block1_conv1 = new Conv2DLayer(device, block1_conv1_kernel, block1_conv1_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block1_conv2_kernel(kernel_size, kernel_size, 64, 64);
        load_tensor_from_file(block1_conv2_kernel, "../../data/vgg16/imagenet/block1_conv2_kernel.txt");
        Tensor_1D block1_conv2_bias(64);
        load_tensor_from_file(block1_conv2_bias, "../../data/vgg16/imagenet/block1_conv2_bias.txt");
        this->block1_conv2 = new Conv2DLayer(device, block1_conv2_kernel, block1_conv2_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        this->block1_pool = new MaxPooling(device, 2);

        Tensor_4D block2_conv1_kernel(kernel_size, kernel_size, 64, 128);
        load_tensor_from_file(block2_conv1_kernel, "../../data/vgg16/imagenet/block2_conv1_kernel.txt");
        Tensor_1D block2_conv1_bias(128);
        load_tensor_from_file(block2_conv1_bias, "../../data/vgg16/imagenet/block2_conv1_bias.txt");
        this->block2_conv1 = new Conv2DLayer(device, block2_conv1_kernel, block2_conv1_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block2_conv2_kernel(kernel_size, kernel_size, 128, 128);
        load_tensor_from_file(block2_conv2_kernel, "../../data/vgg16/imagenet/block2_conv2_kernel.txt");
        Tensor_1D block2_conv2_bias(128);
        load_tensor_from_file(block2_conv2_bias, "../../data/vgg16/imagenet/block2_conv2_bias.txt");
        this->block2_conv2 = new Conv2DLayer(device, block2_conv2_kernel, block2_conv2_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        this->block2_pool = new MaxPooling(device, 2);

        Tensor_4D block3_conv1_kernel(kernel_size, kernel_size, 128, 256);
        load_tensor_from_file(block3_conv1_kernel, "../../data/vgg16/imagenet/block3_conv1_kernel.txt");
        Tensor_1D block3_conv1_bias(256);
        load_tensor_from_file(block3_conv1_bias, "../../data/vgg16/imagenet/block3_conv1_bias.txt");
        this->block3_conv1 = new Conv2DLayer(device, block3_conv1_kernel, block3_conv1_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block3_conv2_kernel(kernel_size, kernel_size, 256, 256);
        load_tensor_from_file(block3_conv2_kernel, "../../data/vgg16/imagenet/block3_conv2_kernel.txt");
        Tensor_1D block3_conv2_bias(256);
        load_tensor_from_file(block3_conv2_bias, "../../data/vgg16/imagenet/block3_conv2_bias.txt");
        this->block3_conv2 = new Conv2DLayer(device, block3_conv2_kernel, block3_conv2_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block3_conv3_kernel(kernel_size, kernel_size, 256, 256);
        load_tensor_from_file(block3_conv3_kernel, "../../data/vgg16/imagenet/block3_conv3_kernel.txt");
        Tensor_1D block3_conv3_bias(256);
        load_tensor_from_file(block3_conv3_bias, "../../data/vgg16/imagenet/block3_conv3_bias.txt");
        this->block3_conv3 = new Conv2DLayer(device, block3_conv3_kernel, block3_conv3_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        this->block3_pool = new MaxPooling(device, 2);

        Tensor_4D block4_conv1_kernel(kernel_size, kernel_size, 256, 512);
        load_tensor_from_file(block4_conv1_kernel, "../../data/vgg16/imagenet/block4_conv1_kernel.txt");
        Tensor_1D block4_conv1_bias(512);
        load_tensor_from_file(block4_conv1_bias, "../../data/vgg16/imagenet/block4_conv1_bias.txt");
        this->block4_conv1 = new Conv2DLayer(device, block4_conv1_kernel, block4_conv1_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block4_conv2_kernel(kernel_size, kernel_size, 512, 512);
        load_tensor_from_file(block4_conv2_kernel, "../../data/vgg16/imagenet/block4_conv2_kernel.txt");
        Tensor_1D block4_conv2_bias(512);
        load_tensor_from_file(block4_conv2_bias, "../../data/vgg16/imagenet/block4_conv2_bias.txt");
        this->block4_conv2 = new Conv2DLayer(device, block4_conv2_kernel, block4_conv2_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block4_conv3_kernel(kernel_size, kernel_size, 512, 512);
        load_tensor_from_file(block4_conv3_kernel, "../../data/vgg16/imagenet/block4_conv3_kernel.txt");
        Tensor_1D block4_conv3_bias(512);
        load_tensor_from_file(block4_conv3_bias, "../../data/vgg16/imagenet/block4_conv3_bias.txt");
        this->block4_conv3 = new Conv2DLayer(device, block4_conv3_kernel, block4_conv3_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        this->block4_pool = new MaxPooling(device, 2);

        Tensor_4D block5_conv1_kernel(kernel_size, kernel_size, 512, 512);
        load_tensor_from_file(block5_conv1_kernel, "../../data/vgg16/imagenet/block5_conv1_kernel.txt");
        Tensor_1D block5_conv1_bias(512);
        load_tensor_from_file(block5_conv1_bias, "../../data/vgg16/imagenet/block5_conv1_bias.txt");
        this->block5_conv1 = new Conv2DLayer(device, block5_conv1_kernel, block5_conv1_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block5_conv2_kernel(kernel_size, kernel_size, 512, 512);
        load_tensor_from_file(block5_conv2_kernel, "../../data/vgg16/imagenet/block5_conv2_kernel.txt");
        Tensor_1D block5_conv2_bias(512);
        load_tensor_from_file(block5_conv2_bias, "../../data/vgg16/imagenet/block5_conv2_bias.txt");
        this->block5_conv2 = new Conv2DLayer(device, block5_conv2_kernel, block5_conv2_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        Tensor_4D block5_conv3_kernel(kernel_size, kernel_size, 512, 512);
        load_tensor_from_file(block5_conv3_kernel, "../../data/vgg16/imagenet/block5_conv3_kernel.txt");
        Tensor_1D block5_conv3_bias(512);
        load_tensor_from_file(block5_conv3_bias, "../../data/vgg16/imagenet/block5_conv3_bias.txt");
        this->block5_conv3 = new Conv2DLayer(device, block5_conv3_kernel, block5_conv3_bias, new ReLU<DEVICE, 4>(device), std::vector<int>{1, 1, 1, 1});

        this->block5_pool = new MaxPooling(device, 2);

        this->flatten = new FlattenLayer<DEVICE, 4>(device);

        int reduced_image_size = IMG_SIZE / 32;  

        const int flatten_size = reduced_image_size * reduced_image_size * 512;

        Tensor_2D WD(flatten_size, hidden_neurons);
        glorot_uniform_initializer(rng, WD);
        this->dense = new DenseLayer(device, WD, use_bias, new ReLU<DEVICE, 2>(device));
        this->dense->set_weight_optimizer(Adam<2>());
        this->dense->set_bias_optimizer(Adam<1>());

        Tensor_2D WC(hidden_neurons, num_classes);
        glorot_uniform_initializer(rng, WC);
        this->classifier_head = new SoftmaxCrossEntropyLayer(device, WC, use_bias);
        this->classifier_head->set_weight_optimizer(Adam<2>());
        this->classifier_head->set_bias_optimizer(Adam<1>());

        Tensor_2D WR_3(hidden_neurons, 128);
        glorot_uniform_initializer(rng, WR_3);
        this->regressor_neck_3 = new DenseLayer(device, WR_3, use_bias, new ReLU<DEVICE, 2>(device));
        this->regressor_neck_3->set_weight_optimizer(Adam<2>());
        this->regressor_neck_3->set_bias_optimizer(Adam<1>());

        Tensor_2D WR_2(128, 64);
        glorot_uniform_initializer(rng, WR_2);
        this->regressor_neck_2 = new DenseLayer(device, WR_2, use_bias, new ReLU<DEVICE, 2>(device));
        this->regressor_neck_2->set_weight_optimizer(Adam<2>());
        this->regressor_neck_2->set_bias_optimizer(Adam<1>());

        Tensor_2D WR_1(64, 32);
        glorot_uniform_initializer(rng, WR_1);
        this->regressor_neck_1 = new DenseLayer(device, WR_1, use_bias, new ReLU<DEVICE, 2>(device));
        this->regressor_neck_1->set_weight_optimizer(Adam<2>());
        this->regressor_neck_1->set_bias_optimizer(Adam<1>());

        Tensor_2D WR(32, 4);
        glorot_uniform_initializer(rng, WR);
        WR = WR * WR.constant(0.001);
        this->regressor_head = new DenseLayer(device, WR, use_bias, new Sigmoid<DEVICE, 2>(device));
        this->regressor_head->set_weight_optimizer(Adam<2>());
        this->regressor_head->set_bias_optimizer(Adam<1>());

        this->block5_conv3->set_kernels_optimizer(Adam<4>());
        this->block5_conv3->set_bias_optimizer(Adam<1>());

    }

    ~Model() {
        delete block1_conv1;
        delete block1_conv2;
        delete block1_pool;

        delete block2_conv1;
        delete block2_conv2;
        delete block2_pool;

        delete block3_conv1;
        delete block3_conv2;
        delete block3_conv3;
        delete block3_pool;

        delete block4_conv1;
        delete block4_conv2;
        delete block4_conv3;
        delete block4_pool;

        delete block5_conv1;
        delete block5_conv2;
        delete block5_conv3;
        delete block5_pool;

        delete flatten;
        delete dense;

        delete classifier_head;

        delete regressor_neck_3;
        delete regressor_neck_2;
        delete regressor_neck_1;
        delete regressor_head;
    }

    void forward(const Tensor_4D &input) {

        this->block1_conv1->forward(input);
        this->block1_conv2->forward(this->block1_conv1->get_output()); 
        this->block1_pool->forward(this->block1_conv2->get_output()); 

        this->block2_conv1->forward(this->block1_pool->get_output());
        this->block2_conv2->forward(this->block2_conv1->get_output()); 
        this->block2_pool->forward(this->block2_conv2->get_output()); 

        this->block3_conv1->forward(this->block2_pool->get_output());
        this->block3_conv2->forward(this->block3_conv1->get_output()); 
        this->block3_conv3->forward(this->block3_conv2->get_output()); 
        this->block3_pool->forward(this->block3_conv3->get_output()); 

        this->block4_conv1->forward(this->block3_pool->get_output());
        this->block4_conv2->forward(this->block4_conv1->get_output()); 
        this->block4_conv3->forward(this->block4_conv2->get_output()); 
        this->block4_pool->forward(this->block4_conv3->get_output()); 

        this->block5_conv1->forward(this->block4_pool->get_output());
        this->block5_conv2->forward(this->block5_conv1->get_output()); 
        this->block5_conv3->forward(this->block5_conv2->get_output()); 
        this->block5_pool->forward(this->block5_conv3->get_output()); 

        this->flatten->forward(this->block5_pool->get_output());
        this->dense->forward(this->flatten->get_output()); 

        this->classifier_head->forward(this->dense->get_output()); 

        this->regressor_neck_3->forward(this->dense->get_output()); 
        this->regressor_neck_2->forward(this->regressor_neck_3->get_output()); 
        this->regressor_neck_1->forward(this->regressor_neck_2->get_output()); 
        this->regressor_head->forward(this->regressor_neck_1->get_output()); 

    }

    void backward(const Tensor_2D &classifier_TRUE, const Tensor_2D &regressor_TRUE) {

        this->classifier_head->backward(classifier_TRUE, true);

        MSE mse_fn;
        const Tensor_2D regressor_upstream = mse_fn.derivative(regressor_TRUE, regressor_head->get_output());
        this->regressor_head->backward(regressor_upstream, true);
        this->regressor_neck_1->backward(this->regressor_head->get_downstream(), true);
        this->regressor_neck_2->backward(this->regressor_neck_1->get_downstream(), true);
        this->regressor_neck_3->backward(this->regressor_neck_2->get_downstream(), true);

        // combined backward

        Tensor_2D combined_gradient = this->classifier_head->get_downstream() + this->regressor_neck_3->get_downstream();
        
        this->dense->backward(combined_gradient, true);

        // Fine tunning. Backwards the last VGG layer
        this->flatten->backward(this->dense->get_downstream(), true);
        this->block5_pool->backward(this->flatten->get_downstream(), true); 
        this->block5_conv3->backward(this->block5_pool->get_downstream(), false); 

    }

    void update(const TYPE learning_rate, int epoch) {
        this->dense->update(learning_rate, epoch);

        this->classifier_head->update(learning_rate, epoch);

        this->regressor_head->update(learning_rate, epoch);
        this->regressor_neck_1->update(learning_rate, epoch);
        this->regressor_neck_2->update(learning_rate, epoch);
        this->regressor_neck_3->update(learning_rate, epoch);

        this->block5_conv3->update(learning_rate, epoch);
    }

    auto predict(const Tensor_4D &input) {
        auto y1 = this->block1_conv1->predict(input);
        auto y2 = this->block1_conv2->predict(y1);
        auto y3 = this->block1_pool->predict(y2);

        auto y4 = this->block2_conv1->predict(y3);
        auto y5 = this->block2_conv2->predict(y4);
        auto y6 = this->block2_pool->predict(y5);

        auto y7 = this->block3_conv1->predict(y6);
        auto y8 = this->block3_conv2->predict(y7);
        auto y9 = this->block3_conv3->predict(y8);
        auto y10 = this->block3_pool->predict(y9);

        auto y11 = this->block4_conv1->predict(y10);
        auto y12 = this->block4_conv2->predict(y11);
        auto y13 = this->block4_conv3->predict(y12);
        auto y14 = this->block4_pool->predict(y13);

        auto y15 = this->block5_conv1->predict(y14);
        auto y16 = this->block5_conv2->predict(y15);
        auto y17 = this->block5_conv3->predict(y16);
        auto y18 = this->block5_pool->predict(y17);

        auto y19 = this->flatten->predict(y18);
        auto y20 = this->dense->predict(y19);

        auto clazz = this->classifier_head->predict(y20);

        auto y21 = this->regressor_neck_3->predict(y20);
        auto y22= this->regressor_neck_2->predict(y21);
        auto y23 = this->regressor_neck_1->predict(y22);
        auto bouding_box = this->regressor_head->predict(y23);

        return std::make_pair(clazz, bouding_box);
    }

    const Tensor_2D get_class_prediction() const {
        return this->classifier_head->get_output();
    }

    const Tensor_2D get_boundingbox_prediction() const {
        return this->regressor_head->get_output();
    }
    
    const int get_IMG_SIZE () const {
        return this->IMG_SIZE;
    }

private:
    const int IMG_SIZE;

    // VGG16 feature extractor

    Conv2DLayer<DEVICE> *block1_conv1;
    Conv2DLayer<DEVICE> *block1_conv2;
    MaxPooling<DEVICE> *block1_pool;

    Conv2DLayer<DEVICE> *block2_conv1;
    Conv2DLayer<DEVICE> *block2_conv2;
    MaxPooling<DEVICE> *block2_pool;

    Conv2DLayer<DEVICE> *block3_conv1;
    Conv2DLayer<DEVICE> *block3_conv2;
    Conv2DLayer<DEVICE> *block3_conv3;
    MaxPooling<DEVICE> *block3_pool;

    Conv2DLayer<DEVICE> *block4_conv1;
    Conv2DLayer<DEVICE> *block4_conv2;
    Conv2DLayer<DEVICE> *block4_conv3;
    MaxPooling<DEVICE> *block4_pool;

    Conv2DLayer<DEVICE> *block5_conv1;
    Conv2DLayer<DEVICE> *block5_conv2;
    Conv2DLayer<DEVICE> *block5_conv3;
    MaxPooling<DEVICE> *block5_pool;

    FlattenLayer<DEVICE, 4> *flatten;
    DenseLayer<DEVICE> *dense;

    // heads

    DenseLayer<DEVICE> *classifier_head;

    DenseLayer<DEVICE> *regressor_head;
    DenseLayer<DEVICE> *regressor_neck_1;
    DenseLayer<DEVICE> *regressor_neck_2;
    DenseLayer<DEVICE> *regressor_neck_3;

};

int main(int, char **)
{

    using DEVICE = Eigen::ThreadPoolDevice;
    using GEN = std::mt19937;
    using MODEL = Model<DEVICE, GEN>;

    std::random_device rd{};
    const auto seed = rd();
    GEN rng(seed);
    std::cout << "Using seed " << seed << "\n";

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    DEVICE device(&tp, threads);

    const int IMAGE_SIZE = 224;

    MODEL model(device, rng, IMAGE_SIZE);

    run_experiment<DEVICE, GEN, MODEL, Data_Augmentation_ParallelBatches<GEN>>(
        "../../data/Oxford-IIIT_Pet_Dataset",
        model, device, rng, 
        3, // epochs
        16, // minibatch_size
        0.001, // learning_rate
        false // square_images
    );

    std::cout << "success\n";

    return 0;
}