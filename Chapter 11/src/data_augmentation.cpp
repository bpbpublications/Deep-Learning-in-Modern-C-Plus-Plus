#include <iostream>

#include "gnuplot-iostream.h"

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/book.hpp"

std::random_device rd{};
const auto seed = rd();
std::mt19937 rng(seed);

template <typename Device>
class Model {

public:

    Model(Device &device, const int hidden_layer_size) {

        const bool use_bias = false;

        Tensor_2D W1(28*28, hidden_layer_size);
        glorot_uniform_initializer(rng, W1);
        this->dense_1 = new DenseLayer(device, W1, use_bias, new ReLU<Device, 2>(device));

        Tensor_2D W2(hidden_layer_size, 64);
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
              const Tensor_2D &validation_images, const Tensor_2D &validation_labels,
              const int MAX_EPOCHS, float learning_rate) {

    CategoricalCrossEntropy cost_fn;

    int epoch = 0;
    std::vector<std::pair<float, float>> training_losses;
    std::vector<std::pair<float, float>> validation_losses;
    std::vector<std::pair<float, float>> training_accs;
    std::vector<std::pair<float, float>> validation_accs;

    float min_validation_loss = std::numeric_limits<float>::max();
    int early_stop_count = 0;
    const int patience = std::numeric_limits<int>::max(); // set to 5 to check early stop

    while (epoch < MAX_EPOCHS && early_stop_count < patience)
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

        auto validation_pred = model.predict(validation_images);
        float validation_acc = accuracy(validation_labels, validation_pred);
        float validation_loss = cost_fn.evaluate(validation_labels, validation_pred);

        std::cout 
                << "epoch:\t" << epoch << "\t"
                << "took:\t" << duration.count() << " mills\t"
                << "\ttraining_loss:\t" << training_loss
                << "\ttraining_acc:\t" << training_acc
                << "\tvalidation_loss:\t" << validation_loss
                << "\tvalidation_acc:\t" << validation_acc
                << "\n";

        training_losses.emplace_back(std::make_pair(epoch, training_loss));
        validation_losses.emplace_back(std::make_pair(epoch, validation_loss));

        training_accs.emplace_back(std::make_pair(epoch, training_acc));
        validation_accs.emplace_back(std::make_pair(epoch, validation_acc));

        if (validation_loss < min_validation_loss) {
            min_validation_loss = validation_loss;
            early_stop_count = 0;
        } else {
            early_stop_count++;
        }

        epoch++;

    }

    {

        Gnuplot gp;

        int max_x = training_losses.back().first;
        float max_y = 0;
        float min_validation_loss = 1'000'000;
        int min_validation_loss_epoch = 0;
        for (unsigned int i = 0; i < training_losses.size(); i++)
        {
            if (training_losses[i].second > max_y) max_y = training_losses[i].second;
            if (validation_losses[i].second > max_y) max_y = validation_losses[i].second;
            if (validation_losses[i].second < min_validation_loss) {
                min_validation_loss = validation_losses[i].second;
                min_validation_loss_epoch = i;
            }
        }

        std::cout << "The lowest validation cost was " << min_validation_loss << " achieved at epoch " << min_validation_loss_epoch << "\n";

        gp << "set xrange [0:" << max_x << "]\nset yrange [0:" << max_y << "]\nset ytic 0.2\n";
        gp << "plot '-' with lines lc rgb \"red\" title 'training loss', '-' with lines  lc rgb \"blue\" title 'validation loss', '-' with lines lc rgb \"green\" lt 2 lw 2 dt 3 title 'min validation loss'\n";

        gp.send1d(training_losses);
        gp.send1d(validation_losses);

        std::vector<std::pair<float, float>> min_validation_losses;
        min_validation_losses.reserve(validation_losses.size());
        for (int i = 0; i < validation_losses.size(); ++i) {
            min_validation_losses.emplace_back(std::make_pair(i, min_validation_loss));
        }
        gp.send1d(min_validation_losses);
    }

    {

        Gnuplot gp;

        int max_x = training_accs.back().first;

        gp << "set xrange [0:" << max_x << "]\n";
        gp << "set yrange [0:100]\nset ytic 5\n";
        gp << "plot '-' with lines lc rgb \"red\" title 'training acc', '-' with lines  lc rgb \"blue\" title 'validation acc', '-' with lines lc rgb \"green\" lt 2 lw 2 dt 3 title 'max validation acc'\n";

        gp.send1d(training_accs);
        gp.send1d(validation_accs);

        std::vector<std::pair<float, float>> max_validation_accs;
        float maxx = 0.f;
        for (unsigned int i = 0; i < validation_accs.size(); i++)
        {
            if (validation_accs[i].second > maxx) maxx = validation_accs[i].second;
        }
        max_validation_accs.reserve(validation_accs.size());
        for (int i = 0; i < validation_accs.size(); ++i) {
            max_validation_accs.emplace_back(std::make_pair(i, maxx));
        }
        gp.send1d(max_validation_accs);

    }

}

std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>> 
load_data(long int training_instances, long int validation_instances) {
    // loading full MNIST.
    // Note: load_mnist shuffle the data
    auto [training_images, training_labels, validation_images, validation_labels] = load_mnist("../../data/mnist", rng);

    // getting slice of data
    Eigen::array<Eigen::Index, 2> zero_offset = {0, 0};
    Eigen::array<Eigen::Index, 2> training_x_extents = {training_instances, 784};
    Eigen::array<Eigen::Index, 2> training_y_extents = {training_instances, 10};
    Eigen::array<Eigen::Index, 2> validation_x_extents = {validation_instances, 784};
    Eigen::array<Eigen::Index, 2> validation_y_extents = {validation_instances, 10};

    Eigen::Tensor<float, 2> training_x_ds = training_images.slice(zero_offset, training_x_extents);
    Eigen::Tensor<float, 2> training_y_ds = training_labels.slice(zero_offset, training_y_extents);

    Eigen::Tensor<float, 2> validation_x_ds = validation_images.slice(zero_offset, validation_x_extents);
    Eigen::Tensor<float, 2> validation_y_ds = validation_labels.slice(zero_offset, validation_y_extents);

    auto result = std::make_tuple(training_x_ds, training_y_ds, validation_x_ds, validation_y_ds);

    return result;
}

Tensor_1D rotate(Tensor_1D &instance, float angle) {

    auto data = instance.data();
    cv::Mat1f src(28, 28, data);
    cv::Mat1f rotated;

    cv::Mat rotation_mat = getRotationMatrix2D(cv::Point2f(src.cols / 2, src.rows / 2), angle, 1.0); 
    cv::warpAffine(src, rotated, rotation_mat, cv::Size(src.cols, src.rows));

    float * float_data = (float*)rotated.data;

    Tensor_1D result = Eigen::TensorMap<Tensor_1D>(float_data, instance.dimension(0));

    return result;
}

std::tuple<Tensor_2D, Tensor_2D> 
augment_dataset(Tensor_2D &src_images, Tensor_2D &src_labels, int number_augmented_instances) {

    const int N = src_images.dimension(0);

    const int M = N + number_augmented_instances;

    Tensor_2D result_images(M, src_images.dimension(1));
    Tensor_2D result_labels(M, src_labels.dimension(1));

    DimArray<2> zero_offset = {0, 0};
    DimArray<2> images_src_extents = {N, src_images.dimension(1)};
    DimArray<2> labels_src_extents = {N, src_labels.dimension(1)};

    // copying original data
    result_images.slice(zero_offset, images_src_extents) = src_images;
    result_labels.slice(zero_offset, labels_src_extents) = src_labels;

    // adding augmented instances

    std::uniform_int_distribution<> index_picker(0, N - 1);
    std::uniform_real_distribution<float> angle_picker(-30, 30);

    for (int i = N; i < M; ++i) {

        int src_index = index_picker(rd);

        Eigen::Tensor<float, 1> instance = src_images.chip<0>(src_index);
        Eigen::Tensor<float, 1> label = src_labels.chip<0>(src_index);

        float angle = angle_picker(rd);

        result_images.chip<0>(i) = rotate(instance, angle);
        result_labels.chip<0>(i) = label;
    }

    auto result = std::make_tuple(result_images, result_labels);

    return result;

}

int main(int, char **)
{
    
    auto [training_images_original, training_labels_original, validation_images, validation_labels] = load_data(100, 2000);
    std::cout << "Data loaded!\n";

    std::cout << "training_images_original dims: " << training_images_original.dimensions() << "\n";
    std::cout << "training_labels_original dims: " << training_labels_original.dimensions() << "\n";
    std::cout << "validation_images dims: " << validation_images.dimensions() << "\n";
    std::cout << "validation_labels dims: " << validation_labels.dimensions() << "\n";

    auto [training_images, training_labels] = augment_dataset(training_images_original, training_labels_original, 400);

    std::cout << "training_images dims: " << training_images.dimensions() << "\n";
    std::cout << "training_labels dims: " << training_labels.dimensions() << "\n";

    const int MAX_EPOCHS = 500;
    const float learning_rate = 0.01f;

    const int threads = std::thread::hardware_concurrency();
    Eigen::ThreadPool tp(threads);
    Eigen::ThreadPoolDevice device(&tp, threads);

    Model model(device, 512);
    training(model, training_images, training_labels, validation_images, validation_labels, MAX_EPOCHS, learning_rate);

    return 0;
}