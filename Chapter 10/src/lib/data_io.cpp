#include <iostream>
#include <fstream>
#include <exception>
#include <algorithm>

#include "book/data_io.hpp"

uint32_t read_uint32(std::ifstream &stream, size_t position)
{
    stream.seekg(position, std::ios::beg);
    uint32_t temp;
    stream.read(reinterpret_cast<char *>(&temp), sizeof(temp));
    uint32_t result = ((temp << 8) & 0xFF00FF00) | ((temp >> 8) & 0xFF00FF);
    return (result << 16) | (result >> 16);
}

Eigen::Tensor<float, 2> load_images(const std::string &filepath, const std::vector<int> indexes)
{
    std::ifstream images_stream(filepath, std::ios::in | std::ios::binary);

    if (!images_stream.is_open())
        throw std::invalid_argument("failed to open the images file.");

    const int magic_number = read_uint32(images_stream, 0);

    if (magic_number != 2051)
        throw std::invalid_argument("failed to read magic number in images file.");

    const unsigned int instances = read_uint32(images_stream, 4);

    if (indexes.size() != instances)
        throw std::invalid_argument("Amount of instances does not match.");

    const int rows = read_uint32(images_stream, 8);

    const int cols = read_uint32(images_stream, 12);

    const int size = rows * cols;

    std::vector<float> raw_data(size * instances);

    std::vector<unsigned char> buffer(size);

    auto normalization = [](unsigned char c){ 
        return static_cast<float>(c) / 255.f; 
    };

    for (unsigned int i = 0; i < instances; ++i)
    {
        auto ref = &buffer[0];
        auto deref = reinterpret_cast<char *>(ref);
        images_stream.read(deref, size);
        int instance = indexes.at(i);
        transform(buffer.begin(), buffer.end(), raw_data.begin() + instance*size, normalization);
    }

    Eigen::Tensor<float, 2> tensor = Eigen::TensorMap<Eigen::Tensor<float, 2>>(raw_data.data(), size, instances);
    Eigen::array<Eigen::Index, 2> shuffling({1, 0});
    Eigen::Tensor<float, 2> result = tensor.shuffle(shuffling);
    return result;
}

Eigen::Tensor<float, 2> load_labels(const std::string &filepath, const std::vector<int> indexes)
{
    std::ifstream labels_stream(filepath, std::ios::in | std::ios::binary);

    if (!labels_stream.is_open())
        throw std::invalid_argument("failed to open the labels file.");

    const int magic_number = read_uint32(labels_stream, 0);

    if (magic_number != 2049)
        throw std::invalid_argument("failed to read magic number in labels file.");

    const unsigned int instances = read_uint32(labels_stream, 4);

    if (indexes.size() != instances)
        throw std::invalid_argument("Amount of instances does not match.");

    Eigen::Tensor<float, 2> result(instances, 10);
    result.setZero();

    for (unsigned int i = 0; i < instances; ++i)
    {
        char label;
        labels_stream.read(&label, 1);
        int int_label = static_cast<int>(label);
        int instance = indexes.at(i);
        result(instance, int_label) = 1.f;
    }

    return result;
}

std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>> load_mnist(const std::string &folder, bool shuffle){

    auto rd = std::random_device {}; 
    auto rng = std::default_random_engine { rd() };
    
    std::vector<int> training_indexes(60'000);
    std::iota(training_indexes.begin(), training_indexes.end(), 0);
    if (shuffle) std::shuffle(training_indexes.begin(), training_indexes.end(), rng);

    auto training_images = load_images(folder + "/train-images.idx3-ubyte", training_indexes);
    auto training_labels = load_labels(folder + "/train-labels.idx1-ubyte", training_indexes);

    std::vector<int> test_indexes(10'000);
    std::iota(test_indexes.begin(), test_indexes.end(), 0);
    if (shuffle) std::shuffle(test_indexes.begin(), test_indexes.end(), rng);

    auto test_images = load_images(folder + "/t10k-images.idx3-ubyte", test_indexes);
    auto test_labels = load_labels(folder + "/t10k-labels.idx1-ubyte", test_indexes);

    return std::make_tuple(training_images, training_labels, test_images, test_labels);
}