/*
 * This file is part of Coding Deep Learning from Scratch Book, BPB PUBLICATIONS .
 *
 * Author: Luiz doleron <doleron@gmail.com>
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef DATA_IO_H_
#define DATA_IO_H_

#include <cstdint>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

#include <opencv2/opencv.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

std::tuple<Tensor_2D, Tensor_2D, Tensor_2D, Tensor_2D> load_mnist(const std::string &folder, bool shuffle = true, long unsigned int seed = 1234);

Tensor_2D load_mnist_chunck(const std::string &folder, long int size, bool shuffle = true, long unsigned int seed = 1234);

template <typename RANDOM_GEN>
std::tuple<Tensor_2D, Tensor_2D, Tensor_2D, Tensor_2D> load_iris_dataset(std::string file_path, RANDOM_GEN random_generator, TYPE split_percentage = .8)
{
    std::ifstream file;
    file.open(file_path);
    const int N_REGISTERS = 150;

    if (!file.is_open())
        throw std::invalid_argument("File " + file_path + " not found.");

    std::vector<std::string> lines;
    lines.reserve(N_REGISTERS);
    std::string line;

    while (getline(file, line))
    {
        lines.push_back(line);
    }

    std::shuffle(lines.begin(), lines.end(), random_generator);

    std::vector<TYPE> data;
    const int EXPECTED_SIZE = N_REGISTERS * (4 + 3); // 4 attributes + 3 hot-encoding for the 3 classes
    data.reserve(EXPECTED_SIZE);
    std::string element;
    const std::string class_setosa = "Iris-setosa";
    const std::string class_versicolor = "Iris-versicolor";
    const std::string class_virginica = "Iris-virginica";

    for (const auto &_line : lines)
    {
        std::stringstream ss(_line);
        while (getline(ss, element, ','))
        {
            if (class_setosa.compare(element) == 0)
            {
                data.push_back(1.);
                data.push_back(0.);
                data.push_back(0.);
            }
            else if (class_versicolor.compare(element) == 0)
            {
                data.push_back(0.);
                data.push_back(1.);
                data.push_back(0.);
            }
            else if (class_virginica.compare(element) == 0)
            {
                data.push_back(0.);
                data.push_back(0.);
                data.push_back(1.);
            }
            else
            {
                TYPE value = std::stof(element);
                data.push_back(value);
            }
        }
    }

    if (data.size() != EXPECTED_SIZE)
    {
        throw std::invalid_argument("Wrong dataset size: " + std::to_string(data.size()));
    }

    Eigen::array<int, 2> dims({1, 0});
    Tensor_2D tensor_map = Eigen::TensorMap<Tensor_2D>(data.data(), 7, 150).shuffle(dims);

    const int split_at = static_cast<int>(N_REGISTERS * split_percentage);

    Eigen::array<Eigen::Index, 2> training_x_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> training_x_extents = {split_at, 4};
    Tensor_2D training_X_ds = tensor_map.slice(training_x_offsets, training_x_extents);

    Eigen::array<Eigen::Index, 2> training_y_offsets = {0, 4};
    Eigen::array<Eigen::Index, 2> training_y_extents = {split_at, 3};
    Tensor_2D training_Y_ds = tensor_map.slice(training_y_offsets, training_y_extents);

    Eigen::array<Eigen::Index, 2> validation_x_offsets = {split_at, 0};
    Eigen::array<Eigen::Index, 2> validation_x_extents = {N_REGISTERS - split_at, 4};
    Tensor_2D validation_X_ds = tensor_map.slice(validation_x_offsets, validation_x_extents);

    Eigen::array<Eigen::Index, 2> validation_y_offsets = {split_at, 4};
    Eigen::array<Eigen::Index, 2> validation_y_extents = {N_REGISTERS - split_at, 3};
    Tensor_2D validation_Y_ds = tensor_map.slice(validation_y_offsets, validation_y_extents);

    auto result = std::make_tuple(training_X_ds, training_Y_ds, validation_X_ds, validation_Y_ds);

    return result;
}

template <typename RANDOM_GEN>
std::tuple<Eigen::Tensor<std::string, 2>, Eigen::Tensor<std::string, 2>>
load_dogs_x_cats_dataset(std::string folder_path, RANDOM_GEN random_generator, TYPE split_percentage = .8, bool rewrite_images = false)
{

    if (rewrite_images) {
        std::cout << "rewrite_images = true. The images will be overrides by themselves\n";
        std::cout << "Press Enter key to continue...\n";
        std::cin.get();
        std::cout << "Loading data...\n";
    } 


    if (!fs::is_directory(folder_path))
        throw std::invalid_argument("Folder " + folder_path + " not found or is not a directory");

    std::vector<std::string> cats;
    cats.reserve(12'501);
    std::vector<std::string> dogs;
    dogs.reserve(12'501);

    for (const auto & entry : fs::directory_iterator(folder_path + std::filesystem::path::preferred_separator + "Cat")) {
        cv::Mat image = cv::imread(entry.path(), cv::IMREAD_COLOR);
        if (!image.empty()) {
            // fix eventually small errors such as:
            // Corrupt JPEG data: 22 extraneous bytes before marker 0xd9
            if (rewrite_images) {
                cv::imwrite(entry.path(), image);
            }
            cats.emplace_back(entry.path());
        } else {

            if (rewrite_images) {
                std::cout << "removing " << entry.path() << "\n";
                fs::remove(entry.path());
            }
        }
    }
    
    for (const auto & entry : fs::directory_iterator(folder_path + std::filesystem::path::preferred_separator + "Dog")) {
        cv::Mat image = imread(entry.path(), cv::IMREAD_COLOR);
        if (!image.empty()) {
            // fix eventually small errors such as:
            // Corrupt JPEG data: 22 extraneous bytes before marker 0xd9
            if (rewrite_images) {
                cv::imwrite(entry.path(), image);
            }
            dogs.emplace_back(entry.path());
        } else {

            if (rewrite_images) {
                std::cout << "removing " << entry.path() << "\n";
                fs::remove(entry.path());
            }
        }
    }

    const int N_REGISTERS = cats.size() + dogs.size();

    const int training_size = (static_cast<int>(N_REGISTERS * split_percentage) / 2) * 2;
    const int validation_size = N_REGISTERS - training_size;

    std::cout << "Total of images: " << N_REGISTERS << "\n";
    std::cout << "Training size: " << training_size << "\n";
    std::cout << "Validation size " << validation_size << "\n";

    if (cats.size() <= training_size / 2) {
        throw std::runtime_error("insuficient number of cat images");
    }

    if (dogs.size() <= training_size / 2) {
        throw std::runtime_error("insuficient number of dog images");
    }

    std::vector<int> training_indexes(training_size);
    std::iota(training_indexes.begin(), training_indexes.end(), 0);

    std::vector<int> validation_indexes(validation_size);
    std::iota(validation_indexes.begin(), validation_indexes.end(), 0);

    std::shuffle(cats.begin(), cats.end(), random_generator);
    std::shuffle(dogs.begin(), dogs.end(), random_generator);
    std::shuffle(training_indexes.begin(), training_indexes.end(), random_generator);
    std::shuffle(validation_indexes.begin(), validation_indexes.end(), random_generator);

    Eigen::Tensor<std::string, 2> training(training_size, 2);
    Eigen::Tensor<std::string, 2> validation(validation_size, 2);

    // Note that, because of corrupted files, the numbers of cat images and dog images are slightly imbalanced

    int cats_index_pointer = 0;
    int dogs_index_pointer = 0;
    for (int i = 0; i < training_size; i+=2) {

        int c_index = training_indexes[i];
        training(c_index, 0) = cats[cats_index_pointer++];
        training(c_index, 1) = "0";

        int d_index = training_indexes[i + 1];
        training(d_index, 0) = dogs[dogs_index_pointer++];
        training(d_index, 1) = "1";

    }

    int validation_index_pointer = 0;

    for (; cats_index_pointer < cats.size(); ++cats_index_pointer) {

        int index = validation_indexes[validation_index_pointer++];
        validation(index, 0) = cats[cats_index_pointer];
        validation(index, 1) = "0";
    }

    for (; dogs_index_pointer < dogs.size(); ++dogs_index_pointer) {

        int index = validation_indexes[validation_index_pointer++];
        validation(index, 0) = dogs[dogs_index_pointer];
        validation(index, 1) = "1";
    }

    auto result = std::make_tuple(training, validation);

    return result;
}

template <int _RANK>
void load_tensor_from_file(Tensor<_RANK> &tensor, const std::string& file_path, bool as_row_major) {

    std::ifstream file;
    file.open(file_path);

    if (!file.is_open()) {throw std::invalid_argument("Can't open " + file_path);}

    std::vector<TYPE> vec;
    vec.reserve(tensor.size());
    
    TYPE val;
    for(std::string line; std::getline(file, line); ) {
        std::istringstream in(line);  
        in >> val;      
        vec.push_back(val);
    }

    if (vec.size() != tensor.size()) {throw std::invalid_argument("Illegal number of lines " + std::to_string(vec.size()));}

    Eigen::array<Eigen::Index, _RANK> suffle_dims;
    for (int i = 0; i < _RANK; ++i) {
        suffle_dims[i] = _RANK - i - 1;
    }

    auto dimensions = tensor.dimensions();
    if (as_row_major) {
        auto mapped = Eigen::TensorMap<Eigen::Tensor<TYPE, _RANK, Eigen::RowMajor>>(&vec[0], dimensions);
        tensor = Eigen::TensorLayoutSwapOp<Eigen::Tensor<TYPE, _RANK, Eigen::RowMajor>>(mapped).shuffle(suffle_dims);
    } else {
        tensor = Eigen::TensorMap<Eigen::Tensor<TYPE, _RANK>>(&vec[0], dimensions);
    }

    file.close();
}

template <int _RANK>
void save_tensor_to_file(const Tensor<_RANK> &tensor, const std::string& file_path, bool as_row_major) {


    std::ofstream file;
    file.open(file_path, std::ios::out | std::ios::trunc);

    if (!file.is_open()) {throw std::invalid_argument("Can't open " + file_path);}

    const float * data = tensor.data();

    Eigen::Tensor<TYPE, _RANK, Eigen::RowMajor> tensor_rm;

    if (as_row_major) {
        Eigen::array<Eigen::Index, _RANK> suffle_dims;
        for (int i = 0; i < _RANK; ++i) {
            suffle_dims[i] = _RANK - i - 1;
        }
        tensor_rm = Eigen::TensorLayoutSwapOp<Eigen::Tensor<TYPE, _RANK, Eigen::ColMajor>>(tensor).shuffle(suffle_dims);

        data = tensor_rm.data();
    }

    const int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        file << data[i];
        file << "\n";
    }

    file.close();

}

#endif