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

#include <unsupported/Eigen/CXX11/Tensor>

template <typename RANDOM_GEN>
std::tuple<Eigen::Tensor<std::string, 2>, Eigen::Tensor<std::string, 2>>
load_dogs_x_cats_dataset(std::string folder_path, RANDOM_GEN random_generator, float split_percentage = .8)
{
    if (!fs::is_directory(folder_path))
        throw std::invalid_argument("Folder " + folder_path + " not found or is not a directory");

    std::vector<std::string> cats;
    cats.reserve(12'501);
    std::vector<std::string> dogs;
    dogs.reserve(12'501);

    for (const auto & entry : fs::directory_iterator(folder_path + std::filesystem::path::preferred_separator + "Cat"))
        cats.emplace_back(entry.path());
    
    for (const auto & entry : fs::directory_iterator(folder_path + std::filesystem::path::preferred_separator + "Dog"))
        dogs.emplace_back(entry.path());

    const int N_REGISTERS = cats.size() + dogs.size();

    const int split_at = static_cast<int>(N_REGISTERS * split_percentage);

    if (cats.size() <= split_at / 2) {
        throw std::runtime_error("insuficient number of cat images");
    }

    if (dogs.size() <= split_at / 2) {
        throw std::runtime_error("insuficient number of dog images");
    }

    std::vector<int> training_indexes(split_at);
    std::iota(training_indexes.begin(), training_indexes.end(), 0);

    std::vector<int> validation_indexes(N_REGISTERS - split_at);
    std::iota(validation_indexes.begin(), validation_indexes.end(), 0);

    std::shuffle(cats.begin(), cats.end(), random_generator);
    std::shuffle(dogs.begin(), dogs.end(), random_generator);
    std::shuffle(training_indexes.begin(), training_indexes.end(), random_generator);
    std::shuffle(validation_indexes.begin(), validation_indexes.end(), random_generator);

    Eigen::Tensor<std::string, 2> training(training_indexes.size(), 2);
    Eigen::Tensor<std::string, 2> validation(validation_indexes.size(), 2);

    // Note that, because of corrupted files, the numbers of cat images and dog images are slightly imbalanced

    int cats_index_pointer = 0;
    int dogs_index_pointer = 0;
    for (int i = 0; i < split_at; i+=2) {

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

#endif