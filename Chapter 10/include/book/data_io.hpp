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
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

Tensor_2D load_images(const std::string &filepath, const std::vector<int> indexes);

Tensor_2D load_labels(const std::string &filepath, const std::vector<int> indexes);

template <typename RANDOM_GEN>
std::tuple<Tensor_2D, Tensor_2D> load_mnist(const std::string &folder, RANDOM_GEN &random_generator) {

    std::vector<int> training_indexes(60'000);
    std::iota(training_indexes.begin(), training_indexes.end(), 0);
    std::shuffle(training_indexes.begin(), training_indexes.end(), random_generator);

    auto training_images = load_images(folder + "/train-images.idx3-ubyte", training_indexes);
    auto training_labels = load_labels(folder + "/train-labels.idx1-ubyte", training_indexes);

    return std::make_tuple(training_images, training_labels);
}

#endif