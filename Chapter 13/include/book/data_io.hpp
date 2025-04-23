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

#include <opencv2/opencv.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

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

template <typename RANDOM_GEN>
std::tuple<Eigen::Tensor<std::string, 2>, Eigen::Tensor<std::string, 2>>
load_flowers_dataset(std::string folder_path, RANDOM_GEN random_generator, TYPE split_percentage = .8)
{

    if (!fs::is_directory(folder_path))
        throw std::invalid_argument("Folder " + folder_path + " not found or is not a directory");

    auto sep = std::filesystem::path::preferred_separator;

    std::unordered_map<std::string, std::string> label_map;
    label_map[folder_path + sep + "daisy"] = "0";
    label_map[folder_path + sep + "dandelion"] = "1";
    label_map[folder_path + sep + "roses"] = "2";
    label_map[folder_path + sep + "sunflowers"] = "3";
    label_map[folder_path + sep + "tulips"] = "4";

    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(3'700);

    for (const auto & entry : fs::recursive_directory_iterator(folder_path)) {
        auto path = entry.path();
        if (path.extension().compare(".jpg") == 0) {
            std::string parent = path.parent_path().string();
            if (label_map.find(parent) != label_map.end()) {
                std::string clazz = label_map[parent];
                entries.emplace_back(std::make_pair(entry.path().string(), clazz));
            }
        }
    }
    
    const int N_REGISTERS = entries.size();
    const int training_size = static_cast<int>(N_REGISTERS * split_percentage);
    const int validation_size = N_REGISTERS - training_size;

    std::cout << "Total of images: " << N_REGISTERS << "\n";
    std::cout << "Training size: " << training_size << "\n";
    std::cout << "Validation size " << validation_size << "\n";

    std::shuffle(entries.begin(), entries.end(), random_generator);

    Eigen::Tensor<std::string, 2> training(training_size, 2);
    Eigen::Tensor<std::string, 2> validation(validation_size, 2);

    for (int i = 0; i < training_size; ++i) {
        auto entry = entries[i];
        training(i, 0) = std::get<0>(entry);
        training(i, 1) = std::get<1>(entry);
    }

    for (int i = 0; i < validation_size; ++i) {
        auto entry = entries[i + training_size];
        validation(i, 0) = std::get<0>(entry);
        validation(i, 1) = std::get<1>(entry);
    }

    auto result = std::make_tuple(training, validation);

    return result;
}

/**
 * This function load the class and bounding box annotations from xml files in the annotations folder subdir of the 
 * provided folder_path dir repository.
 * 
 * Files with two or more objects are discarded.
*/
template <typename RANDOM_GEN>
std::tuple<Eigen::Tensor<std::string, 2>, Eigen::Tensor<std::string, 2>>
load_oxford_pets_dataset(std::string folder_path, RANDOM_GEN random_generator, TYPE split_percentage = .8, bool check_file_integrity = false)
{

    if (!fs::is_directory(folder_path))
        throw std::invalid_argument("Folder " + folder_path + " not found or is not a directory");

    auto sep = std::filesystem::path::preferred_separator;

    std::vector<std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string>> entries;
    entries.reserve(3'700);

    const auto extract_token = [](const std::string &text, const std::string &tag, std::string &value) {
        const std::string query = "<" + tag + ">[A-Za-z0-9_.]+</" + tag + ">";
        std::smatch _match;
        int result = -1;
        if (std::regex_search(text, _match, std::regex(query))) {
            std::string str = _match.str();
            int tag_len = tag.length();
            value = str.substr(tag_len + 2, str.length() - 2*tag_len - 5);
            result = _match.position();
        }
        return result;
    };

    int dog_counter = 0;

    for (const auto & entry : fs::recursive_directory_iterator(folder_path + sep + "annotations" + sep + "xmls")) {
        auto path = entry.path();
        if (path.extension().compare(".xml") == 0) {

            std::ifstream is(path);
            std::stringstream buffer;
            buffer << is.rdbuf();
            std::string file_data = buffer.str();

            std::string clazz;
            int class_pos = extract_token(file_data, "name", clazz);
            if (class_pos < 0) {
                continue; // skip the file if class is not found     
            }

            if (extract_token(file_data.substr(class_pos + 1), "name", clazz) >= 0) {
                continue; // skip the file if two objects were found
            }

            std::string image_width;
            if (extract_token(file_data, "width", image_width) < 0) {
                continue;
            }

            std::string image_height;
            if (extract_token(file_data, "height", image_height) < 0) {
                continue; 
            }

            std::string image_depth;
            if (extract_token(file_data, "depth", image_depth) < 0 || std::stoi(image_depth) != 3) {
                continue; 
            }

            std::string bbox_xmin;
            if (extract_token(file_data, "xmin", bbox_xmin) < 0) {
                continue; 
            }

            std::string bbox_ymin;
            if (extract_token(file_data, "ymin", bbox_ymin) < 0 ) {
                continue; 
            }

            std::string bbox_xmax;
            if (extract_token(file_data, "xmax", bbox_xmax) < 0) {
                continue; 
            }

            std::string bbox_ymax;
            if (extract_token(file_data, "ymax", bbox_ymax) < 0) {
                continue; 
            }

            std::string image;
            if (extract_token(file_data, "filename", image) < 0) {
                continue; 
            }

            std::string image_file = folder_path + sep + "images" + sep + image;

            auto entry = std::make_tuple(image_file, clazz, image_width, image_height, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax);

            // std::cout << image << "\t" << clazz << "\t" << image_width << "x" << image_height << "\t(" << bbox_xmin << ", " << bbox_ymin << ")\t(" << bbox_xmax << ", " << bbox_ymax << ")\n";

            if (check_file_integrity) {
                cv::Mat img = cv::imread(image_file, cv::IMREAD_COLOR);

                if (!img.empty()) {
                    entries.emplace_back(entry);
                    // rewriting to remove possibly wrong bytes 
                    cv::imwrite(image_file, img);
                } else {
                    std::cout << "corrupted image: " << image_file << "\n";
                }
            } else {
                entries.emplace_back(entry);
            }


            if (clazz.compare("dog") == 0) dog_counter++;

        }

    }
    
    const int N_REGISTERS = entries.size();
    const int training_size = static_cast<int>(N_REGISTERS * split_percentage);
    const int validation_size = N_REGISTERS - training_size;

    std::cout << "Total of images: " << N_REGISTERS << "\n";
    std::cout << "Total of dog images: " << dog_counter << "\n";
    std::cout << "Training size: " << training_size << "\n";
    std::cout << "Validation size " << validation_size << "\n";

    std::shuffle(entries.begin(), entries.end(), random_generator);

    Eigen::Tensor<std::string, 2> training(training_size, 8);
    Eigen::Tensor<std::string, 2> validation(validation_size, 8);

    for (int i = 0; i < training_size; ++i) {
        auto entry = entries[i];
        training(i, 0) = std::get<0>(entry);
        training(i, 1) = std::get<1>(entry);
        training(i, 2) = std::get<2>(entry);
        training(i, 3) = std::get<3>(entry);
        training(i, 4) = std::get<4>(entry);
        training(i, 5) = std::get<5>(entry);
        training(i, 6) = std::get<6>(entry);
        training(i, 7) = std::get<7>(entry);
    }

    for (int i = 0; i < validation_size; ++i) {
        auto entry = entries[i + training_size];
        validation(i, 0) = std::get<0>(entry);
        validation(i, 1) = std::get<1>(entry);
        validation(i, 2) = std::get<2>(entry);
        validation(i, 3) = std::get<3>(entry);
        validation(i, 4) = std::get<4>(entry);
        validation(i, 5) = std::get<5>(entry);
        validation(i, 6) = std::get<6>(entry);
        validation(i, 7) = std::get<7>(entry);
    }

    auto result = std::make_tuple(training, validation);

    return result;
}

template <int _RANK>
void load_tensor_from_file(Tensor<_RANK> &tensor, const std::string& file_path, bool as_row_major = true) {

    std::ifstream file;
    file.open(file_path);

    if (!file) {throw std::invalid_argument("Can't open " + file_path);}

    std::vector<TYPE> vec;
    vec.reserve(tensor.size());
    
    TYPE val;
    for(std::string line; std::getline(file, line); ) {
        std::istringstream in(line);  
        in >> val;      
        vec.push_back(val);
    }

    if (vec.size() != tensor.size()) {throw std::invalid_argument("Illegal number of lines " + std::to_string(vec.size()));}

    auto dimensions = tensor.dimensions();
    if (as_row_major) {
        Eigen::array<Eigen::Index, _RANK> suffle_dims;
        for (int i = 0; i < _RANK; ++i) {
            suffle_dims[i] = _RANK - i - 1;
        }
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

    const TYPE * data = tensor.data();

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
        file << std::fixed << std::setprecision(18) << data[i];
        file << "\n";
    }

    file.close();

}

#endif