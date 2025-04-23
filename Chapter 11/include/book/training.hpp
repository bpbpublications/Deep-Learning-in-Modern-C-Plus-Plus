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

#ifndef _TRAINING_
#define _TRAINING_

#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>
using namespace std::chrono;

#include <unsupported/Eigen/CXX11/Tensor>

#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"

template <int X_RANK, int T_RANK>
struct Batch {

    Batch(const Eigen::array<Eigen::Index, X_RANK> x_dims, const Eigen::array<Eigen::Index, T_RANK> t_dims) {
        this->X = Tensor<X_RANK>(x_dims);
        this->T = Tensor<T_RANK>(t_dims);
    }

    Batch() : Batch(Eigen::array<Eigen::Index, X_RANK>({0}), Eigen::array<Eigen::Index, T_RANK>({0})) {}

    Tensor<X_RANK> X;
    Tensor<T_RANK> T;
};

/**
 * A synchronous batch controller for 2-dim tensors
 * 
 * FIXME: make it work with arbitrary rank tensors
 * 
*/
template<typename Generator>
class Batches {

public:
    Batches(Generator& gen, int batch_size, const Tensor_2D *X, const Tensor_2D *T): batch_size(batch_size), X(X), T(T) {
        
        this->num_registers = X->dimension(0);
        int num_batches = std::ceil(static_cast<float>(this->num_registers) / this->batch_size);
        this->indexes = std::vector<int>(num_batches);
        std::iota(this->indexes.begin(), this->indexes.end(), 0);
        std::shuffle(this->indexes.begin(), this->indexes.end(), gen);
        this->pointer = 0;
    }

    Batch<2,2>* next() {
        Batch<2,2>* result = nullptr;
        if (this->pointer < this->indexes.size()) {
            int index = this->indexes.at(this->pointer);
            int begin = index * this->batch_size;
            int end = std::min(begin + this->batch_size, this->num_registers);
            int length = end - begin;
            Eigen::array<Eigen::Index, 2> x_extent = {length, this->X->dimension(1)};
            Eigen::array<Eigen::Index, 2> t_extent = {length, this->T->dimension(1)};
            Eigen::array<Eigen::Index, 2> offset = {begin, 0};
            result = &this->batch;
            if (this->batch.X.size() != (length * this->X->dimension(1))) {
                this->batch.X = Tensor_2D(length, this->X->dimension(1));
                this->batch.T = Tensor_2D(length, this->T->dimension(1));
            }
            this->batch.X = this->X->slice(offset, x_extent);
            this->batch.T = this->T->slice(offset, t_extent);
            this->pointer++;
        }
        return result;
    }

private:
    int batch_size;
    const Tensor_2D *X;
    const Tensor_2D *T;

    int num_registers;
    std::vector<int> indexes;
    int pointer;

    Batch<2,2> batch;

};

#endif