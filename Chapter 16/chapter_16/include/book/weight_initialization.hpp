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

#ifndef _WEIGTH_INITIALIZATION_
#define _WEIGTH_INITIALIZATION_

#include <unsupported/Eigen/CXX11/Tensor>

template<typename Generator>
void glorot_uniform_initializer(Generator& gen, Tensor_4D &tensor)
{

    const int kernel_height = tensor.dimension(0);
    const int kernel_width = tensor.dimension(1);
    const int receptive_field_size = kernel_height * kernel_width;
    const int input_channels = tensor.dimension(2);
    const int output_channels = tensor.dimension(3);
    const int fan_in = input_channels * receptive_field_size;
    const int fan_out = output_channels * receptive_field_size;

    TYPE scale = std::max(TYPE(1.), (fan_in + fan_out) / TYPE(2.));
    TYPE range = std::sqrt(TYPE(3.) / scale);

    std::uniform_real_distribution<TYPE> uniform_distro(-range, range);
    tensor = tensor.unaryExpr([&uniform_distro, &gen](TYPE)
                              { return uniform_distro(gen); });
}

template<typename Generator>
void glorot_uniform_initializer(Generator& gen, Tensor_2D &tensor)
{
    const int fan_in = tensor.dimension(0);
    const int fan_out = tensor.dimension(1);

    TYPE scale = std::max(TYPE(1.), (fan_in + fan_out) / TYPE(2.));
    TYPE range = std::sqrt(TYPE(3.) / scale);

    std::uniform_real_distribution<TYPE> uniform_distro(-range, range);
    tensor = tensor.unaryExpr([&uniform_distro, &gen](TYPE)
                              { return uniform_distro(gen); });
}

template<typename Generator>
void he_uniform_initializer(Generator& gen, Tensor_4D &tensor)
{

    const int kernel_height = tensor.dimension(0);
    const int kernel_width = tensor.dimension(1);
    const int receptive_field_size = kernel_height * kernel_width;
    const int input_channels = tensor.dimension(2);
    const int output_channels = tensor.dimension(3);
    const TYPE fan_in = input_channels * receptive_field_size;

    TYPE scale = std::max(TYPE(1.), fan_in);
    TYPE range = std::sqrt(TYPE(3.) / scale);

    std::uniform_real_distribution<TYPE> uniform_distro(-range, range);
    tensor = tensor.unaryExpr([&uniform_distro, &gen](TYPE)
                              { return uniform_distro(gen); });
}

template<typename Generator>
void he_uniform_initializer(Generator& gen, Tensor_2D &tensor)
{
    const int fan_in = tensor.dimension(0);

    TYPE scale = std::max(TYPE(1.), static_cast<TYPE>(fan_in));
    TYPE range = std::sqrt(TYPE(3.) / scale);

    std::uniform_real_distribution<TYPE> uniform_distro(-range, range);
    tensor = tensor.unaryExpr([&uniform_distro, &gen](TYPE)
                              { return uniform_distro(gen); });
}

#endif