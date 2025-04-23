/*
 * This file is part of Coding Deep Learning from Scratch Book, BPB PUBLICATIONS .
 *
 * Author: Luiz doleron <doleron@gmail.com>
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef _DEFINITIONS_
#define _DEFINITIONS_

#include <string>
#include <exception>
#include <iomanip>

#include <unsupported/Eigen/CXX11/Tensor>

using TYPE = float;

using Tensor_0D = Eigen::Tensor<TYPE, 0>;
using Tensor_1D = Eigen::Tensor<TYPE, 1>;
using Tensor_2D = Eigen::Tensor<TYPE, 2>;
using Tensor_3D = Eigen::Tensor<TYPE, 3>;
using Tensor_4D = Eigen::Tensor<TYPE, 4>;

template<int _RANK>
using Tensor = Eigen::Tensor<TYPE, _RANK>;

template<int _RANK>
using DimArray = Eigen::array<Eigen::DenseIndex, _RANK>;

#endif