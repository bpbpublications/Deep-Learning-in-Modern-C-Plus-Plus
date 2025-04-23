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

#ifndef UTILS_H_
#define UTILS_H_

namespace book
{

    namespace utils
    {

        const struct
        {
            TYPE EPSILON = 1e-7f;
            int PRECISION = 7;
        } GLOBAL;

        template <typename T>
        T clip(T value, T min_value = GLOBAL.EPSILON, T max_value = (T(1.) - GLOBAL.EPSILON))
        {
            if (value < min_value) {
                value = min_value;
            } else if (value > max_value) {
                value = max_value;
            }

            return value;
        }

        template <int _RANK>
        Tensor<_RANK> clip_by_norm(const Tensor<_RANK> &t, TYPE max_norm = TYPE(2.))
        {

            TYPE norm = std::sqrt(((Tensor_0D)((t * t).sum()))(0));
            Tensor<_RANK> result = t;
            if (norm > max_norm) {
                result = result * result.constant(max_norm) / result.constant(norm);
            }

            return result;
        }

        template <typename T>
        Eigen::Tensor<T, 2> one_hot(const Tensor_2D &t)
        {

            const int batch_size = t.dimension(0);
            const int number_classes = t.dimension(1);

            Eigen::Tensor<Eigen::DenseIndex, 1> max_arg_t = t.argmax(1);

            Eigen::Tensor<T, 2> result(batch_size, number_classes);
            result.setConstant(T(0));

            for (int i = 0; i < batch_size; ++i) {
                int index = max_arg_t(i);
                result(i, index) = T(1);
            }

            return std::move(result);
        }

        template <typename T>
        T signal(T val) {
            if (val > T(0)) return T(1);
            else if (val < T(0)) return T(-1);
            else return T(0);
        }
    }
}

template <typename T>
std::string format_time(T time_in_mills) {

    std::string result = std::to_string(time_in_mills) + " milliseconds";

    if (time_in_mills > 3'600'000) {
        auto hours = time_in_mills / 3'600'000;
        auto minutes = (time_in_mills - hours * 3'600'000) / 60'000;
        auto seconds = (time_in_mills - hours * 3'600'000 - minutes * 60'000) / 1'000;
        result = std::to_string(hours) + " hours, " + std::to_string(minutes) + " minutes and " + std::to_string(seconds) + " seconds";
    } else if (time_in_mills > 60'000) {
        auto minutes = time_in_mills / 60'000;
        auto seconds = (time_in_mills - minutes * 60'000) / 1'000;
        result = std::to_string(minutes) + " minutes and " + std::to_string(seconds) + " seconds";
    } else if (time_in_mills > 10'000){
        result = std::to_string(time_in_mills / 1'000) + " seconds";
    }
    return result;
}

#endif