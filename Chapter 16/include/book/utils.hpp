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