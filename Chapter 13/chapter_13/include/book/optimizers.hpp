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

#ifndef _OPTIMIZERS_
#define _OPTIMIZERS_

#include <unsupported/Eigen/CXX11/Tensor>

template <int _RANK>
class DEFAULT_UPDATE {

public:  

    Tensor<_RANK> operator()(Tensor<_RANK> &grad, TYPE learning_rate, int epoch) {
        Tensor<_RANK> result = grad * grad.constant(-learning_rate);
        return result;
    }

};

template <int _RANK>
class Momentum {

public:  

    Momentum(TYPE momentum): momentum(momentum) { }
    Momentum(): Momentum(0.9) { }

    Tensor<_RANK> operator()(Tensor<_RANK> &grad, TYPE learning_rate, int epoch) {
        if (this->velocity.size() != grad.size()) {
            this->velocity = grad.constant(0);
        }
        this->velocity = this->velocity * grad.constant(this->momentum) + grad * grad.constant(-learning_rate);
        return this->velocity;
    }

private:
    TYPE momentum;
    Tensor<_RANK> velocity;
};

template <int _RANK>
class RMSProp {

public:  

    RMSProp(TYPE beta): beta(beta) { }
    RMSProp(): RMSProp(0.9) { }

    Tensor<_RANK> operator()(Tensor<_RANK> &grad, TYPE learning_rate, int epoch) {
        if (this->S.size() != grad.size()) {
            this->S = grad.constant(0);
        }
        this->S = this->S * grad.constant(this->beta) + grad * grad * grad.constant(1.f - this->beta);
        Tensor<_RANK> result = grad * (this->S + this->S.constant(1e-5)).rsqrt() * grad.constant(-learning_rate);
        return std::move(result);
    }

private:
    TYPE beta;
    Tensor<_RANK> S;
};

template <int _RANK>
class Adam {

public:  

    Adam(TYPE beta1, TYPE beta2): beta1(beta1), beta2(beta2) { }
    Adam(): Adam(0.9, 0.999) { }

    Tensor<_RANK> operator()(Tensor<_RANK> &grad, TYPE learning_rate, int epoch) {
        if (this->V.size() != grad.size()) {
            this->V = grad.constant(0);
        }
        if (this->S.size() != grad.size()) {
            this->S = grad.constant(0);
        }
        if (epoch <= 0) {
            epoch = 1; // avoiding nan
        }
        
        // avoiding numeric instability
        TYPE beta_1_power = std::pow(this->beta1, epoch);
        TYPE beta_2_power = std::pow(this->beta2, epoch);
        TYPE correction = learning_rate * std::sqrt(1. - beta_2_power) / (1. - beta_1_power);

        this->V = this->V + (grad - this->V) * grad.constant(1. - this->beta1);
        this->S = this->S + ((grad * grad) - this->S) * grad.constant(1. - this->beta2);
        
        Tensor<_RANK> result = (this->V * grad.constant(-correction)) / (this->S.sqrt() + this->S.constant(1e-7));
        return std::move(result);
    }

private:
    TYPE beta1;
    TYPE beta2;
    Tensor<_RANK> V;
    Tensor<_RANK> S;
};

#endif