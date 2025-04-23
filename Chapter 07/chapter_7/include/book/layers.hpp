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

#ifndef LAYERS_H_
#define LAYERS_H_

class Dense
{
public:
    Dense(Tensor_2D weights, ActivationFunction<2> *activation) : activation(activation), weights(std::move(weights)) {}
    virtual ~Dense() {}

    virtual Tensor_2D forward(Tensor_2D &X) 
    {
        this->X = X;
        int X_dims = this->X.NumDimensions;
        Eigen::array<Eigen::IndexPair<int>, 1> op_dims = {Eigen::IndexPair<int>(X_dims - 1, 0)};
        this->Z = this->X.contract(this->weights, op_dims);
        this->T = this->activation->evaluate(this->Z);
        return this->T;
    }

    virtual void backward(Tensor_2D &dC_dT)  {

        const int output_size = dC_dT.dimension(1);
        const int input_size = this->X.dimension(1);
        Tensor_2D init(input_size, output_size);
        init.setZero();

        const Eigen::array<Eigen::Index, 2> input_extent = {1, input_size}; 
        const Eigen::array<Eigen::Index, 2> output_extent = {1, output_size};
        const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
        const Eigen::array<Eigen::IndexPair<int>, 1> t_product_dims = {Eigen::IndexPair<int>(0, 0)};

        const Eigen::array<Eigen::Index, 2> dTdZ_i_dim = {output_size, output_size};

        auto gradient_batch = [this, &dC_dT, &input_extent, &output_extent, &product_dims, &t_product_dims, &dTdZ_i_dim] (const Tensor_2D &acc, int index) {

            Eigen::array<Eigen::Index, 2> offset = {index, 0};

            const auto X_i = this->X.slice(offset, input_extent);
            const auto Z_i = this->Z.slice(offset, output_extent);
            const Tensor_2D dCdT_i = dC_dT.slice(offset, output_extent);
            const Tensor_2D dTdZ_i = this->activation->jacobian(Z_i).reshape(dTdZ_i_dim);

            const auto dCdZ = dCdT_i.contract(dTdZ_i, product_dims);
            const auto dCdW = X_i.contract(dCdZ, t_product_dims);

            Tensor_2D result = acc + dCdW.eval();

            return result; 

        };

        const int batch_size = dC_dT.size() / output_size;

        std::vector<int> batch(batch_size);
        std::iota(batch.begin(), batch.end(), 0);

        Tensor_2D result = std::accumulate(batch.begin(), batch.end(), init, gradient_batch);

        this->weights_grad = result / result.constant(batch_size);
    }

    virtual Tensor_2D predict(Tensor_2D &input) const
    {
        int X_dims = input.NumDimensions;
        Eigen::array<Eigen::IndexPair<int>, 1> op_dims = {Eigen::IndexPair<int>(X_dims - 1, 0)};
        auto ZZ = input.contract(this->weights, op_dims);
        auto out = this->activation->evaluate(ZZ);
        return out;
    }

    virtual void update_state(float learning_rate)
    {
        auto update = this->weights_grad * learning_rate;
        this->weights = this->weights - update;
    }

    virtual const Tensor_2D& get_weights()
    {
        return this->weights;
    }

    virtual const Tensor_2D& get_weights_grad()
    {
        return this->weights_grad;
    }

private:

    ActivationFunction<2> *activation;
    Tensor_2D weights;
    Tensor_2D weights_grad;
    Tensor_2D X;
    Tensor_2D Z;
    Tensor_2D T;
};

#endif