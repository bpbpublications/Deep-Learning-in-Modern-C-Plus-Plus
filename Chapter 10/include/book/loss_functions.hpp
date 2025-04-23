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
 *
 */

#ifndef LOSSES_H_
#define LOSSES_H_

class LossFunction
{
public:
    LossFunction(std::string name) : name(std::move(name)) {}
    virtual ~LossFunction() {}

    virtual TYPE evaluate(const Tensor_2D &PRED, const Tensor_2D &TRUE) const = 0;

    virtual Tensor_2D derivative(const Tensor_2D &PRED, const Tensor_2D &TRUE) const = 0;

    virtual TYPE operator()(const Tensor_2D &PRED, const Tensor_2D &TRUE) const
    {
        return this->evaluate(PRED, TRUE);
    }

    const std::string &get_name() const
    {
        return this->name;
    }

protected:
    virtual TYPE loss(const TYPE expected, const TYPE actual) const = 0;
    virtual TYPE cwise_derivative(const TYPE expected, const TYPE actual) const = 0;

private:
    std::string name;
};

class CategoricalCrossEntropy : public LossFunction
{
public:
    CategoricalCrossEntropy() : LossFunction("book.losses.categorical_cross_entropy") {}
    virtual ~CategoricalCrossEntropy() {}

    virtual TYPE evaluate(const Tensor_2D &TRUE, const Tensor_2D &PRED) const
    {
        if (PRED.dimension(0) != TRUE.dimension(0) || PRED.dimension(0) != TRUE.dimension(0)) {
            throw std::invalid_argument("The operands dimensions do not match.");
        }
        Tensor_2D temp = TRUE.binaryExpr(PRED, [this](const TYPE expected, const TYPE actual)
                                                  { return this->loss(expected, actual); });

        const Tensor_0D red = temp.sum();

        int instance_size = TRUE.dimension(1);

        TYPE result = red(0) * instance_size / TRUE.size();
        return result;

    }

    virtual Tensor_2D derivative(const Tensor_2D &TRUE, const Tensor_2D &PRED) const
    {
        const int M = TRUE.dimension(0);
        Tensor_2D result = TRUE.binaryExpr(PRED, [this, &M](const TYPE expected, const TYPE actual)
                                                    { return this->cwise_derivative(expected, actual) / M; });
        return std::move(result);
    }

protected:
    virtual TYPE loss(const TYPE expected, const TYPE actual) const
    {
        TYPE y_true = expected;
        TYPE y_pred = book::utils::clip(actual);
        TYPE result = -y_true * log(y_pred);
        return result;
    }

    virtual TYPE cwise_derivative(const TYPE expected, const TYPE actual) const
    {
        TYPE y_pred = book::utils::clip(actual);
        TYPE result = 1. + -expected / (y_pred);

        return result;
    }
};

class MSE : public LossFunction
{
public:
    MSE() : LossFunction("book.losses.mse") {}
    virtual ~MSE() {}

    virtual float evaluate(const Tensor_2D &TRUE, const Tensor_2D &PRED) const {

        Tensor_2D temp = TRUE.binaryExpr(PRED, [this](const float expected, const float actual)
                                                    { return this->loss(expected, actual); });

        const Tensor_0D summup = temp.sum();
        const int M = TRUE.size();

        float result = summup(0) / M;
        return result;
    }

    virtual Tensor_2D derivative(const Tensor_2D &TRUE, const Tensor_2D &PRED) const
    {
        const int M = TRUE.size();
        Tensor_2D result = TRUE.binaryExpr(PRED, [this, &M](const float expected, const float actual)
                                                    { return this->cwise_derivative(expected, actual) / M; });
        return std::move(result);
    }

protected:
    virtual float loss(const float expected, const float actual) const
    {
        const float diff = expected - actual;
        const float result = diff * diff;
        return result;
    }

    virtual float cwise_derivative(const float expected, const float actual) const
    {
        const float diff = actual - expected;
        const float result = 2 * diff;
        return result;
    }
};

#endif