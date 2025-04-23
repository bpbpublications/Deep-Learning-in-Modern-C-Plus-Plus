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

#ifndef _TRAINING_DOGS_X_CATS_
#define _TRAINING_DOGS_X_CATS_

#include "book/book.hpp"

template <int _RANK>
using Dim_Array = Eigen::array<Eigen::Index, _RANK>;

struct BBOXBatch {

    BBOXBatch(const Dim_Array<4> x_dims, const Dim_Array<2> b_dims) {
        this->X = Tensor<4>(x_dims);
        this->B = Tensor<2>(b_dims);
    }

    BBOXBatch() : BBOXBatch(Dim_Array<4>({0}), Dim_Array<2>({0})) {}

    Tensor<4> X;
    Tensor<2> B;
};

template<typename Generator>
class ParallelBatches {

enum BATCH_STATUS {AVAILABLE, UNAVAILABLE};

public:
    ParallelBatches(Generator& gen, int batch_size, const Eigen::Tensor<std::string, 2> *data, int buffer_size, int image_size, bool center_images): 
        batch_size(batch_size), data(data), image_size(image_size), center_images(center_images), rng(gen) {
        
        this->num_registers = data->dimension(0);
        this->instance_indexes = std::vector<int>(num_registers);
        std::iota(this->instance_indexes.begin(), this->instance_indexes.end(), 0);
        std::shuffle(this->instance_indexes.begin(), this->instance_indexes.end(), gen);

        this->num_batches = std::ceil(static_cast<float>(this->num_registers) / this->batch_size);

        this->index_pointer = 0;
        this->buffer_size = std::min(this->num_batches, buffer_size);
        this->buffer_status = std::vector<BATCH_STATUS>(this->buffer_size, UNAVAILABLE);
        this->buffer = std::vector<BBOXBatch>(this->buffer_size, BBOXBatch());
        this->done = this->index_pointer >= this->num_batches;
        this->write_pointer = 0;
        this->read_pointer = 0;
    }

    // for profile
    long long int waiting_time = 0;

    BBOXBatch* next() {

        BBOXBatch* result = nullptr;
        
        auto begin = high_resolution_clock::now();

        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_cond_var.wait(lock, [&]{ return (this->done || (this->buffer_status[this->read_pointer] == AVAILABLE)); });
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        this->waiting_time += duration.count();
        
        if (this->buffer_status[this->read_pointer] == AVAILABLE) {
            result = &this->buffer[this->read_pointer];
            this->read_pointer = (this->read_pointer + 1) % this->buffer_size;
        }

        lock.unlock();

        return result;
    }

    size_t get_num_batches() const {
        return this->num_batches;
    }

    /**
     * Notifies the ParallelBatch controller that the batch is no longer in use
    */
    void release(BBOXBatch* batch) {
        int index = std::distance(this->buffer.data(), batch);
        this->buffer_status[index] = UNAVAILABLE;
        this->_cond_var.notify_one();
    }

    void load_batch(){
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_cond_var.wait(lock, [&]{ return (this->done || (this->buffer_status[this->write_pointer] == UNAVAILABLE)); });

        if (!this->done) {
            this->load_batch_helper();
        }

        lock.unlock();
        this->_cond_var.notify_one();
    }

    void start() {

        // prefill the buffer
        for (int i = 0; i < this->buffer_size; ++i) {
            load_batch_helper();
        }

        std::thread producer([&]() {
            while(!this->done) {
                load_batch();
            }
        });

        producer.detach(); // turns the thread in a daemon

    }

protected:

    Generator &rng;

    virtual void processing_image(cv::Mat &src, cv::Mat &dest, int &xmin, int &ymin, int &xmax, int &ymax) {
        dest = src;
    }

    void fill_batch(int begin, int end, BBOXBatch &batch) {

        const int batch_size = end - begin;

        batch.X = Tensor_4D(batch_size, this->image_size, this->image_size, 3);
        batch.B = Tensor_2D(batch_size, 4);

        Tensor_3D tensor(this->image_size, this->image_size, 3);
        float scale = 1.f/255.f;
        for (int i = 0; i < batch_size; ++i) {
            int instance = this->instance_indexes[begin + i];

            const std::string &path = (*this->data)(instance, 0);
            cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
            if (src.empty()) {
                throw std::invalid_argument(path + " is not a valid image.");
            }

            int width = std::stoi((*this->data)(instance, 2));
            int height = std::stoi((*this->data)(instance, 3));
            int xmin = std::stoi((*this->data)(instance, 4));
            int ymin = std::stoi((*this->data)(instance, 5));
            int xmax = std::stoi((*this->data)(instance, 6));
            int ymax = std::stoi((*this->data)(instance, 7));

            cv::Mat processed_image, squared;

            if (this->center_images) {
                make_image_square(src, squared, xmin, ymin, xmax, ymax);
                int max_side = std::max(width, height);
                width = max_side;
                height = max_side;
            } else {
                squared = src;
            }

            this->processing_image(squared, processed_image, xmin, ymin, xmax, ymax);

            convert_to_tensor(processed_image, tensor, scale);
            batch.X.chip<0>(i) = tensor;

            batch.B(i, 0) = std::max(TYPE(0.), static_cast<TYPE>(xmin) / width);
            batch.B(i, 1) = std::max(TYPE(0.), static_cast<TYPE>(ymin) / height);
            batch.B(i, 2) = std::min(TYPE(1.), static_cast<TYPE>(xmax) / width);
            batch.B(i, 3) = std::min(TYPE(1.), static_cast<TYPE>(ymax) / height);

        }

    }

private:
    int batch_size;
    const Eigen::Tensor<std::string, 2> *data;
    int image_size;
    
    std::vector<int> instance_indexes;

    int num_batches;
    int num_registers;
    int index_pointer;

    int buffer_size;
    std::vector<BBOXBatch> buffer;
    std::vector<BATCH_STATUS> buffer_status;
    int read_pointer = 0;
    int write_pointer = 0;
    bool done;

    std::mutex _mutex;
    std::condition_variable _cond_var;

    bool center_images;

    void load_batch_helper() {
        int index = this->index_pointer;

        int begin = index * this->batch_size;
        int end = std::min(begin + this->batch_size, this->num_registers);
        BBOXBatch & batch = this->buffer[this->write_pointer];
        this->fill_batch(begin, end, batch);

        this->buffer_status[this->write_pointer] = AVAILABLE;
        this->write_pointer = (this->write_pointer + 1) % this->buffer_size;

        this->index_pointer++;
        this->done = this->index_pointer >= num_batches;
    }

};

/**
 * An asynchronous batching manager class that uses data augmentation
*/
template<typename Generator>
class Data_Augmentation_ParallelBatches : public ParallelBatches<Generator> {

public:
    Data_Augmentation_ParallelBatches(Generator& gen, int batch_size, 
        const Eigen::Tensor<std::string, 2> *data, int buffer_size, int image_size, bool center_images): 
        ParallelBatches<Generator>(gen, batch_size, data, buffer_size, image_size, center_images) {
            this->distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
        }

    virtual void processing_image(cv::Mat &src, cv::Mat &dest, int &xmin, int &ymin, int &xmax, int &ymax) {

        cv::Mat flipped, zoomed, rotated;

        bool flips = distribution(this->rng) > 0.5;

        if (flips) {
            flipped = horizontal_flip(src);
        } else {
            flipped = src;
        }

        float zoomin = 0.75f + distribution(this->rng) * 0.5f;

        zoomed = zoom(flipped, zoomin);

        int angle = static_cast<int>((2.f * distribution(this->rng) - 1.f) * 15);
        int vertical_delta = static_cast<int>((2.f * distribution(this->rng) - 1.f) * src.rows * 0.1);
        int horizontal_delta = static_cast<int>((2.f * distribution(this->rng) - 1.f) * src.cols * 0.1);

        dest = rotate_and_translate(zoomed, angle, horizontal_delta, vertical_delta);

        // updating bounding box

        if (flips) {
            // updaing bounding box if flipped
            int temp = src.cols - xmax;
            xmax = src.cols - xmin;
            xmin = temp;
        }

        // zooming
        float x_med = src.cols / 2;

        xmin = static_cast<int>(x_med - ((x_med - xmin) * zoomin));
        xmax = static_cast<int>(x_med - ((x_med - xmax) * zoomin));

        float y_med = src.rows / 2;

        ymin = static_cast<int>(y_med - ((y_med - ymin) * zoomin));
        ymax = static_cast<int>(y_med - ((y_med - ymax) * zoomin));

        // translating
        xmin += horizontal_delta;
        xmax += horizontal_delta;

        ymin += vertical_delta;
        ymax += vertical_delta;

        // rotating

        if (angle != 0) {

            float c_x = src.cols / 2;
            float c_y = src.rows / 2;

            float angle_rad = M_PI * angle / 180.;

            float cos_a = std::cos(angle_rad);
            float sin_a = std::sin(angle_rad);

            float x1 = c_x + cos_a * (xmin - c_x) - sin_a * (ymin - c_x);
            float y1 = c_y + sin_a * (xmin - c_y) + cos_a * (ymin - c_y);

            float x2 = c_x + cos_a * (xmax - c_x) - sin_a * (ymin - c_x);
            float y2 = c_y + sin_a * (xmax - c_y) + cos_a * (ymin - c_y);

            float x3 = c_x + cos_a * (xmax - c_x) - sin_a * (ymax - c_x);
            float y3 = c_y + sin_a * (xmax - c_y) + cos_a * (ymax - c_y);

            float x4 = c_x + cos_a * (xmin - c_x) - sin_a * (ymax - c_x);
            float y4 = c_y + sin_a * (xmin - c_y) + cos_a * (ymax - c_y);

            // obtaining the new bounding box parallel to image sides

            xmin = static_cast<int>(std::min(x1, x4));
            xmax = static_cast<int>(std::max(x2, x3));
            ymin = static_cast<int>(std::min(y1, y2));
            ymax = static_cast<int>(std::max(y3, y4));
        }

    }

private:
    std::uniform_real_distribution<float> distribution;

};

/**
 * A simple function to inspect the batch images. Useful for debugging only.
*/
template <typename Device>
void visualize(const BBOXBatch * batch) {

    int instance = 0;

    char key = 0;

    const char * title = "Oxford Pets";

    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);

    const int number_instances = batch->X.dimension(0);

    while(key != 27 && key >= 0)
    {
        Eigen::Tensor<float, 3> image_tensor = batch->X.chip<0>(instance);
        Eigen::Tensor<float, 3> image_tensor_255 = image_tensor * image_tensor.constant(255.f);

        Eigen::Tensor<float, 1> bbox_tensor = batch->B.chip<0>(instance);
        
        cv::Mat image;
        cv::eigen2cv(image_tensor_255, image);
        cv::Mat dest;
        image.convertTo(dest, CV_8U, 1);

        int x1 = static_cast<int>(bbox_tensor(0) * image.cols);
        int y1 = static_cast<int>(bbox_tensor(1) * image.rows);
        int x2 = static_cast<int>(bbox_tensor(2) * image.cols);
        int y2 = static_cast<int>(bbox_tensor(3) * image.rows);
        cv::rectangle(dest, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        cv::imshow(title, dest);

        do {
            key = cv::waitKey(0);
        } while(key != 'a' && key != 'd' && key != 27);

        if(key == 'd' && instance < (number_instances - 1)) instance++;
        if(key == 'a' && instance > 0) instance--;
    }

}

template <typename MODEL>
void check(const BBOXBatch * batch, MODEL &model) {

    int instance = 0;

    char key = 0;

    const char * title = "Oxford Pets";

    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);

    const int number_instances = batch->X.dimension(0);

    while(key != 27 && key >= 0)
    {
        Tensor_3D image_tensor = batch->X.chip<0>(instance);
        Tensor_3D image_tensor_255 = image_tensor * image_tensor.constant(255.f);

        Dim_Array<4> four_dims{{1, image_tensor.dimension(0), image_tensor.dimension(1), image_tensor.dimension(2)}};
        Tensor_4D input = image_tensor.reshape(four_dims);
        auto bbox_pred = model.predict(input);
        
        Tensor_1D bbox_ground_truth = batch->B.chip<0>(instance);
        
        cv::Mat image;
        cv::eigen2cv(image_tensor_255, image);

        cv::Mat dest;
        image.convertTo(dest, CV_8U, 1);

        int gt_x1 = static_cast<int>(bbox_ground_truth(0) * image.cols);
        int gt_y1 = static_cast<int>(bbox_ground_truth(1) * image.rows);
        int gt_x2 = static_cast<int>(bbox_ground_truth(2) * image.cols);
        int gt_y2 = static_cast<int>(bbox_ground_truth(3) * image.rows);
        cv::rectangle(dest, cv::Point(gt_x1, gt_y1), cv::Point(gt_x2, gt_y2), cv::Scalar(0, 255, 0), 2);

        int pred_x1 = static_cast<int>(bbox_pred(0, 0) * image.cols);
        int pred_y1 = static_cast<int>(bbox_pred(0, 1) * image.rows);
        int pred_x2 = static_cast<int>(bbox_pred(0, 2) * image.cols);
        int pred_y2 = static_cast<int>(bbox_pred(0, 3) * image.rows);
        cv::rectangle(dest, cv::Point(pred_x1, pred_y1), cv::Point(pred_x2, pred_y2), cv::Scalar(255, 0, 0), 2);

        cv::imshow(title, dest);

        do {
            key = cv::waitKey(0);
        } while(key != 'a' && key != 'd' && key != 27);

        if(key == 'd' && instance < (number_instances - 1)) instance++;
        if(key == 'a' && instance > 0) instance--;
    }

}

template <typename Device, typename GEN, class MODEL, class BATCH_MANAGER>
void training(MODEL &model, GEN &rng, const Eigen::Tensor<std::string, 2> &training_data, 
              const Eigen::Tensor<std::string, 2> &validation_data, 
              Device &device, const int max_epochs, 
              const int minibatch_size, const TYPE learning_rate,
              const bool center_images, const int batch_buffer_size) {

    CategoricalCrossEntropy classifier_cost_fn;
    MSE regressor_cost_fn;

    const int IMG_SIZE = model.get_IMG_SIZE();

    const std::int64_t start_ts = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();

    int epoch = 0;
    while (epoch < max_epochs) {

        auto begin = high_resolution_clock::now();

        int steps = 0;

        BATCH_MANAGER batches(rng, minibatch_size, &training_data, batch_buffer_size, IMG_SIZE, center_images);

        batches.start();

        BBOXBatch* batch = batches.next();

        int training_success = 0;
        TYPE regression_training_loss = 0;
        int instances_count = 0;
        const int TRAINING_SIZE = training_data.dimension(0);

        size_t num_batches = batches.get_num_batches();

        while (batch) {

            // visualize<Device>(batch);

            std::string temp_update_desc = "Step: " + std::to_string(steps) + "/" + std::to_string(num_batches);

            if (steps > 0) {
                auto step_end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(step_end - begin).count();
                auto estimate = (num_batches - steps) * duration / steps;
                temp_update_desc = temp_update_desc + " remaining time: " + format_time(estimate);
                TYPE temp_acc = training_success * 100.0 / instances_count;
                temp_update_desc = temp_update_desc + " regress train loss " + std::to_string(regression_training_loss / steps);
            }

            std::cout << temp_update_desc << std::flush;

            // training loop
            model.forward(batch->X);
            model.backward(batch->B);
            model.update(learning_rate, epoch + 1);

            // computing batch metrics
            auto regression_pred = model.get_boundingbox_prediction();
            regression_training_loss += regressor_cost_fn.evaluate(batch->B, regression_pred);

            instances_count += batch->X.dimension(0);

            batches.release(batch);

            steps++;
            batch = batches.next();

            auto b_count = temp_update_desc.length();

            for (int i = 0; i < b_count; ++i) {
                std::cout << "\b \b";
            }
            std::cout << std::flush;
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        TYPE training_acc = (training_success * 10000 / TRAINING_SIZE) / 100.;
        regression_training_loss /= steps;

        ParallelBatches validation_batches(rng, minibatch_size, &validation_data, batch_buffer_size, IMG_SIZE, center_images);

        validation_batches.start();

        BBOXBatch* val_batch = validation_batches.next();

        int validation_success = 0;
        TYPE regression_validation_loss = 0;
        int val_steps = 0;
        while (val_batch) {
            
            auto regression = model.predict(val_batch->X);
            regression_validation_loss += regressor_cost_fn.evaluate(val_batch->B, regression);
            val_steps++;
            validation_batches.release(val_batch);
            val_batch = validation_batches.next();
        }

        regression_validation_loss /= val_steps;

        std::cout 
                << "epoch:\t" << epoch << "\t"
                << "took:\t" << format_time(duration.count()) << "\t"
                << "\ttraining loss:\t" << regression_training_loss
                << "\tvalidation loss:\t" << regression_validation_loss
                << "\n";

        epoch++;

    }

    ParallelBatches test_batches(rng, validation_data.dimension(0), &validation_data, 1, IMG_SIZE, center_images);
    test_batches.start();
    BBOXBatch* test_batch = test_batches.next();
    check(test_batch, model);    
    test_batches.release(test_batch);

}

template <typename Device, typename GEN, class MODEL, class BATCH_MANAGER>
void run_experiment(const std::string &datapath, MODEL &model, Device &device, GEN &rng, 
    const int max_epochs = 5,
    const int minibatch_size = 16, 
    const TYPE learning_rate = 0.001,
    bool center_images = false,
    const int batch_buffer_size = 3) {
    
    std::cout << "Loading the data, please wait!\n";
    auto [training_data, validation_data] = load_oxford_pets_dataset(datapath, rng, 0.8);
    std::cout << "Data loaded!\n";

    std::cout << "Using image size " << model.get_IMG_SIZE() << "x" << model.get_IMG_SIZE() << "\n";

    std::cout << "Training for " << max_epochs << " epochs\n";

    training<Device, GEN, MODEL, BATCH_MANAGER>(model, rng, training_data, validation_data, device, max_epochs, minibatch_size, learning_rate, center_images, batch_buffer_size);

}

#endif