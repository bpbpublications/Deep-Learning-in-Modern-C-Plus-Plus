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

template<typename Generator>
class ParallelBatches {

enum BATCH_STATUS {AVAILABLE, UNAVAILABLE};

public:
    ParallelBatches(Generator& gen, int batch_size, const Eigen::Tensor<std::string, 2> *data, int number_of_classes, int buffer_size, int image_size, bool center_images): 
        batch_size(batch_size), data(data), number_of_classes(number_of_classes), image_size(image_size), center_images(center_images), rng(gen) {
        
        this->num_registers = data->dimension(0);
        this->instance_indexes = std::vector<int>(num_registers);
        std::iota(this->instance_indexes.begin(), this->instance_indexes.end(), 0);
        std::shuffle(this->instance_indexes.begin(), this->instance_indexes.end(), gen);

        this->num_batches = std::ceil(static_cast<float>(this->num_registers) / this->batch_size);

        this->index_pointer = 0;
        this->buffer_size = std::min(this->num_batches, buffer_size);
        this->buffer_status = std::vector<BATCH_STATUS>(this->buffer_size, UNAVAILABLE);
        this->buffer = std::vector<Batch<4,2>>(this->buffer_size, Batch<4,2>());
        this->done = this->index_pointer >= this->num_batches;
        this->write_pointer = 0;
        this->read_pointer = 0;
    }

    long long int waiting_time = 0;

    Batch<4,2>* next() {

        Batch<4,2>* result = nullptr;
        
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

    void release(Batch<4,2>* batch) {
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

    virtual void processing_image(cv::Mat &src, cv::Mat &dest) {
        dest = src;
    }

    void fill_batch(int begin, int end, Batch<4,2> &batch) {

        const int batch_size = end - begin;

        batch.X = Tensor_4D(batch_size, this->image_size, this->image_size, 3);
        batch.T = Tensor_2D(batch_size, this->number_of_classes);
        batch.T.setConstant(0);

        Tensor_3D tensor(this->image_size, this->image_size, 3);
        float scale = 1.f/255.f;
        for (int i = 0; i < batch_size; ++i) {
            int instance = this->instance_indexes[begin + i];

            const std::string &path = (*this->data)(instance, 0);
            cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
            if (src.empty()) {
                throw std::invalid_argument(path + " is not a valid image.");
            }

            cv::Mat processed_image, squared;

            if (this->center_images) {
                make_image_square(src, squared);
            } else {
                squared = src;
            }

            this->processing_image(squared, processed_image);

            convert_to_tensor(processed_image, tensor, scale);
            batch.X.chip<0>(i) = tensor;

            const std::string &clazz = (*this->data)(instance, 1);
            int index = std::stoi(clazz);
            batch.T(i, index) = TYPE(1.);

        }

    }

private:
    int batch_size;
    const Eigen::Tensor<std::string, 2> *data;
    const int number_of_classes;
    int image_size;
    
    std::vector<int> instance_indexes;

    int num_batches;
    int num_registers;
    int index_pointer;

    int buffer_size;
    std::vector<Batch<4,2>> buffer;
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
        Batch<4,2> & batch = this->buffer[this->write_pointer];
        this->fill_batch(begin, end, batch);

        this->buffer_status[this->write_pointer] = AVAILABLE;
        this->write_pointer = (this->write_pointer + 1) % this->buffer_size;

        this->index_pointer++;
        this->done = this->index_pointer >= num_batches;
    }

};

template<typename Generator>
class Data_Augmentation_ParallelBatches : public ParallelBatches<Generator> {

public:
    Data_Augmentation_ParallelBatches(Generator& gen, int batch_size, 
        const Eigen::Tensor<std::string, 2> *data, int number_of_classes, int buffer_size, int image_size, bool center_images): 
        ParallelBatches<Generator>(gen, batch_size, data, number_of_classes, buffer_size, image_size, center_images) {
            this->distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
        }

    virtual void processing_image(cv::Mat &src, cv::Mat &dest) {

        cv::Mat flipped, rotated;

        if (distribution(this->rng) > 0.5) {
            flipped = horizontal_flip(src);
        } else {
            flipped = src;
        }

        int angle = static_cast<int>((2.f * distribution(this->rng) - 1.f) * 15);
        int vertical_delta = static_cast<int>((2.f * distribution(this->rng) - 1.f) * src.rows * 0.1);
        int horizontal_delta = static_cast<int>((2.f * distribution(this->rng) - 1.f) * src.cols * 0.1);

        dest = rotate_and_translate(flipped, angle, horizontal_delta, vertical_delta);
    }

private:
    std::uniform_real_distribution<float> distribution;

};

template <typename Device>
void visualize(const Batch<4, 2> * batch) {

    int instance = 0;

    char key = 0;

    const char * title = "Instance";

    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);

    const int number_instances = batch->X.dimension(0);

    const int number_of_classes = batch->T.dimension(1);

    Eigen::array<Eigen::Index, 2> extent_label = {1, number_of_classes};
    while(key != 27 && key >= 0)
    {
        Eigen::array<Eigen::Index, 2> offset_label = {instance, 0};
        Eigen::Tensor<float, 3> image_tensor = batch->X.chip<0>(instance);
        Eigen::Tensor<float, 3> image_tensor_255 = image_tensor * image_tensor.constant(255.f);
        Eigen::Tensor<float, 2> label_tensor = batch->T.slice(offset_label, extent_label);
        
        cv::Mat image_rgb, image;
        cv::eigen2cv(image_tensor_255, image_rgb);
        cv::cvtColor(image_rgb, image, cv::COLOR_RGB2BGR);
        Eigen::Index label = ((Eigen::Tensor<Eigen::Index, 0>)label_tensor.argmax())(0);

        std::cout << "======================================\n";
        std::cout << "instance: " << instance << "\tLabel: " << label << "\n";
        cv::Mat dest;
        image.convertTo(dest, CV_8U, 1);
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
              Device &device, 
              const int max_epochs, 
              const int minibatch_size, 
              const TYPE learning_rate,
              const bool center_images, 
              const std::string &path_to_store_model,
              const int batch_buffer_size) {

    CategoricalCrossEntropy cost_fn;

    const int IMG_SIZE = model.get_IMG_SIZE();
    const int NUMBER_OF_CLASSES = model.get_number_of_classes();

    const std::int64_t start_ts = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();

    int epoch = 0;
    while (epoch < max_epochs) {

        auto begin = high_resolution_clock::now();

        int steps = 0;

        BATCH_MANAGER batches(rng, minibatch_size, &training_data, NUMBER_OF_CLASSES, batch_buffer_size, IMG_SIZE, center_images);

        batches.start();

        Batch<4, 2>* batch = batches.next();

        int training_success = 0;
        TYPE training_loss = 0;
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
                temp_update_desc = temp_update_desc + " training acc " + std::to_string(temp_acc) + "%";
                temp_update_desc = temp_update_desc + " training loss " + std::to_string(training_loss / steps);
            }

            std::cout << temp_update_desc << std::flush;

            // training loop
            model.forward(batch->X);
            model.backward(batch->T);
            model.update(learning_rate, epoch + 1);

            // computing batch metrics
            auto output = model.get_output();
            training_success += count_success(batch->T, output);
            training_loss += cost_fn.evaluate(batch->T, output);
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
        training_loss /= steps;

        ParallelBatches validation_batches(rng, minibatch_size, &validation_data, NUMBER_OF_CLASSES, batch_buffer_size, IMG_SIZE, center_images);

        validation_batches.start();

        Batch<4, 2>* val_batch = validation_batches.next();

        int validation_success = 0;
        TYPE validation_loss = 0;
        int val_steps = 0;
        while (val_batch) {
            
            auto output = model.predict(val_batch->X);
            validation_success += count_success(val_batch->T, output);
            validation_loss += cost_fn.evaluate(val_batch->T, output);
            val_steps++;
            validation_batches.release(val_batch);
            val_batch = validation_batches.next();
        }

        TYPE validation_acc = (validation_success * 10000 / validation_data.dimension(0)) / 100.;
        validation_loss /= val_steps;

        std::cout 
                << "epoch:\t" << epoch << "\t"
                << "took:\t" << format_time(duration.count()) << "\t"
                << "IO time:\t" << format_time(batches.waiting_time) << " \t"
                << "\ttraining_loss:\t" << training_loss
                << "\ttraining_acc:\t" << training_acc
                << "\tvalidation_loss:\t" << validation_loss
                << "\tvalidation_acc:\t" << validation_acc
                << "\n";
        
        if (!path_to_store_model.empty()) {
            model.save(path_to_store_model + std::to_string(start_ts) + "/" + std::to_string(epoch) + "/", false, true);
        }

        epoch++;

    }

    if (!path_to_store_model.empty()) {
        std::cout << "Model saved at " << path_to_store_model << start_ts << "\n";
    }

}

template <typename Device, typename GEN, class MODEL, class BATCH_MANAGER>
void run_experiment(const Eigen::Tensor<std::string, 2> &training_data,  const Eigen::Tensor<std::string, 2> &validation_data, 
                        MODEL &model, Device &device, GEN &rng, 
                        const int max_epochs = 40,
                        const int minibatch_size = 32, 
                        const TYPE learning_rate = 0.001,
                        bool center_images = false,
                        const std::string &path_to_store_model = "",
                        const int batch_buffer_size = 3) {

    std::cout << "Using image size " << model.get_IMG_SIZE() << "x" << model.get_IMG_SIZE() << "\n";

    std::cout << "Training for " << max_epochs << " epochs\n";

    training<Device, GEN, MODEL, BATCH_MANAGER>(model, rng, training_data, validation_data, device, max_epochs, minibatch_size, learning_rate, center_images, path_to_store_model, batch_buffer_size);
}

#endif