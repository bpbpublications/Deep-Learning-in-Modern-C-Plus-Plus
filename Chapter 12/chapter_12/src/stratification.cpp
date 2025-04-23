#include <iostream>
#include <unordered_map>
#include <queue>
#include <exception>

#include "book/data_io.hpp"

std::random_device rd{};
const auto seed = 1234; // rd();
std::mt19937 rng(seed);

/**
 * function to retrieve the class index of a given instance
 */
Eigen::Index find_class_index(const Tensor_1D &instance, long int num_classes)
{
    long int instance_size = instance.size();
    DimArray<1> offset = {instance_size - num_classes};
    DimArray<1> extents = {num_classes};
    auto label_vec = instance.slice(offset, extents);
    Eigen::Tensor<Eigen::Index, 0> label_index = label_vec.argmax(0);
    auto result = label_index(0);
    return result;
}

/**
 * Class to represent one fold in the cross-validation schema
 */
class Fold
{

public:
    Fold(long int instance_size, long int num_classes, size_t capacity) : num_classes(num_classes), capacity(capacity)
    {
        this->data = Eigen::Tensor<float, 2>(this->capacity, instance_size);
        this->data.setZero();
        this->instance_count = 0;
    }

    void add_instance(const Tensor_1D &instance)
    {
        if (!this->is_full())
        {

            // storing the instance
            data.chip<0>(instance_count) = instance;

            auto label_index = find_class_index(instance, this->num_classes);

            // increment counters
            ++this->counters[label_index];
            ++this->instance_count;
        }
        else
        {
            throw std::runtime_error("attempt to include instance to a full fold.");
        }
    }

    bool is_full() const
    {
        return this->instance_count >= this->capacity;
    }

    const int get_count(int label) const
    {
        int result = 0;
        if (this->counters.find(label) != this->counters.end())
        {
            result = this->counters.at(label);
        }
        return result;
    }

    ~Fold() {}

private:
    long int capacity;
    long int num_classes;

    Eigen::Tensor<float, 2> data;

    size_t instance_count;
    std::unordered_map<int, int> counters;
};

using min_heap = std::priority_queue<Fold *, std::vector<Fold *>, std::function<bool(Fold *, Fold *)>>;

int main(int, char **)
{
    srand(seed);

    auto data = load_mnist_chunck("../../data/mnist", 200, true, seed);
    std::cout << "Data loaded!\n";
    std::cout << "data dimensions: " << data.dimensions() << "\n";

    const long int num_registers = data.dimension(0);
    const long int instance_size = data.dimension(1);
    const long int num_classes = 10;
    const size_t num_folds = 5;

    DimArray<2> offset = {0, instance_size - num_classes};
    DimArray<2> extents = {num_registers, num_classes};
    auto labels = data.slice(offset, extents);

    Tensor_1D labels_total_count = labels.sum(Eigen::array<Eigen::Index, 1>{0});

    std::cout << "\nData distribution:\nLabels:\t";
    for (size_t clazz = 0; clazz < num_classes; ++clazz)
    {
        std::cout << clazz << "\t";
    }
    std::cout << "\n#regs:\t";
    for (size_t clazz = 0; clazz < num_classes; ++clazz)
    {
        std::cout << labels_total_count(clazz) << "\t";
    }
    std::cout << "\n\n";

    const size_t instances_per_fold = static_cast<size_t>(std::round(static_cast<float>(num_registers) / static_cast<float>(num_folds)));

    size_t capacity = num_registers - (num_folds - 1) * instances_per_fold;

    std::vector<Fold> folds;
    folds.reserve(num_folds);

    for (size_t i = 0; i < num_folds; ++i)
    {
        Fold fold(instance_size, num_classes, capacity);
        folds.emplace_back(fold);
        capacity = instances_per_fold;
    }

    std::unordered_map<int, min_heap> heaps;

    for (size_t i = 0; i < num_classes; ++i)
    {

        auto compare = [i](const Fold *a, const Fold *b)
        {
            int count_a = a->get_count(i);
            int count_b = b->get_count(i);
            bool result = count_a > count_b;

            return result;
        };

        auto heap = min_heap(compare);
        for (auto &fold : folds)
        {
            heap.push(&fold);
        }

        heaps[i] = heap;
    }

    size_t row = 0;
    while (row < num_registers)
    {

        auto instance = data.chip<0>(row);

        auto label = find_class_index(instance, num_classes);

        auto &heap = heaps.at(label);

        if (heap.empty())
        {
            throw std::runtime_error("Heap for class " + std::to_string(label) + " is empty!");
        }

        Fold *fold = heap.top();
        heap.pop();

        if (!fold->is_full())
        {
            fold->add_instance(instance);
            ++row;
            heap.push(fold);
        }
    }

    int total_registers = 0;

    for (size_t fold_index = 0; fold_index < num_folds; ++fold_index)
    {

        Fold &fold = folds.at(fold_index);

        std::cout << "Fold #" << fold_index << ":\nLabels:\t";
        for (size_t clazz = 0; clazz < num_classes; ++clazz)
        {
            std::cout << clazz << "\t";
        }
        std::cout << "\n#regs:\t";
        for (size_t clazz = 0; clazz < num_classes; ++clazz)
        {
            int count = fold.get_count(clazz);
            total_registers += count;
            std::cout << count << "\t";
        }

        std::cout << "\n\n";
    }

    std::cout << "Total count of registers in the " << num_folds << " folds: " << total_registers << "\n";

    return 0;
}