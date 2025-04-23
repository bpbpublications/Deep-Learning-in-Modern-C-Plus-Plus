#ifndef IMAGE_IO_TUTORIAL_H_
#define IMAGE_IO_TUTORIAL_H_

#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "book/definitions.hpp"

cv::Mat resize_image(const cv::Mat &image, int target_rows, int target_cols)
{
    const int image_rows = image.rows;
    const int image_cols = image.cols;

    int new_rows = 0;
    int new_cols = 0;

    if (image_rows > image_cols) {
        new_rows = target_rows;
        new_cols = image_cols * target_rows / image_rows;
    } else {
        new_cols = target_cols;
        new_rows = image_rows * target_cols / image_cols;
    }
    cv::Mat resized;
    resize(image, resized, cv::Size(new_cols, new_rows), cv::INTER_LINEAR);

    cv::Mat result = cv::Mat::zeros(cv::Size(target_cols, target_rows), CV_8UC1);

    resized.copyTo(result(cv::Rect((target_cols - new_cols)/2, (target_rows - new_rows)/2, resized.cols, resized.rows)));

    return result;
}

auto load_dataset(const std::string data_folder, const Tensor_2D &Gen, const int image_size) {

    std::vector<std::string> files;

    for (const auto & entry : fs::directory_iterator(data_folder)) {
        files.push_back(data_folder + entry.path().c_str());
    }

    Tensor_3D X(files.size(), image_size, image_size);
    Tensor_3D T(files.size(), image_size, image_size);

    const DimArray<3> extent = {1, image_size, image_size};

    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(1, 1);
    padding[1] = std::make_pair(1, 1);
    padding[2] = std::make_pair(0, 0);

    for (unsigned int i = 0; i < files.size(); ++i) {
        const auto & file = files[i];
        cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
        cv::Mat formatted_image = resize_image(image, image_size, image_size);
        cv::Mat frame32f;
        formatted_image.convertTo(frame32f, CV_32F);
        frame32f /= 255.f;
        
        Tensor_3D eigen_frame(image_size, image_size, 1);
        cv::cv2eigen(frame32f, eigen_frame);

        Tensor_3D convolved(image_size, image_size, 1);
        Eigen::array<int, 2> dims({0, 1});
        convolved = eigen_frame.pad(padding).convolve(Gen, dims);

        DimArray<3> offset = {i, 0, 0};
        DimArray<3> new_dim({image_size, image_size, 1});
        X.slice(offset, extent) = eigen_frame.reshape(extent);
        T.slice(offset, extent) = convolved.reshape(extent);

    }

    return std::make_tuple(X, T);

};

#endif