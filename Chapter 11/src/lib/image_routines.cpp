#include "book/definitions.hpp"
#include "book/image_routines.hpp"

void convert_to_tensor(cv::Mat &mat, Eigen::Tensor<float, 3> &tensor, float scale) {

    if (mat.channels() != tensor.dimension(2)) {
        throw std::invalid_argument("Failed to convert cv::Mat to Eigen::Tensor: number of channels do not match.");
    }

    cv::Mat processed_image, image_32f, dest;
    mat.convertTo(image_32f, CV_32F, scale);
    cv::resize(image_32f, dest, cv::Size(tensor.dimension(1), tensor.dimension(0)), cv::INTER_LINEAR);

    Eigen::array<Eigen::Index, 3> suffle_dims {2, 1, 0};
    float *mat_data = (float *) dest.data;
    auto mapped = Eigen::TensorMap<Eigen::Tensor<TYPE, 3, Eigen::RowMajor>>(mat_data, tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));
    tensor = Eigen::TensorLayoutSwapOp<Eigen::Tensor<TYPE, 3, Eigen::RowMajor>>(mapped).shuffle(suffle_dims);
}

void make_image_square(cv::Mat &src, cv::Mat &dest, int &x1, int &y1, int &x2, int &y2) {
    
    const int max_size = std::max(src.cols, src.rows);
    
    dest = cv::Mat(cv::Size(max_size, max_size), CV_8UC3, cv::Scalar(255/2,255/2,255/2)); 

    int left = (max_size - src.cols) / 2;
    int top = (max_size - src.rows) / 2;

    cv::Mat dest_roi = dest(cv::Rect(left, top, src.cols, src.rows));
    src.copyTo(dest_roi);

    x1 += left;
    x2 += left;
    y1 += top;
    y2 += top;
}

cv::Mat horizontal_flip(const cv::Mat &src) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);
    cv::flip(src, result, 1);  
    return result;
}

cv::Mat rotate_and_translate(const cv::Mat &src, double angle, int left, int top) {

    cv::Mat affine = cv::getRotationMatrix2D(cv::Point(src.rows / 2, src.cols / 2), -angle, 1);

    affine.at<double>(0, 2) += left;
    affine.at<double>(1, 2) += top;

    cv::Mat result = cv::Mat(cv::Size(src.rows, src.cols), CV_8UC3, cv::Scalar(255/2, 255/2, 255/2)); 

    cv::warpAffine(src, result, affine, src.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    
    return result;
}

cv::Mat zoom(const cv::Mat &src, float by) {
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(), by, by);
    cv::Mat result(cv::Size(src.cols, src.rows), CV_8UC3, cv::Scalar(255/2, 255/2, 255/2));
    if (by >= 1) {
        int left = (resized.cols - src.cols)/2;
        int width = src.cols;
        int top = (resized.rows - src.rows)/2;
        int height = src.rows;
        resized(cv::Rect(left, top, width, height)).copyTo(result);
    } else {
        int left = (src.cols - resized.cols)/2;
        int width = resized.cols;
        int top = (src.rows - resized.rows)/2;
        int height = resized.rows;

        resized.copyTo(result(cv::Rect(left, top, width, height)));
    }
    return result;
}