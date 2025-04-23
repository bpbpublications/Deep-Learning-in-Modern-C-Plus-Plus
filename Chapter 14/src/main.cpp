#include <iostream>

#include <opencv2/opencv.hpp>
#include "book/data_io.hpp"

std::random_device rd{};
const auto seed = rd();
std::mt19937 rng(seed);

cv::Mat flip(const cv::Mat &src) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);
    cv::flip(src, result, 1);  
    return result;
}

cv::Mat zoom(const cv::Mat &src, double by) {
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

cv::Mat translate(const cv::Mat &src, int left, int top) {
    cv::Mat result(cv::Size(src.cols, src.rows), CV_8UC3, cv::Scalar(255/2, 255/2, 255/2));

    int src_left = 0;
    int dest_left = left;    
    if (left < 0) {
        src_left = -left;
        dest_left = 0; 
    }
    int src_top = 0;
    int dest_top = top;
    if (top < 0) {
        src_top = -top;
        dest_top = 0;
    }
    int width = src.cols - std::abs(left);
    int height = src.rows - std::abs(top);
    src(cv::Rect(src_left, src_top, width, height)).copyTo(result(cv::Rect(dest_left, dest_top, width, height)));

    return result;
}

cv::Mat rotate(const cv::Mat &src, double angle) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);
    cv::Point2f pc(src.cols/2., src.rows/2.);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(pc, angle, 1.0);
    cv::warpAffine(src, result, rotationMatrix, src.size());
    return result;
}

cv::Mat blur(const cv::Mat &src) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);
    cv::GaussianBlur(src, result, cv::Size(11, 11), 0, 0);
    return result;
}

cv::Mat gamma_correction(const cv::Mat &src, float gamma) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::LUT(src, lookUpTable, result);

    return result;
}

cv::Mat hsv(const cv::Mat &src) {
    cv::Mat result = cv::Mat(src.rows, src.cols, CV_8UC3);
    cv::cvtColor(src, result, cv::COLOR_BGR2HSV);
    return result;
}

const int IMAGE_SIZE = 244;

void vizualize(const Eigen::Tensor<std::string, 2> &data) {

    int instance = 0;

    char key = 0;

    const char * title = "Dogs vs Cats";

    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);

    const int number_instances = data.dimension(0);

    Eigen::array<Eigen::Index, 2> extent_label = {1, 2};
    while(key != 27 && key >= 0)
    {
        const std::string image_path = data(instance, 0);
        const std::string label = data(instance, 1);

        cv::Mat dest(cv::Size(IMAGE_SIZE, IMAGE_SIZE), CV_8UC3, cv::Scalar(255/2, 255/2, 255/2));
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        std::string text = "Unavailable";
        if (!image.empty()) {
            
            float f;
            if (image.cols >= image.rows) {
                f = IMAGE_SIZE / static_cast<float>(image.cols);
            } else {
                f = IMAGE_SIZE / static_cast<float>(image.rows);
            }
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(), f, f, cv::INTER_LINEAR);

            int left = (IMAGE_SIZE - resized.cols) / 2;
            int top = (IMAGE_SIZE - resized.rows) / 2;

            resized.copyTo(dest(cv::Rect(left, top,resized.cols, resized.rows)));

            text = "Cat";
            if (label.compare("0")) {
                text = "Dog";
            }
        }
        int font = cv::FONT_HERSHEY_DUPLEX; double scale = 1.; int baseline = 0;
        int thickness = 2;
        cv::Size text_size = cv::getTextSize(text, font, scale, thickness, &baseline);
        cv::Point text_pos((dest.cols - text_size.width)/2, (dest.rows - text_size.height)/2);
        cv::putText(dest, text, text_pos, font, scale, cv::Scalar(0, 255, 0), thickness);

        cv::imshow(title, dest);

        auto augmented = zoom(dest, 0.604341); // or blur etc

        cv::imshow("augmented", augmented);

        do {
            key = cv::waitKey(0);
        } while(key != 'a' && key != 'd' && key != 27);

        if(key == 'd' && instance < number_instances) instance++;
        if(key == 'a' && instance > 0) instance--;
    }

}

int main(int, char **)
{
    auto [training_data, validation_data] = load_dogs_x_cats_dataset("../../data/dogs_x_cats/PetImages", rng);
    std::cout << "Data loaded!\n";
    std::cout << "training: " << training_data.dimensions() << "\n";
    std::cout << "validation: " << validation_data.dimensions() << "\n";

    vizualize(training_data);

    return 0;
}