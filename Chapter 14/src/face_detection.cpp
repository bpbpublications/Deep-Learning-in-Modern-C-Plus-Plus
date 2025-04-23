#include <iostream>

#include <opencv2/opencv.hpp>

void detect_face(cv::Mat &image, cv::CascadeClassifier &face_detector)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(gray, faces);
    cv::Mat dest = image.clone();
    for (int i = 0; i < faces.size(); i++) {
        cv::Rect &face = faces[i];
        cv::rectangle(dest, cv::Point(face.x, face.y), cv::Point(face.x + face.width, face.y + face.height), cv::Scalar(0, 0, 255), 4);
    }
    imshow("Face detection", dest);
    cv::waitKey();
}

int main(int, char**) {

    std::string haar_cascade_path = "../haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier face_detector;
    if(!face_detector.load(haar_cascade_path)) {
        throw std::invalid_argument("Haar cascade file not found");
    }

    std::string image_path = "../my_image.jpg";
    // std::string image_path = "../my_image_90.jpg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()) {
        throw std::invalid_argument("Image file not found");
    }

    detect_face(image, face_detector);

    return 0;
}