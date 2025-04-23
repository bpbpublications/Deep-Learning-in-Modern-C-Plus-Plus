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

#ifndef _IMAGE_ROUTINES_
#define _IMAGE_ROUTINES_

#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>

/**
 * This function convert an OpenCV mat to Eigen::Tensor
 * Note that the cv::cv2eigen function is not useful since we need to control number of channels, layout, etc...
 * The function also scale the mat values using the provided argument
*/
void convert_to_tensor(cv::Mat &mat, Eigen::Tensor<float, 3> &tensor, float scale);

/**
 * This function make an arbitrary size mat and make it square by centering
 * it in a S by S square where S is given by S = max(rows, cols).
 * If the src image is landscape, the dest mat contains a padding area at the top
 * and bottom. If the src image format is portrait, additional padding areas are
 * include at left and right of the image.
 * The additional areas along the sides or top/bottom are filled by random pixels
*/
void make_image_square(cv::Mat &src, cv::Mat &dest, int &x1, int &y1, int &x2, int &y2);

/**
 * returns a horizontal flipped version of src image
*/
cv::Mat horizontal_flip(const cv::Mat &src);

/**
 * Rotates and translates the src image.
 * @param angle is given in degrees
 * @param left is the horizontal displacement in pixels
 * @param top is the vertical displacement in pixels
 * 
*/
cv::Mat rotate_and_translate(const cv::Mat &src, double angle, int left, int top);

/**
 * Resizes the images.
 * @param by is the zoom factor varying from 0 to 1 or more. If by is between 0 and 1, the image is reduced. If by is > 1, the image is enlarged.
 * 
*/
cv::Mat zoom(const cv::Mat &src, float by);

#endif