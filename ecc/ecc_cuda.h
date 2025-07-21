#pragma once
#include "opencv2/opencv.hpp"

using namespace cv;

// Define motion types if not already defined by OpenCV
#ifndef MOTION_TRANSLATION
#define MOTION_TRANSLATION 0
#endif

#ifndef MOTION_EUCLIDEAN
#define MOTION_EUCLIDEAN 1
#endif

#ifndef MOTION_AFFINE
#define MOTION_AFFINE 2
#endif

#ifndef MOTION_HOMOGRAPHY
#define MOTION_HOMOGRAPHY 3
#endif

double findTransformECCGpu(InputArray templateImage, InputArray inputImage,
    InputOutputArray warpMatrix, int motionType,
    TermCriteria criteria, int gaussFiltSize = 5,
    InputArray inputMask = cv::noArray());