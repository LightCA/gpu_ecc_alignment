#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ecc_cuda.h"
#include <opencv2/core/core.hpp>

namespace py = pybind11;

// Define the motion type constants if they're not already defined
// const int MOTION_TRANSLATION = 0;
// const int MOTION_EUCLIDEAN = 1;
// const int MOTION_AFFINE = 2;
// const int MOTION_HOMOGRAPHY = 3;

cv::Mat numpy_uint8_to_cv_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    return mat;
}

py::array_t<float> cv_mat_to_numpy_float(const cv::Mat& mat) {
    py::array_t<float> array({mat.rows, mat.cols});
    py::buffer_info buf = array.request();
    float* ptr = (float*)buf.ptr;
    memcpy(ptr, mat.data, mat.rows * mat.cols * sizeof(float));
    return array;
}

py::array_t<float> find_transform_ecc_gpu_py(
    py::array_t<unsigned char>& template_image,
    py::array_t<unsigned char>& input_image,
    py::array_t<float>& warp_matrix,
    int motion_type,
    int max_iterations,
    double epsilon,
    int gauss_filt_size = 5
) {
    // Convert numpy arrays to cv::Mat
    cv::Mat cv_template = numpy_uint8_to_cv_mat(template_image);
    cv::Mat cv_input = numpy_uint8_to_cv_mat(input_image);
    
    // Initialize warp matrix
    py::buffer_info warp_buf = warp_matrix.request();
    cv::Mat cv_warp(warp_buf.shape[0], warp_buf.shape[1], CV_32F, (float*)warp_buf.ptr);
    
    // Create termination criteria
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                            max_iterations, epsilon);

    // Call the CUDA implementation
    double correlation = findTransformECCGpu(cv_template, cv_input, cv_warp, 
                                           motion_type, criteria, gauss_filt_size);
    
    // Convert result back to numpy array
    return cv_mat_to_numpy_float(cv_warp);
}

PYBIND11_MODULE(ecc_gpu, m) {
    m.doc() = "CUDA ECC image alignment algorithm Python bindings";
    
    m.def("find_transform_ecc_gpu", &find_transform_ecc_gpu_py, 
          py::arg("template_image"),
          py::arg("input_image"),
          py::arg("warp_matrix"),
          py::arg("motion_type"),
          py::arg("max_iterations"),
          py::arg("epsilon"),
          py::arg("gauss_filt_size") = 5,
          "Find the geometric transform between two images using ECC criterion (GPU accelerated)");
          
    // Add motion type constants as module attributes
    m.attr("MOTION_TRANSLATION") = py::int_(MOTION_TRANSLATION);
    m.attr("MOTION_EUCLIDEAN") = py::int_(MOTION_EUCLIDEAN);
    m.attr("MOTION_AFFINE") = py::int_(MOTION_AFFINE);
    m.attr("MOTION_HOMOGRAPHY") = py::int_(MOTION_HOMOGRAPHY);
} 