// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

void reduceChannelsKernel(unsigned char *image, int width, int height);
void blockChannelsKernel(unsigned char *image, int width, int height);
void multipleImagesKernel(unsigned char *image, unsigned char *output1, unsigned char *output2, unsigned char *output3, int width, int height);

void cu_run_reduce_channels(CudaImg input, CudaImg output);
void cu_run_block_channels(CudaImg input, CudaImg output);
void cu_run_multiple_images(CudaImg input, CudaImg output1, CudaImg output2, CudaImg output3);

int main()
{
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    cv::Mat inputImage = cv::imread("obrazek.jpg", cv::IMREAD_COLOR); // Load image

    uint3 m_size;
    m_size.x = inputImage.cols;
    m_size.y = inputImage.rows;
    CudaImg inputCudaImg = CudaImg(m_size);
    inputCudaImg.m_p_uchar3 = (uchar3 *)inputImage.data;

    cv::Mat outputImage1(inputImage.size(), CV_8UC3);
    cv::Mat outputImage2(inputImage.size(), CV_8UC3);
    cv::Mat outputImage3(inputImage.size(), CV_8UC3);

    m_size.x = outputImage1.cols;
    m_size.y = outputImage1.rows;
    CudaImg outputCudaImg1 = CudaImg(m_size);
    outputCudaImg1.m_p_uchar3 = (uchar3 *)outputImage1.data;

    m_size.x = outputImage2.cols;
    m_size.y = outputImage2.rows;
    CudaImg outputCudaImg2 = CudaImg(m_size);
    outputCudaImg2.m_p_uchar3 = (uchar3 *)outputImage2.data;

    m_size.x = outputImage3.cols;
    m_size.y = outputImage3.rows;
    CudaImg outputCudaImg3 = CudaImg(m_size);
    outputCudaImg3.m_p_uchar3 = (uchar3 *)outputImage3.data;
    cu_run_reduce_channels(inputCudaImg, inputCudaImg);
    cu_run_block_channels(inputCudaImg, outputCudaImg1);
    cu_run_multiple_images(inputCudaImg, outputCudaImg1, outputCudaImg2, outputCudaImg3);

    cv::imwrite("ch_green.jpg", outputImage1);
    cv::imwrite("ch_red.jpg", outputImage2);
    cv::imwrite("ch_blue.jpg", outputImage3);
    return 0;
}