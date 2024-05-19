#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

void cu_resize_zmenseni(CudaImg bigpic, CudaImg smallpic, int zmenseni);
void cu_rotate1(CudaImg input, CudaImg output, float angle_degrees);
void cu_merge_halves(CudaImg img1, CudaImg img2, CudaImg output);

int main()
{
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    cv::Mat nacteny_obrazek = cv::imread("obrazek1.jpg", cv::IMREAD_UNCHANGED); // cv:IMREAD_COLOR
    CudaImg pomocny_obrazek;
    pomocny_obrazek.m_size.x = nacteny_obrazek.cols;
    pomocny_obrazek.m_size.y = nacteny_obrazek.rows;
    pomocny_obrazek.m_p_uchar4 = (uchar4 *)nacteny_obrazek.data;
    cv::Mat nacteny_obrazek2 = cv::imread("obrazek2.jpg", cv::IMREAD_UNCHANGED); // cv:IMREAD_COLOR
    CudaImg pomocny_obrazek2;
    pomocny_obrazek2.m_size.x = nacteny_obrazek2.cols;
    pomocny_obrazek2.m_size.y = nacteny_obrazek2.rows;
    pomocny_obrazek2.m_p_uchar4 = (uchar4 *)nacteny_obrazek2.data;

    cv::Mat output_image(pomocny_obrazek.m_size.x, pomocny_obrazek2.m_size.y, CV_8UC4);
    CudaImg output_image_cuda;
    output_image_cuda.m_size.x = output_image.cols;
    output_image_cuda.m_size.y = output_image.rows;
    output_image_cuda.m_p_uchar4 = (uchar4 *)output_image.data;

    cu_merge_halves(pomocny_obrazek, pomocny_obrazek2, output_image_cuda);
    cv::imwrite("merged_image.png", output_image);

    cv::Mat small3x(pomocny_obrazek.m_size.y / 3, pomocny_obrazek.m_size.x / 3, CV_8UC4);
    CudaImg trojnasobne_mensi_obrazek2;
    trojnasobne_mensi_obrazek2.m_size.x = small3x.cols;
    trojnasobne_mensi_obrazek2.m_size.y = small3x.rows;
    trojnasobne_mensi_obrazek2.m_p_uchar4 = (uchar4 *)small3x.data;
    cu_resize_zmenseni(pomocny_obrazek, trojnasobne_mensi_obrazek2, 3);
    cv::imwrite("small3x.jpg", small3x);

    cv::Mat small9x(trojnasobne_mensi_obrazek2.m_size.y / 3, trojnasobne_mensi_obrazek2.m_size.x / 3, CV_8UC4);
    CudaImg devitinasobne_mensi_obrazek2;
    devitinasobne_mensi_obrazek2.m_size.x = small9x.cols;
    devitinasobne_mensi_obrazek2.m_size.y = small9x.rows;
    devitinasobne_mensi_obrazek2.m_p_uchar4 = (uchar4 *)small9x.data;
    cu_resize_zmenseni(trojnasobne_mensi_obrazek2, devitinasobne_mensi_obrazek2, 3);
    cv::imwrite("small9x.jpg", small9x);

    cv::Mat obrazek_rotate(pomocny_obrazek.m_size.y, pomocny_obrazek.m_size.x, CV_8UC4);
    CudaImg obrazek_rotate2;
    obrazek_rotate2.m_size.x = obrazek_rotate.cols;
    obrazek_rotate2.m_size.y = obrazek_rotate.rows;
    obrazek_rotate2.m_p_uchar4 = (uchar4 *)obrazek_rotate.data;
    cu_rotate1(pomocny_obrazek, obrazek_rotate2, 90);
    cv::imwrite("rotovany.jpg", obrazek_rotate);
}