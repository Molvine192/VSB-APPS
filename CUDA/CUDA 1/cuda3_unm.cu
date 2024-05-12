// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Manipulation with prepared image.
//
// ***********************************************************************

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"

__global__ void reduceChannelsKernel(uchar3 *image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        uchar3 pixel = image[y * width + x];
        pixel.x /= 2;
        pixel.y /= 2;
        pixel.z /= 2;
        image[y * width + x] = pixel;
    }
}

void cu_run_reduce_channels(CudaImg input, CudaImg output)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((input.m_size.x + l_block_size - 1) / l_block_size, (input.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    reduceChannelsKernel<<<l_blocks, l_threads>>>(input.m_p_uchar3, input.m_size.x, input.m_size.y);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void blockChannelsKernel(uchar3 *image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channelIndex = (blockIdx.x / 2 + blockIdx.y / 2) % 3;
    if (x < width && y < height)
    {
        uchar3 pixel = image[y * width + x];
        if (channelIndex == 0) // Red
        {
            pixel.y = 0;
            pixel.z = 0;
        }
        else if (channelIndex == 1) // Green
        {
            pixel.x = 0;
            pixel.z = 0;
        }
        else if (channelIndex == 2) // Blue
        {
            pixel.x = 0;
            pixel.y = 0;
        }
        image[y * width + x] = pixel;
    }
}

void cu_run_block_channels(CudaImg input, CudaImg output)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((input.m_size.x + l_block_size - 1) / l_block_size, (input.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    blockChannelsKernel<<<l_blocks, l_threads>>>(input.m_p_uchar3, input.m_size.x, input.m_size.y);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void multipleImagesKernel(uchar3 *image, uchar3 *outputImage1, uchar3 *outputImage2, uchar3 *outputImage3, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        uchar3 pixel = image[y * width + x];
        outputImage1[y * width + x] = make_uchar3(pixel.z, pixel.x, pixel.y);
        outputImage2[y * width + x] = make_uchar3(pixel.y, pixel.z, pixel.x);
        outputImage3[y * width + x] = make_uchar3(pixel.x, pixel.y, pixel.z);
    }
}

void cu_run_multiple_images(CudaImg input, CudaImg output1, CudaImg output2, CudaImg output3)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((input.m_size.x + l_block_size - 1) / l_block_size, (input.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    multipleImagesKernel<<<l_blocks, l_threads>>>(input.m_p_uchar3, output1.m_p_uchar3, output2.m_p_uchar3, output3.m_p_uchar3, input.m_size.x, input.m_size.y);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}