// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
class CudaImg
{
public:
    uint3 m_size;
    union
    {
        void *m_p_void;
        uchar1 *m_p_uchar1;
        uchar3 *m_p_uchar3;
        uchar4 *m_p_uchar4;
    };

    CudaImg(uint3 m_size)
    {
        this->m_size = m_size;
    }

    __host__ __device__ uchar4 &atuchar4(int y, int x)
    {
        return m_p_uchar4[y * m_size.x + x];
    }
    
    __host__ __device__ uchar3 &atuchar3(int y, int x)
    {
        return m_p_uchar3[y * m_size.x + x];
    }
};