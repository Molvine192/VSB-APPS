#pragma once
#include <opencv2/core/mat.hpp>
struct CudaImg
{
    uint3 m_size;             // size of picture
    union 
    {
        void* m_p_void;     // data of picture
        uchar1* m_p_uchar1;   // data of picture
        uchar3* m_p_uchar3;   // data of picture
        uchar4* m_p_uchar4;   // data of picture
    };
    __host__ __device__ uchar4& atuchar4(int y, int x) 
    { 
        return m_p_uchar4[y * m_size.x + x];
    }
    __host__ __device__ uchar3& atuchar3(int y, int x) 
    { 
        return m_p_uchar3[y * m_size.x + x];
    }
};