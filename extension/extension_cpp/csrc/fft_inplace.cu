#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cufft.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

// #define CREAL crealf
// #define CIMAG cimagf
// #define CEXP cexpf
#define MATH_PI M_PI

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp {


  __device__ uint32_t reverse_bits(uint32_t x)
  {
      x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
      x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
      x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
      x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
      return (x >> 16) | (x << 16);
  }


  template<typename real_t>
  __global__ void fft_inplace_kernel(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;  
    int r = blockIdx.y;  
    int c = blockIdx.x;  

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // --- Step 1: Bit-reversal permutation ---
    for (uint32_t i = tid; i < N; i += stride)
    {
      uint32_t j = reverse_bits(i) >> (32 - log2N);
      if (j > i)
      {
        real_t tmp = x[block_offset+i];
        x[block_offset+i] = x[block_offset+j];
        x[block_offset+j] = tmp;
      }
    }
    __syncthreads();

    // --- Step 2: FFT computation ---
    for (int s = 1; s <= log2N; ++s)
    {
      int L = 1 << s;
      int num_groups = N / L;
      int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
      int total_work_items = num_groups * num_j;

      for (int idx = tid; idx < total_work_items; idx += stride)
      {
        int group_id = idx / num_j;
        int j = idx % num_j;
        int k = group_id * L;

        if (j == 0){
          real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
          real_t t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        } else{
          real_t angle1 = -2 * M_PI * j / L;
          real_t angle2 = -1 * M_PI * (L-2*j) / L;
          real_t t1 = (j == L/4) ? x[block_offset+k+j]+x[block_offset+k+j+L/2]*cosf(angle1) : x[block_offset+k+j]+x[block_offset+k+j+L/2]*cosf(angle1)-x[block_offset+k+L-j]*sinf(angle1);
          real_t t2 = (j == L/4) ? 0 : x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*sinf(angle1)+x[block_offset+k+L-j]*cosf(angle1);
          real_t t3 = (j == L/4) ? 0 : x[block_offset+k+j]+x[block_offset+k+j+L/2]*cosf(angle2)+x[block_offset+k+L-j]*sinf(angle2);
          real_t t4 = (j == L/4) ? x[block_offset+k+j+L/2]*sinf(angle1) : -x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*sinf(angle2)-x[block_offset+k+L-j]*cosf(angle2);

          x[block_offset+k+j] = t1;
          x[block_offset+k+L/2-j] = (t3!= 0) ? t3 : x[block_offset+k+L/2-j]; 
          x[block_offset+k+j+L/2] = t4; 
          x[block_offset+k+L-j] = (t2!= 0) ? t2 : x[block_offset+k+L-j];  
        }
    }
     __syncthreads(); // sync between stages
  }
   
  }

  template<typename real_t>
  __global__ void fft_inplace_kernel_double(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   
    int r = blockIdx.y;  
    int c = blockIdx.x;  

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // --- Step 1: Bit-reversal permutation ---
    for (uint32_t i = tid; i < N; i += stride)
    {
      uint32_t j = reverse_bits(i) >> (32 - log2N);
      if (j > i)
      {
        real_t tmp = x[block_offset+i];
        x[block_offset+i] = x[block_offset+j];
        x[block_offset+j] = tmp;
      }
    }
    __syncthreads();

    // --- Step 2: FFT computation ---
    for (int s = 1; s <= log2N; ++s)
    {
      int L = 1 << s;
      int num_groups = N / L;
      int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
      int total_work_items = num_groups * num_j;

      for (int idx = tid; idx < total_work_items; idx += stride)
      {
        int group_id = idx / num_j;
        int j = idx % num_j;
        int k = group_id * L;

        if (j == 0){
          real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
          real_t t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        } else{
          real_t angle1 = -2 * M_PI * j / L;
          real_t angle2 = -1 * M_PI * (L-2*j) / L;
          real_t t1 = (j == L/4) ? x[block_offset+k+j]+x[block_offset+k+j+L/2]*cos(angle1) : x[block_offset+k+j]+x[block_offset+k+j+L/2]*cos(angle1)-x[block_offset+k+L-j]*sin(angle1);
          real_t t2 = (j == L/4) ? 0 : x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*sin(angle1)+x[block_offset+k+L-j]*cos(angle1);
          real_t t3 = (j == L/4) ? 0 : x[block_offset+k+j]+x[block_offset+k+j+L/2]*cos(angle2)+x[block_offset+k+L-j]*sin(angle2);
          real_t t4 = (j == L/4) ? x[block_offset+k+j+L/2]*sin(angle1) : -x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*sin(angle2)-x[block_offset+k+L-j]*cos(angle2);

          x[block_offset+k+j] = t1;
          x[block_offset+k+L/2-j] = (t3!= 0) ? t3 : x[block_offset+k+L/2-j]; 
          x[block_offset+k+j+L/2] = t4; 
          x[block_offset+k+L-j] = (t2!= 0) ? t2 : x[block_offset+k+L-j];  
        }
    }
     __syncthreads(); // sync between stages
  }
   
  }

  __global__ void fft_inplace_kernel_bf(__nv_bfloat16* x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   
    int r = blockIdx.y;  
    int c = blockIdx.x; 

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // --- Step 1: Bit-reversal permutation ---
    for (uint32_t i = tid; i < N; i += stride)
    {
      uint32_t j = reverse_bits(i) >> (32 - log2N);
      if (j > i)
      {
        __nv_bfloat16 tmp = x[block_offset+i];
        x[block_offset+i] = x[block_offset+j];
        x[block_offset+j] = tmp;
      }
    }
    __syncthreads();

    // --- Step 2: FFT computation ---
    for (int s = 1; s <= log2N; ++s)
    {
      int L = 1 << s;
      int num_groups = N / L;
      int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
      int total_work_items = num_groups * num_j;

      for (int idx = tid; idx < total_work_items; idx += stride)
      {
        int group_id = idx / num_j;
        int j = idx % num_j;
        int k = group_id * L;

        if (j == 0){
          __nv_bfloat16 t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
          __nv_bfloat16 t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        } else if (j == L/4){
          float angle1_f = -2 * M_PI * j / L;
          __nv_bfloat16 angle1 = __float2bfloat16(angle1_f);
          __nv_bfloat16 t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2]*hcos(angle1);
          __nv_bfloat16 t2 = x[block_offset+k+j+L/2]*hsin(angle1);
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        } else{
          float angle1_f = -2 * M_PI * j / L;
          float angle2_f = -1 * M_PI * (L-2*j) / L;
          __nv_bfloat16 angle1 = __float2bfloat16(angle1_f);
          __nv_bfloat16 angle2 = __float2bfloat16(angle2_f);
          __nv_bfloat16 t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2]*hcos(angle1)-x[block_offset+k+L-j]*hsin(angle1);
          __nv_bfloat16 t2 = x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*hsin(angle1)+x[block_offset+k+L-j]*hcos(angle1);
          __nv_bfloat16 t3 = x[block_offset+k+j]+x[block_offset+k+j+L/2]*hcos(angle2)+x[block_offset+k+L-j]*hsin(angle2);
          __nv_bfloat16 t4 = -x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*hsin(angle2)-x[block_offset+k+L-j]*hcos(angle2);
          x[block_offset+k+j] = t1;
          x[block_offset+k+L/2-j] = t3;
          x[block_offset+k+j+L/2] = t4;
          x[block_offset+k+L-j] = t2;
        }
            
    }
     __syncthreads(); // sync between stages
  }
   
  }



 template<typename real_t>
  __global__ void ifft_inplace_kernel(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;  
    int r = blockIdx.y; 
    int c = blockIdx.x; 
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // Step 1: IFFT butterfly stages (in reverse order)
    for (int s = log2N; s >= 1; --s)
    {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid; tid_global < num_work_items; tid_global += stride)
        {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0)
            {
                real_t x1 = (x[block_offset+ k + j] + x[block_offset+ k + j + L / 2]) * 0.5f;
                real_t x2 = (x[block_offset+ k + j] - x[block_offset+ k + j + L / 2]) * 0.5f;
                x[block_offset+ k + j] = x1;
                x[block_offset+ k + j + L / 2] = x2;
            }
            else if (j == L / 4)
            {
                real_t xr = x[block_offset+ k + j];
                real_t xi = x[block_offset+ k + j + L / 2];
                real_t real = (xr + xr) * 0.5f;
                real_t imag = (xi - (-xi)) * 0.5f;

                real_t theta = 2.0f * MATH_PI * j / L;
                real_t c = cosf(theta);
                real_t s = sinf(theta);

                real_t sub_real = (xr - xr) * 0.5f;
                real_t sub_imag = (xi + xi) * 0.5f;

                x[block_offset+ k + j] = real;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
            }
            else
            {
                real_t ar = x[block_offset+ k + j];
                real_t ai = x[block_offset+ k + L - j];
                real_t br = x[block_offset+ k + L / 2 - j];
                real_t bi = -x[block_offset+ k + L / 2 + j];

                real_t add_real = (ar + br) * 0.5f;
                real_t add_imag = (ai + bi) * 0.5f;
                real_t sub_real = (ar - br) * 0.5f;
                real_t sub_imag = (ai - bi) * 0.5f;

                real_t theta = 2.0f * MATH_PI * j / L;
                real_t c = cosf(theta);
                real_t s = sinf(theta);

                x[block_offset+ k + j] = add_real;
                x[block_offset+ k + L / 2 - j] = add_imag;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
                x[block_offset+ k + L - j] = c * sub_imag + s * sub_real;
            }
        }
        __syncthreads();
    }

    // Step 2: Bit-reversal permutation
    for (uint32_t i = tid; i < N; i += stride)
    {
        uint32_t j = reverse_bits(i) >> (32 - log2N);
        if (j > i)
        {
            real_t tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}

 template<typename real_t>
  __global__ void ifft_inplace_kernel_double(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   
    int r = blockIdx.y;  
    int c = blockIdx.x;  
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // Step 1: IFFT butterfly stages (in reverse order)
    for (int s = log2N; s >= 1; --s)
    {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid; tid_global < num_work_items; tid_global += stride)
        {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0)
            {
                real_t x1 = (x[block_offset+ k + j] + x[block_offset+ k + j + L / 2]) * 0.5f;
                real_t x2 = (x[block_offset+ k + j] - x[block_offset+ k + j + L / 2]) * 0.5f;
                x[block_offset+ k + j] = x1;
                x[block_offset+ k + j + L / 2] = x2;
            }
            else if (j == L / 4)
            {
                real_t xr = x[block_offset+ k + j];
                real_t xi = x[block_offset+ k + j + L / 2];
                real_t real = (xr + xr) * 0.5;
                real_t imag = (xi - (-xi)) * 0.5;

                real_t theta = 2.0 * MATH_PI * j / L;
                real_t c = cos(theta);
                real_t s = sin(theta);

                real_t sub_real = (xr - xr) * 0.5;
                real_t sub_imag = (xi + xi) * 0.5;

                x[block_offset+ k + j] = real;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
            }
            else
            {
                real_t ar = x[block_offset+ k + j];
                real_t ai = x[block_offset+ k + L - j];
                real_t br = x[block_offset+ k + L / 2 - j];
                real_t bi = -x[block_offset+ k + L / 2 + j];

                real_t add_real = (ar + br) * 0.5;
                real_t add_imag = (ai + bi) * 0.5;
                real_t sub_real = (ar - br) * 0.5;
                real_t sub_imag = (ai - bi) * 0.5;

                real_t theta = 2.0 * MATH_PI * j / L;
                real_t c = cos(theta);
                real_t s = sin(theta);

                x[block_offset+ k + j] = add_real;
                x[block_offset+ k + L / 2 - j] = add_imag;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
                x[block_offset+ k + L - j] = c * sub_imag + s * sub_real;
            }
        }
        __syncthreads();
    }

    // Step 2: Bit-reversal permutation
    for (uint32_t i = tid; i < N; i += stride)
    {
        uint32_t j = reverse_bits(i) >> (32 - log2N);
        if (j > i)
        {
            real_t tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}

__global__ void ifft_inplace_kernel_bf(__nv_bfloat16* x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z; 
    int r = blockIdx.y; 
    int c = blockIdx.x; 
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // Step 1: IFFT butterfly stages (in reverse order)
    for (int s = log2N; s >= 1; --s)
    {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid; tid_global < num_work_items; tid_global += stride)
        {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0)
            {
                __nv_bfloat16 x1 = __float2bfloat16(__bfloat162float(x[block_offset +k + j] + x[block_offset +k + j + L / 2]) * 0.5f);
                __nv_bfloat16 x2 = __float2bfloat16(__bfloat162float(x[block_offset +k + j] - x[block_offset +k + j + L / 2]) * 0.5f);
                x[block_offset +k + j] = x1;
                x[block_offset +k + j + L / 2] = x2;
            }
            else if (j == L / 4)
            {
                __nv_bfloat16 xr = x[block_offset +k + j];
                __nv_bfloat16 xi = x[block_offset +k + j + L / 2];
                __nv_bfloat16 real = __float2bfloat16(__bfloat162float(xr + xr) * 0.5f);
                __nv_bfloat16 imag = __float2bfloat16(__bfloat162float(xi - (-xi)) * 0.5f);

                float theta_f = 2.0f * MATH_PI * j / L;
                 __nv_bfloat16 theta = __float2bfloat16(theta_f);
                __nv_bfloat16 c = hcos(theta);
                __nv_bfloat16 s = hsin(theta);

                __nv_bfloat16 sub_real =  __float2bfloat16(__bfloat162float(xr - xr) * 0.5f);
                __nv_bfloat16 sub_imag =  __float2bfloat16(__bfloat162float(xi + xi) * 0.5f);

                x[block_offset +k + j] = real;
                x[block_offset +k + j + L / 2] = sub_real * c - s * sub_imag;
            }
            else
            {
                __nv_bfloat16 ar = x[block_offset +k + j];
                __nv_bfloat16 ai = x[block_offset +k + L - j];
                __nv_bfloat16 br = x[block_offset +k + L / 2 - j];
                __nv_bfloat16 bi = -x[block_offset +k + L / 2 + j];

                __nv_bfloat16 add_real = __float2bfloat16(__bfloat162float(ar + br) * 0.5f);
                __nv_bfloat16 add_imag = __float2bfloat16(__bfloat162float(ai + bi) * 0.5f);
                __nv_bfloat16 sub_real = __float2bfloat16(__bfloat162float(ar - br) * 0.5f);
                __nv_bfloat16 sub_imag = __float2bfloat16(__bfloat162float(ai - bi) * 0.5f);

                float theta_f = 2.0f * MATH_PI * j / L;
                __nv_bfloat16 theta = __float2bfloat16(theta_f);
                __nv_bfloat16 c = hcos(theta);
                __nv_bfloat16 s = hsin(theta);

                x[block_offset +k + j] = add_real;
                x[block_offset +k + L / 2 - j] = add_imag;
                x[block_offset +k + j + L / 2] = c * sub_real - s * sub_imag;
                x[block_offset +k + L - j] = c * sub_imag + s * sub_real;
            }
        }
        __syncthreads();
    }

    // Step 2: Bit-reversal permutation
    for (uint32_t i = tid; i < N; i += stride)
    {
        uint32_t j = reverse_bits(i) >> (32 - log2N);
        if (j > i)
        {
            __nv_bfloat16 tmp = x[block_offset +i];
            x[block_offset +i] = x[block_offset +j];
            x[block_offset +j] = tmp;
        }
    }
}


  at::Tensor myrfft1_cuda(at::Tensor& x) {
    printf("myrfft\n");
    TORCH_CHECK(
      x.dtype() == at::kFloat || x.dtype() == at::kDouble|| x.dtype() == at::kBFloat16,
      "Input must be float, double or bfloat16"
    );
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");

    int64_t _ = x.size(0);
    int64_t r = x.size(1); 
    int64_t c = x.size(2);
    int64_t k = x.size(3);

    int num_steps = (int)log2(k);

    dim3 block_wx1(1024);
    dim3 grid_wx1(c , r , _); 
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.dtype() == at::kFloat){
      float* d_data = x.data_ptr<float>();
      using real_t = float;

    fft_inplace_kernel<real_t><<<grid_wx1, block_wx1, 0, stream>>>(d_data, _, r, c, k, num_steps);
  
    }
    else if (x.dtype() == at::kDouble){
      double* d_data = x.data_ptr<double>();
      using real_t = double;

    fft_inplace_kernel_double<real_t><<<grid_wx1, block_wx1, 0, stream>>>(d_data, _, r, c, k, num_steps);

    }
    else if (x.dtype() == at::kBFloat16){
      __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
      fft_inplace_kernel_bf<<<grid_wx1, block_wx1, 0, stream>>>(d_data, _, r, c, k, num_steps);
    }

    return x;
  }



  at::Tensor myirfft1_cuda(at::Tensor& x) {
    TORCH_CHECK(
      x.dtype() == at::kFloat || x.dtype() == at::kDouble || x.dtype() == at::kBFloat16,
      "Input must be float or double or bfloat16"
    );
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");

    int64_t _ = x.size(0);
    int64_t r = x.size(1); 
    int64_t c = x.size(2);
    int64_t k = x.size(3);

    int num_steps = (int)log2(k);

    dim3 block_wx1(1024);
    dim3 grid_wx1(c , r , _); 
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.dtype() == at::kFloat){
      float* d_data = x.data_ptr<float>();
      using real_t = float;

      ifft_inplace_kernel<real_t><<<grid_wx1, block_wx1,0,stream>>>(d_data, _, r, c, k, num_steps);
   
    } else if (x.dtype() == at::kDouble){
      double* d_data = x.data_ptr<double>();
      using real_t = double;

      ifft_inplace_kernel_double<real_t><<<grid_wx1, block_wx1,0,stream>>>(d_data, _, r, c, k, num_steps);

    }
    else if (x.dtype() == at::kBFloat16){
      __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
      ifft_inplace_kernel_bf<<<grid_wx1, block_wx1,0,stream>>>(d_data, _, r, c, k, num_steps);
    }

    return x;
  }

  TORCH_LIBRARY_FRAGMENT(extension_cpp, m) {
    m.def("myrfft1(Tensor a) -> Tensor");
    m.def("myirfft1(Tensor a) -> Tensor");
  }

  // Registers CUDA implementations 
  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("myrfft1", &myrfft1_cuda);
    m.impl("myirfft1", &myirfft1_cuda);
  }

}
