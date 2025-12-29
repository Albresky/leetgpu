#include <cuda_runtime.h>
#define __K4

/* 0. naive impl */
#ifdef __K0
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    C[idx] = A[idx] + B[idx];
}
#endif

/* 1. using smem */
#ifdef __K1
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  __shared__ float s_A[256];
  __shared__ float s_B[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < N) {
    s_A[tid] = A[idx];
    s_B[tid] = B[idx];
  }
  __syncthreads();

  if (idx < N) {
    C[idx] = s_A[tid] + s_B[tid];
  }
}
#endif

/* 2. vectorized float4 */
#ifdef __K2
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // process 4 elem per thread
  int bound = N / 4;
  if (idx < bound) {
    float4 a4 = reinterpret_cast<const float4 *>(A)[idx];
    float4 b4 = reinterpret_cast<const float4 *>(B)[idx];
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    reinterpret_cast<float4 *>(C)[idx] = c4;
  }

  // tail loop
  for (int i = bound * 4 + idx; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}
#endif

/* 3. Grid-Stride Loop + float4 + __restrict__ */
#ifdef __K3
__global__ void vector_add(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // process 4 elem per thread
  int bound = N / 4;
  for (int i = idx; i < bound; i += stride) {
    float4 val_a = reinterpret_cast<const float4 *>(A)[i];
    float4 val_b = reinterpret_cast<const float4 *>(B)[i];
    float4 val_c;
    val_c.x = val_a.x + val_b.x;
    val_c.y = val_a.y + val_b.y;
    val_c.z = val_a.z + val_b.z;
    val_c.w = val_a.w + val_b.w;
    reinterpret_cast<float4 *>(C)[i] = val_c;
  }

  // tail loop
  for (int i = bound * 4 + idx; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}
#endif

/* 4. ILP Optimized: Float4 + Manual Unrolling + Async-like patterns */
#ifdef __K4
__global__ void vector_add(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const float4 *A4 = reinterpret_cast<const float4 *>(A);
  const float4 *B4 = reinterpret_cast<const float4 *>(B);
  float4 *C4 = reinterpret_cast<float4 *>(C);

  int bound = N / 4;

  // latency hiding via ILP
  int i = idx;
  for (; i + 3 * stride < bound; i += 4 * stride) {
    // 1. batch load 8 float4, 4 for A, 4 for B
    float4 a0 = A4[i];
    float4 a1 = A4[i + stride];
    float4 a2 = A4[i + 2 * stride];
    float4 a3 = A4[i + 3 * stride];

    float4 b0 = B4[i];
    float4 b1 = B4[i + stride];
    float4 b2 = B4[i + 2 * stride];
    float4 b3 = B4[i + 3 * stride];

    // 2. compute
    float4 c0, c1, c2, c3;

    // unroll add
    c0.x = a0.x + b0.x;
    c0.y = a0.y + b0.y;
    c0.z = a0.z + b0.z;
    c0.w = a0.w + b0.w;

    c1.x = a1.x + b1.x;
    c1.y = a1.y + b1.y;
    c1.z = a1.z + b1.z;
    c1.w = a1.w + b1.w;

    c2.x = a2.x + b2.x;
    c2.y = a2.y + b2.y;
    c2.z = a2.z + b2.z;
    c2.w = a2.w + b2.w;

    c3.x = a3.x + b3.x;
    c3.y = a3.y + b3.y;
    c3.z = a3.z + b3.z;
    c3.w = a3.w + b3.w;

    // 3. batch store
    C4[i] = c0;
    C4[i + stride] = c1;
    C4[i + 2 * stride] = c2;
    C4[i + 3 * stride] = c3;
  }

  // tail loop
  for (; i < bound; i += stride) {
    float4 val_a = A4[i];
    float4 val_b = B4[i];
    float4 val_c;
    val_c.x = val_a.x + val_b.x;
    val_c.y = val_a.y + val_b.y;
    val_c.z = val_a.z + val_b.z;
    val_c.w = val_a.w + val_b.w;
    C4[i] = val_c;
  }

  int tail_start = bound * 4 + idx;
  for (int k = tail_start; k < N; k += stride) {
    C[k] = A[k] + B[k];
  }
}
#endif

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;

#if defined __K3 || defined __K4
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  int blocksPerGrid = 32 * numSMs;
#elif defined(__K2)
  int num_threads = N / 4;
  int blocksPerGrid = (num_threads + threadsPerBlock - 1) / threadsPerBlock;
#else
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
#endif

  vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}
