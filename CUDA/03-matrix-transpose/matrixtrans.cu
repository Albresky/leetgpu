#include <cuda_runtime.h>

#define __K0

/* 0. naive impl */
#ifdef __K0
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if(tidy < rows && tidx < cols){
        output[tidx*rows+tidy] = input[tidy*cols+tidx];
    }
}
#endif

/* 1. using smem for memory coalecing */
#ifdef __K1
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    constexpr int TILESZ = 32; // TILESZ == blockDim
    __shared__ float tile[TILESZ][TILESZ + 1];  // +1 to avoid bank conflict
    
    // input idx
    int irow = blockIdx.y*blockDim.y + threadIdx.y;
    int icol = blockIdx.x*blockDim.x + threadIdx.x;

    // output idx
    int orow = blockIdx.x*blockDim.x + threadIdx.y;
    int ocol = blockIdx.y*blockDim.y + threadIdx.x;
    
    // G2S
    if(irow < rows && icol < cols){
        tile[threadIdx.y][threadIdx.x] = input[irow*cols+icol];
    }
    __syncthreads();

    // S2G
    if(orow < cols && ocol < rows){
        output[orow*rows+ocol] = tile[threadIdx.x][threadIdx.y];
    }
}
#endif

/* 2. using float4 + smem for memory coalecing */
#ifdef __K2
__global__ void matrix_transpose_kernel(const float *input, float *output,
                                        int rows, int cols) {
  constexpr int TILESZ = 32;
  __shared__ float tile[TILESZ][TILESZ + 1]; // resolve bank conflict

  int irow = blockIdx.y * TILESZ + threadIdx.y;
  int icol = blockIdx.x * TILESZ + threadIdx.x * 4;

  if (irow < rows && icol + 3 < cols) {
    float4 vec = *reinterpret_cast<const float4 *>(&input[irow * cols + icol]);
    tile[threadIdx.y][threadIdx.x * 4 + 0] = vec.x;
    tile[threadIdx.y][threadIdx.x * 4 + 1] = vec.y;
    tile[threadIdx.y][threadIdx.x * 4 + 2] = vec.z;
    tile[threadIdx.y][threadIdx.x * 4 + 3] = vec.w;
  }
  __syncthreads();

  int orow = blockIdx.x * TILESZ + threadIdx.y;
  int ocol = blockIdx.y * TILESZ + threadIdx.x * 4;

  if (orow < cols && ocol + 3 < rows) {
    float4 vec; // non memory-coalescing for smem
    vec.x = tile[threadIdx.x * 4 + 0][threadIdx.y];
    vec.y = tile[threadIdx.x * 4 + 1][threadIdx.y];
    vec.z = tile[threadIdx.x * 4 + 2][threadIdx.y];
    vec.w = tile[threadIdx.x * 4 + 3][threadIdx.y];
    *reinterpret_cast<float4 *>(&output[orow * rows + ocol]) =
        vec; // memory-coalescing for gmem
  }
}
#endif

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
#ifdef __K2
    dim3 threadsPerBlock(8, 32);  // 8*4=32 cols, 32 rows per block
    dim3 blocksPerGrid((cols + 31) / 32, (rows + 31) / 32);
#else
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((cols + 32-1) / 32, (rows + 32-1) / 32);
#endif

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
