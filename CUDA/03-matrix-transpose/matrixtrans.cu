#include <cuda_runtime.h>

#define __K1

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
    constexpr int TILESZ = 16; // TILESZ == blockDim
    __shared__ float tile[TILESZ][TILESZ];
    
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

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
