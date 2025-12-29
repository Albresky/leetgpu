#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

class Problem {
public:
    virtual ~Problem() = default;
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void verify() = 0;
    
    virtual long long get_bytes() = 0;
    virtual long long get_flops() = 0;
};

// factory function
Problem* create_problem();
