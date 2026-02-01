#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(call)                                                      \
  {                                                                      \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                           \
    }                                                                    \
  }

inline int cudaDeviceCount()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

inline void char_to_NB(size_t size, char* output)
{
  double dsize = (double)size;
  if (dsize < 1024)
    sprintf(output, "%.2f B", dsize);
  else if (dsize < 1024 * 1024)
    sprintf(output, "%.2f KB", dsize / 1024);
  else if (dsize < 1024 * 1024 * 1024)
    sprintf(output, "%.2f MB", dsize / (1024 * 1024));
  else
    sprintf(output, "%.2f GB", dsize / (1024 * 1024 * 1024));
}

inline void cudaPrintProperties(const cudaDeviceProp& a)
{
  printf("Device: [%s] \n", a.name);
  unsigned char uuid[16] = {0};
  for (int u = 0; u < 16; u++) {
    uuid[u] = a.uuid.bytes[u];
  }
  printf("               UUID: %x%x%x%x-%x%x%x%x-%x%x%x%x-%x%x%x%x\n",
         uuid[0],
         uuid[1],
         uuid[2],
         uuid[3],
         uuid[4],
         uuid[5],
         uuid[6],
         uuid[7],
         uuid[8],
         uuid[9],
         uuid[10],
         uuid[11],
         uuid[12],
         uuid[13],
         uuid[14],
         uuid[15]);

  unsigned char luid[8] = {0};
  for (int u = 0; u < 8; u++) {
    luid[u] = a.luid[u];
  }
  printf("               LUID: %x%x%x%x-%x%x%x%x\n",
         luid[0],
         luid[1],
         luid[2],
         luid[3],
         luid[4],
         luid[5],
         luid[6],
         luid[7]);

  printf(" luidDeviceNodeMask: %u\n", a.luidDeviceNodeMask);
  char metric[20]  = {0};
  size_t total_mem = a.totalGlobalMem;
  char_to_NB(total_mem, metric);
  printf("     totalGlobalMem: %zu (%s)\n", a.totalGlobalMem, metric);
  printf("  sharedMemPerBlock: %zu\n", a.sharedMemPerBlock);
  printf("       regsPerBlock: %d\n", a.regsPerBlock);
  printf("           warpSize: %d\n", a.warpSize);
  printf("           memPitch: %zu\n", a.memPitch);
  printf("     memoryBusWidth: %d\n", a.memoryBusWidth);
  printf(" maxThreadsPerBlock: %d\n", a.maxThreadsPerBlock);
  printf("maxThreadsPerMultiP: %d\n", a.maxThreadsPerMultiProcessor);
  printf("ShrdMemPerMultiProc: %zu\n", a.sharedMemPerMultiprocessor);
  printf("regsPerMultiProcess: %d\n", a.regsPerMultiprocessor);
  printf("   maxThreadsDim(x): %d\n", a.maxThreadsDim[0]);
  printf("   maxThreadsDim(y): %d\n", a.maxThreadsDim[1]);
  printf("   maxThreadsDim(z): %d\n", a.maxThreadsDim[2]);
  printf("     maxGridSize(x): %d\n", a.maxGridSize[0]);
  printf("     maxGridSize(y): %d\n", a.maxGridSize[1]);
  printf("     maxGridSize(z): %d\n", a.maxGridSize[2]);
  printf("      totalConstMem: %zu\n", a.totalConstMem);
  printf("              major: %d\n", a.major);
  printf("              minor: %d\n", a.minor);
  printf("   textureAlignment: %zu\n", a.textureAlignment);
  printf("multiProcessorCount: %d\n", a.multiProcessorCount);
  printf("         integrated: %d\n", a.integrated);
  printf("   canMapHostMemory: %d\n", a.canMapHostMemory);
  printf("       maxTexture1D: %d\n", a.maxTexture1D);
  printf(" maxTexture1DMipmap: %d\n", a.maxTexture1DMipmap);
  printf("   surfaceAlignment: %zu\n", a.surfaceAlignment);
  printf("  concurrentKernels: %d\n", a.concurrentKernels);
  printf("         ECCEnabled: %d\n", a.ECCEnabled);
  printf("           pciBusID: %d\n", a.pciBusID);
  printf("        pciDeviceID: %d\n", a.pciDeviceID);
  printf("        pciDomainID: %d\n", a.pciDomainID);
  printf("          tccDriver: %d\n", a.tccDriver);
  printf("   asyncEngineCount: %d\n", a.asyncEngineCount);
  printf(" streamPrioritiesSp: %d\n", a.streamPrioritiesSupported);
  printf("globalL1CacheSupprt: %d\n", a.globalL1CacheSupported);
  printf(" localL1CacheSupprt: %d\n", a.localL1CacheSupported);
  printf("\n");
}

class Problem {
 public:
  virtual ~Problem()    = default;
  virtual void init()   = 0;
  virtual void run()    = 0;
  virtual void verify() = 0;

  virtual long long get_bytes() = 0;
  virtual long long get_flops() = 0;
};

// factory function
Problem* create_problem();
