#include "common.h"
#include <memory>

int main(int argc, char **argv) {
    printf("Running global main for %s...\n", argv[0]);

    std::unique_ptr<Problem> problem(create_problem());

    problem->init();

    constexpr int WARMUP_ITER = 5;
    constexpr int TEST_ITER = 20;

    printf("Warming up...\n");
    for (int i = 0; i < WARMUP_ITER; ++i) {
        problem->run();
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    printf("Running benchmark (%d iterations)...\n", TEST_ITER);

    for (int i = 0; i < TEST_ITER; ++i) {
        float milliseconds = 0;
        CHECK(cudaEventRecord(start));
        problem->run();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        total_time += milliseconds;
        if (milliseconds < min_time) min_time = milliseconds;
        if (milliseconds > max_time) max_time = milliseconds;
    }

    float avg_time = total_time / TEST_ITER;
    
    long long bytes = problem->get_bytes();
    long long flops = problem->get_flops();
    
    double mem_bandwidth = (double)bytes / (avg_time * 1e-3) / 1e9; // GB/s
    double gflops = (double)flops / (avg_time * 1e-3) / 1e9; // GFLOPS

    printf("\nPerformance Metrics:\n");
    printf("  Total Time:     %-10.4f ms\n", total_time);
    printf("  Avg Latency:    %-10.4f ms\n", avg_time);
    printf("  Min Latency:    %-10.4f ms\n", min_time);
    printf("  Max Latency:    %-10.4f ms\n", max_time);
    printf("  Mem Bandwidth:  %-10.4f GB/s\n", mem_bandwidth);
    printf("  Throughput:     %-10.4f GFLOPS\n", gflops);
    printf("\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaGetLastError()); 

    problem->verify();

    return 0;
}
