#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
    constexpr int kBlock{16};

#define CUDA_CHECK(expr)                                                       \
    do                                                                         \
    {                                                                          \
        const cudaError_t err{(expr)};                                         \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "   \
                      << __FILE__ << ':' << __LINE__ << std::endl;             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (false)

    template <typename T>
    __global__ void matmul_naive(T *out, const T *a, const T *b, std::size_t n)
    {
        const std::size_t row{blockIdx.y * blockDim.y + threadIdx.y};
        const std::size_t col{blockIdx.x * blockDim.x + threadIdx.x};

        if (row < n && col < n)
        {
            T sum{0};
            for (std::size_t k{0}; k < n; ++k)
                sum += a[row * n + k] * b[k * n + col];
            out[row * n + col] = sum;
        }
    }

    template <typename T>
    double run_test(std::size_t n)
    {
        const std::size_t elem_count{n * n};
        const std::size_t bytes{elem_count * sizeof(T)};

        std::vector<T> host_a(elem_count);
        std::vector<T> host_b(elem_count);

        T *dev_a{nullptr};
        T *dev_b{nullptr};
        T *dev_o{nullptr};

        CUDA_CHECK(cudaMalloc(&dev_a, bytes));
        CUDA_CHECK(cudaMalloc(&dev_b, bytes));
        CUDA_CHECK(cudaMalloc(&dev_o, bytes));

        CUDA_CHECK(
            cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 block(kBlock, kBlock);
        const dim3 grid((n + block.x - 1) / block.x,
                        (n + block.y - 1) / block.y);

        cudaEvent_t start_evt;
        cudaEvent_t stop_evt;
        CUDA_CHECK(cudaEventCreate(&start_evt));
        CUDA_CHECK(cudaEventCreate(&stop_evt));

        CUDA_CHECK(cudaEventRecord(start_evt));
        matmul_naive<<<grid, block>>>(dev_o, dev_a, dev_b, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_evt));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));

        float milliseconds{0.0f};
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_evt, stop_evt));

        CUDA_CHECK(cudaEventDestroy(start_evt));
        CUDA_CHECK(cudaEventDestroy(stop_evt));
        CUDA_CHECK(cudaFree(dev_o));
        CUDA_CHECK(cudaFree(dev_b));
        CUDA_CHECK(cudaFree(dev_a));

        return milliseconds;
    }

    bool parse_size(const char *arg, std::size_t &n)
    {
        const char *end{arg + std::strlen(arg)};
        const auto [ptr, ec] = std::from_chars(arg, end, n);
        return ec == std::errc{} && ptr == end && n > 0;
    }
} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: cuda-matrix-mul <N> [--benchmark]" << std::endl;
        return 1;
    }

    std::size_t n{0};
    if (!parse_size(argv[1], n))
    {
        std::cerr << "Invalid matrix width: " << argv[1] << std::endl;
        return 1;
    }

    int device_id{0};
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    run_test<int>(256);   // warm up
    run_test<float>(256); // warm up
    const double elapsed_int_ms{run_test<int>(n)};
    const double elapsed_float_ms{run_test<float>(n)};

    const bool benchmark_mode =
        argc >= 3 && std::string_view(argv[2]) == "--benchmark";
    if (benchmark_mode)
    {
        std::cout << elapsed_int_ms << std::endl;
        std::cout << elapsed_float_ms << std::endl;
    }
    else
    {
        std::cout << "Running on " << prop.name << std::endl;
        std::cout << "Matrix width: " << n << std::endl;
        std::cout << "CUDA(int) elapsed: " << elapsed_int_ms << " ms"
                  << std::endl;
        std::cout << "CUDA(float) elapsed: " << elapsed_float_ms << " ms"
                  << std::endl;
    }

    return 0;
}