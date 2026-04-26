#include "base.hxx"
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ratio>
#include <vector>

#define MATRIX_MUL_KERNEL(TYPE)                                                \
    "__kernel void multMatrixTiled("                                           \
    "__global " #TYPE " *mO, __global const " #TYPE " *mA,"                    \
    "__global const " #TYPE " *mB, const uint width) {"                        \
    "const uint gx = get_global_id(0);"                                        \
    "const uint gy = get_global_id(1);"                                        \
    "const uint lx = get_local_id(0);"                                         \
    "const uint ly = get_local_id(1);"                                         \
    "const uint group_x = get_group_id(0);"                                    \
    "const uint group_y = get_group_id(1);"                                    \
    "const uint local_size_x = get_local_size(0);"                             \
    "const uint local_size_y = get_local_size(1);"                             \
    ""                                                                         \
    "__local " #TYPE " tileA[16][16];"                                         \
    "__local " #TYPE " tileB[16][16];"                                         \
    ""                                                                         \
    "" #TYPE " sum = 0.0f;"                                                    \
    "const uint tiled_k = (width + local_size_x - 1) / local_size_x;"          \
    "for (uint t = 0; t < tiled_k; ++t) {"                                     \
    "const uint a_col = t * local_size_x + lx;"                                \
    "const uint a_row = group_y * local_size_y + ly;"                          \
    "const uint b_row = t * local_size_y + ly;"                                \
    "const uint b_col = group_x * local_size_x + lx;"                          \
    "tileA[ly][lx] ="                                                          \
    "(a_row < width && a_col < width) ? mA[a_row * width + a_col] : 0.0f;"     \
    "tileB[ly][lx] ="                                                          \
    "(b_row < width && b_col < width) ? mB[b_row * width + b_col] : 0.0f;"     \
    "barrier(CLK_LOCAL_MEM_FENCE);"                                            \
    ""                                                                         \
    "for (uint k = 0; k < local_size_x; ++k)"                                  \
    "sum += tileA[ly][k] * tileB[k][lx];"                                      \
    "barrier(CLK_LOCAL_MEM_FENCE);"                                            \
    "}"                                                                        \
    "if (gy < width && gx < width)"                                            \
    "mO[gy * width + gx] = sum;"                                               \
    "}"

template <typename T>
struct kernel;
template <>
struct kernel<float>
{
    static constexpr const char *source{MATRIX_MUL_KERNEL(float)};
};

template <>
struct kernel<double>
{
    static constexpr const char *source{MATRIX_MUL_KERNEL(double)};
};

template <>
struct kernel<int>
{
    static constexpr const char *source{MATRIX_MUL_KERNEL(int)};
};
template <>
struct kernel<long>
{
    static constexpr const char *source{MATRIX_MUL_KERNEL(long)};
};

template <typename T>
std::chrono::duration<double, std::milli> test(std::size_t n)
{
    const std::size_t elem_count{n * n};
    const std::size_t bytes{elem_count * sizeof(T)};

    std::vector<T> h_a(elem_count);
    std::vector<T> h_b(elem_count);

    cl_device_id device{get_device()};
    cl_handler<cl_context> context(
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr),
        clReleaseContext);
    cl_handler<cl_command_queue> queue(
        clCreateCommandQueueWithProperties(context.get(), device, nullptr,
                                           nullptr),
        clReleaseCommandQueue);

    cl_handler<cl_mem> gpu_a(clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                                            bytes, nullptr, nullptr),
                             clReleaseMemObject);
    cl_handler<cl_mem> gpu_b(clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                                            bytes, nullptr, nullptr),
                             clReleaseMemObject);
    cl_handler<cl_mem> gpu_o(clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                            bytes, nullptr, nullptr),
                             clReleaseMemObject);

    clEnqueueWriteBuffer(queue.get(), gpu_a.get(), CL_TRUE, 0, bytes,
                         h_a.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), gpu_b.get(), CL_TRUE, 0, bytes,
                         h_b.data(), 0, nullptr, nullptr);
    clFinish(queue.get());

    const char *kernel_source{kernel<T>::source};

    cl_handler<cl_program> program(clCreateProgramWithSource(context.get(), 1,
                                                             &kernel_source,
                                                             nullptr, nullptr),
                                   clReleaseProgram);
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel(
        clCreateKernel(program.get(), "multMatrixTiled", nullptr),
        clReleaseKernel);

    const cl_uint width{static_cast<cl_uint>(n)};
    set_kernel_args(kernel.get(), gpu_o.get_ptr(), gpu_a.get_ptr(),
                    gpu_b.get_ptr(), &width);

    std::size_t global_size[2]{n, n};
    std::size_t local_size[2]{16, 16}; // Adjust local size as needed
    auto start{std::chrono::high_resolution_clock::now()};
    clEnqueueNDRangeKernel(queue.get(), kernel.get(), 2, nullptr, global_size,
                           local_size, 0, nullptr, nullptr);
    clFinish(queue.get());
    auto end{std::chrono::high_resolution_clock::now()};

    return end - start;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: matrix-mul-optimized <N> [--benchmark]"
                  << std::endl;
        return 1;
    }

    std::size_t n;
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), n);

    auto time_int{test<int>(n)};
    auto time_long{test<long>(n)};
    auto time_float{test<float>(n)};
    auto time_double{test<double>(n)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int.count() << std::endl
                  << time_long.count() << std::endl
                  << time_float.count() << std::endl
                  << time_double.count() << std::endl;
    }
    else
    {
        std::cout << "Matrix width: " << n << std::endl;
        std::cout << "OpenCL time using `int`: " << time_int.count() << " ms"
                  << std::endl;
        std::cout << "OpenCL time using `long`: " << time_long.count() << " ms"
                  << std::endl;
        std::cout << "OpenCL time using `float`: " << time_float.count()
                  << " ms" << std::endl;
        std::cout << "OpenCL time using `double`: " << time_double.count()
                  << " ms" << std::endl;
    }
}
