#include <ratio>
#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

constexpr cl_uint TILE_SIZE{16};

#define MATRIX_MUL_KERNEL(TYPE)                                                \
    "__kernel void multMatrixTiled("                                           \
    "__global " #TYPE " *mO, __global const " #TYPE " *mA,"                      \
    "__global const " #TYPE " *mB, const uint width) {"                         \
    "const uint gx = get_global_id(0);"                                        \
    "const uint gy = get_global_id(1);"                                        \
    "const uint lx = get_local_id(0);"                                         \
    "const uint ly = get_local_id(1);"                                         \
    "const uint group_x = get_group_id(0);"                                    \
    "const uint group_y = get_group_id(1);"                                    \
    "const uint local_size_x = get_local_size(0);"                             \
    "const uint local_size_y = get_local_size(1);"                             \
    ""                                                                         \
    "__local " #TYPE " tileA[16][16];"                                          \
    "__local " #TYPE " tileB[16][16];"                                          \
    ""                                                                         \
    "" #TYPE " sum = 0.0f;"                                                     \
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
    static constexpr const char *source = MATRIX_MUL_KERNEL(float);
};

template <>
struct kernel<double>
{
    static constexpr const char *source = MATRIX_MUL_KERNEL(double);
};

cl_device_id get_device()
{
    cl_uint platform_count;
    clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_count == 0)
        return nullptr;

    auto platforms{std::make_unique<cl_platform_id[]>(platform_count)};
    clGetPlatformIDs(platform_count, platforms.get(), nullptr);

    cl_device_id device{nullptr};
    for (cl_uint i{0}; i < platform_count; i++)
    {
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device,
                           nullptr) == CL_SUCCESS)
            break;
    }
    return device;
}

template <typename T>
std::chrono::duration<double, std::milli> test(std::size_t n)
{
    const std::size_t elem_count{n * n};
    const std::size_t bytes{elem_count * sizeof(T)};

    std::vector<T> h_a(elem_count);
    std::vector<T> h_b(elem_count);

    cl_device_id device{get_device()};
    cl_context context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_command_queue queue{
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr)};

    cl_mem gpu_a{
        clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr)};
    cl_mem gpu_b{
        clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr)};
    cl_mem gpu_o{
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr)};

    clEnqueueWriteBuffer(queue, gpu_a, CL_TRUE, 0, bytes, h_a.data(), 0,
                         nullptr, nullptr);
    clEnqueueWriteBuffer(queue, gpu_b, CL_TRUE, 0, bytes, h_b.data(), 0,
                         nullptr, nullptr);
    clFinish(queue);

    const char *kernel_source{kernel<T>::source};

    cl_program program{clCreateProgramWithSource(context, 1, &kernel_source,
                                                 nullptr, nullptr)};
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel{clCreateKernel(program, "multMatrixTiled", nullptr)};

    const cl_uint width{static_cast<cl_uint>(n)};
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_o);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_b);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &width);

    std::size_t global_size[2]{n, n};
    std::size_t local_size[2]{TILE_SIZE, TILE_SIZE};

    auto start{std::chrono::high_resolution_clock::now()};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_size,
                           0, nullptr, nullptr);
    clFinish(queue);
    auto end{std::chrono::high_resolution_clock::now()};

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(gpu_o);
    clReleaseMemObject(gpu_b);
    clReleaseMemObject(gpu_a);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

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

    auto time_float{test<float>(n)};
    auto time_double{test<double>(n)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_float.count() << std::endl
                  << time_double.count() << std::endl;
    }
    else
    {
        std::cout << "Matrix width: " << n << std::endl;
        std::cout << "OpenCL tiled time using `float`: " << time_float.count()
                  << " ms" << std::endl;
        std::cout << "OpenCL tiled time using `double`: " << time_double.count()
                  << " ms" << std::endl;
    }
}
