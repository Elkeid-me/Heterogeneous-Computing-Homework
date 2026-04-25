#define CL_TARGET_OPENCL_VERSION 300

#include "base.hxx"
#include <CL/cl.h>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <ratio>
#include <span>
#include <vector>

const char *kernel_source = R"(
__kernel void vector_add(__global const float *A,
                         __global const float *B,
                         __global float *C)
{
    size_t i = get_global_id(0);
    C[i] = A[i] + B[i];
}
)";

void vector_add_cpu(const float *A, const float *B, float *C,
                    const std::size_t N)
{
    for (std::size_t i{0}; i < N; i++)
        C[i] = A[i] + B[i];
}

using unit = std::nano;

constexpr const char *unit_str()
{
    if constexpr (std::is_same_v<unit, std::nano>)
        return "ns";
    else if constexpr (std::is_same_v<unit, std::micro>)
        return "us";
    else if constexpr (std::is_same_v<unit, std::milli>)
        return "ms";
    return "s";
}

int main(int argc, char *argv[])
{
    std::size_t N;
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), N);
    const std::size_t size{N * sizeof(float)};

    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C_CPU(N); // CPU 结果
    std::vector<float> h_C_GPU(N); // GPU 结果

    auto start_cpu{std::chrono::high_resolution_clock::now()};
    vector_add_cpu(h_A.data(), h_B.data(), h_C_CPU.data(), N);
    auto end_cpu{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, unit> cpu_time{end_cpu - start_cpu};

    cl_device_id device;
    cl_uint platform_count;
    clGetPlatformIDs(0, nullptr, &platform_count);
    auto platforms{std::make_unique<cl_platform_id[]>(platform_count)};
    clGetPlatformIDs(platform_count, platforms.get(), nullptr);
    std::span<cl_platform_id> platform_span(platforms.get(), platform_count);
    for (auto platform : platform_span)
    {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) ==
            CL_SUCCESS)
            break;
    }

    cl_handler<cl_context> context(
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr),
        clReleaseContext);
    cl_handler<cl_command_queue> queue(
        clCreateCommandQueueWithProperties(context.get(), device, nullptr,
                                           nullptr),
        clReleaseCommandQueue);

    cl_handler<cl_mem> gpu_A(
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY, size, nullptr, nullptr),
        clReleaseMemObject);
    cl_handler<cl_mem> gpu_B(
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY, size, nullptr, nullptr),
        clReleaseMemObject);
    cl_handler<cl_mem> gpu_C(clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                            size, nullptr, nullptr),
                             clReleaseMemObject);

    clEnqueueWriteBuffer(queue.get(), gpu_A.get(), CL_TRUE, 0, size, h_A.data(),
                         0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), gpu_B.get(), CL_TRUE, 0, size, h_B.data(),
                         0, nullptr, nullptr);
    clFinish(queue.get());
    cl_handler<cl_program> program(clCreateProgramWithSource(context.get(), 1,
                                                             &kernel_source,
                                                             nullptr, nullptr),
                                   clReleaseProgram);
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel(
        clCreateKernel(program.get(), "vector_add", nullptr), clReleaseKernel);

    clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), gpu_A.get_ptr());
    clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), gpu_B.get_ptr());
    clSetKernelArg(kernel.get(), 2, sizeof(cl_mem), gpu_C.get_ptr());

    auto start_opencl{std::chrono::high_resolution_clock::now()};
    const std::size_t global_size{N};
    clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, &global_size,
                           nullptr, 0, nullptr, nullptr);
    clFinish(queue.get());
    auto end_opencl{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, unit> opencl_time{end_opencl - start_opencl};
    clEnqueueReadBuffer(queue.get(), gpu_C.get(), CL_TRUE, 0, size,
                        h_C_GPU.data(), 0, nullptr, nullptr);

    bool success{true};
    for (std::size_t i{0}; i < N; i++)
    {
        if (h_C_CPU[i] != h_C_GPU[i])
        {
            success = false;
            break;
        }
    }
    if (success)
    {
        if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
        {
            std::cout << cpu_time.count() << std::endl;
            std::cout << opencl_time.count() << std::endl;
        }
        else
        {
            std::cout << "Results match!" << std::endl;
            std::cout << "Vector size (N): " << N << std::endl;
            std::cout << "CPU Time: " << cpu_time.count() << " " << unit_str()
                      << std::endl;
            std::cout << "OpenCL Time: " << opencl_time.count() << " "
                      << unit_str() << std::endl;
        }
    }
    else
        std::cout << "Results do NOT match!" << std::endl;

    return 0;
}
