#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <ratio>
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
    for (cl_uint i{0}; i < platform_count; i++)
    {
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device,
                           nullptr) == CL_SUCCESS)
            break;
    }

    cl_context context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_command_queue queue{
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr)};

    cl_mem gpu_A{
        clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, nullptr)};
    cl_mem gpu_B{
        clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, nullptr)};
    cl_mem gpu_C{
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, nullptr)};

    clEnqueueWriteBuffer(queue, gpu_A, CL_TRUE, 0, size, h_A.data(), 0, nullptr,
                         nullptr);
    clEnqueueWriteBuffer(queue, gpu_B, CL_TRUE, 0, size, h_B.data(), 0, nullptr,
                         nullptr);
    clFinish(queue);
    cl_program program{clCreateProgramWithSource(context, 1, &kernel_source,
                                                 nullptr, nullptr)};
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel{clCreateKernel(program, "vector_add", nullptr)};

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_C);

    auto start_opencl{std::chrono::high_resolution_clock::now()};
    const std::size_t global_size{N};
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0,
                           nullptr, nullptr);
    clFinish(queue);
    auto end_opencl{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, unit> opencl_time{end_opencl - start_opencl};
    clEnqueueReadBuffer(queue, gpu_C, CL_TRUE, 0, size, h_C_GPU.data(), 0,
                        nullptr, nullptr);

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

    clReleaseMemObject(gpu_A);
    clReleaseMemObject(gpu_B);
    clReleaseMemObject(gpu_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
