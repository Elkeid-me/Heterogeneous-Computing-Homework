#include "base.hxx"
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ratio>
#include <vector>

#define KERNEL_SOURCE(TYPE)                                                    \
    "__kernel void mul_matrix_simple(__global" #TYPE "*output,\n"              \
    "                                __global" #TYPE "*input_a,\n"             \
    "                                __global" #TYPE "*input_b,\n"             \
    "                                uint width)\n"                            \
    "{\n"                                                                      \
    "    int global_id_x = get_global_id(0);\n"                                \
    "    int global_id_y = get_global_id(1);\n"                                \
    "    " #TYPE "sum = 0;\n"                                                  \
    "    for (int i = 0; i < width; i++)\n"                                    \
    "    {\n"                                                                  \
    "        " #TYPE "temp_a = input_a[global_id_y * width + i];\n"            \
    "        " #TYPE "temp_b = input_b[i * width + global_id_x];\n"            \
    "        sum += temp_a * temp_b;\n"                                        \
    "    }\n"                                                                  \
    "    output[global_id_y * width + global_id_x] = sum;\n"                   \
    "}"

// From AMD's *Introduction to OpenCL Programming*
template <typename T>
struct kernel;

#define DEFINE_KERNEL(TYPE)                                                    \
    template <>                                                                \
    struct kernel<TYPE>                                                        \
    {                                                                          \
        static constexpr const char *source{KERNEL_SOURCE(TYPE)};              \
    }

DEFINE_KERNEL(float);
DEFINE_KERNEL(double);

template <typename T>
std::chrono::duration<double, std::milli> test(std::size_t n)
{
    const std::size_t elem_count{n * n};
    const std::size_t bytes{elem_count * sizeof(T)};

    std::vector<T> h_a(elem_count);
    std::vector<T> h_b(elem_count);

    cl_device_id device{get_device()};
    cl_handler<cl_context> context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_handler<cl_command_queue> queue{clCreateCommandQueueWithProperties(
        context.get(), device, nullptr, nullptr)};

    cl_handler<cl_mem> gpu_a{clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                                            bytes, nullptr, nullptr)};
    cl_handler<cl_mem> gpu_b{clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                                            bytes, nullptr, nullptr)};
    cl_handler<cl_mem> gpu_o{clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                            bytes, nullptr, nullptr)};

    clEnqueueWriteBuffer(queue.get(), gpu_a.get(), CL_TRUE, 0, bytes,
                         h_a.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), gpu_b.get(), CL_TRUE, 0, bytes,
                         h_b.data(), 0, nullptr, nullptr);
    clFinish(queue.get());

    const char *kernel_source{kernel<T>::source};
    cl_handler<cl_program> program{clCreateProgramWithSource(
        context.get(), 1, &kernel_source, nullptr, nullptr)};
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel{
        clCreateKernel(program.get(), "mul_matrix_simple", nullptr)};

    const cl_uint width{static_cast<cl_uint>(n)};
    set_kernel_args(kernel.get(), gpu_o.get_ptr(), gpu_a.get_ptr(),
                    gpu_b.get_ptr(), &width);

    std::size_t global_size[2]{n, n};
    auto start{std::chrono::high_resolution_clock::now()};
    clEnqueueNDRangeKernel(queue.get(), kernel.get(), 2, nullptr, global_size,
                           nullptr, 0, nullptr, nullptr);
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
        std::cout << "OpenCL time using `float`: " << time_float.count()
                  << " ms" << std::endl;
        std::cout << "OpenCL time using `double`: " << time_double.count()
                  << " ms" << std::endl;
    }
}
