#include "base.hxx"
#include "utils.hxx"

#include <charconv>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

#define KERNEL_SOURCE(TYPE)                                                    \
    "__kernel void convolution_dense(__global const " #TYPE " *input,\n"       \
    "                                __global const " #TYPE " *kernel_,\n"     \
    "                                __global " #TYPE " *output,\n"            \
    "                                const uint input_width,\n"                \
    "                                const uint input_height,\n"               \
    "                                const uint kernel_width,\n"               \
    "                                const uint kernel_height)\n"              \
    "{\n"                                                                      \
    "    const uint ox = get_global_id(0);\n"                                  \
    "    const uint oy = get_global_id(1);\n"                                  \
    "    const uint output_width = input_width - kernel_width + 1;\n"          \
    "    const uint output_height = input_height - kernel_height + 1;\n"       \
    "    if (ox >= output_width || oy >= output_height)\n"                     \
    "        return;\n"                                                        \
    "    " #TYPE " sum = 0;\n"                                                 \
    "    for (uint ky = 0; ky < kernel_height; ky++)\n"                        \
    "    {\n"                                                                  \
    "        for (uint kx = 0; kx < kernel_width; kx++)\n"                     \
    "        {\n"                                                              \
    "            const uint input_index = (oy + ky) * input_width +\n"         \
    "                                     ox + kx;\n"                          \
    "            const uint kernel_index = ky * kernel_width + kx;\n"          \
    "            sum += kernel_[kernel_index] * input[input_index];\n"         \
    "        }\n"                                                              \
    "    }\n"                                                                  \
    "    output[oy * output_width + ox] = sum;\n"                              \
    "}"

template <typename T>
struct kernel_source;
#define DEFINE_KERNEL(CPP_TYPE, CL_TYPE)                                       \
    template <>                                                                \
    struct kernel_source<CPP_TYPE>                                             \
    {                                                                          \
        constexpr static const char *value{KERNEL_SOURCE(CL_TYPE)};            \
    }

DEFINE_KERNEL(std::int8_t, char);
DEFINE_KERNEL(std::int16_t, short);
DEFINE_KERNEL(std::int32_t, int);
DEFINE_KERNEL(float, float);
DEFINE_KERNEL(double, double);
DEFINE_KERNEL(half_t, half);

template <typename T>
constexpr const char *kernel_source_v = kernel_source<T>::value;

template <typename T>
std::chrono::duration<double, std::milli>
benchmark_case(const std::size_t input_width)
{
    const std::size_t input_height{input_width};
    const std::size_t output_width{input_width - KERNEL_WIDTH + 1};
    const std::size_t output_height{input_height - KERNEL_HEIGHT + 1};
    const std::size_t input_bytes{input_width * input_height * sizeof(T)};
    const std::size_t kernel_bytes{KERNEL_WIDTH * KERNEL_HEIGHT * sizeof(T)};
    const std::size_t output_bytes{output_width * output_height * sizeof(T)};

    const auto input{make_input<T>(input_width, input_height)};
    const auto kernel{make_kernel<T>()};

    cl_device_id device{get_device()};
    cl_handler<cl_context> context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_handler<cl_command_queue> queue{clCreateCommandQueueWithProperties(
        context.get(), device, nullptr, nullptr)};

    cl_handler<cl_mem> input_buffer{
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       input_bytes, const_cast<T *>(input.data()), nullptr)};
    cl_handler<cl_mem> kernel_buffer{
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       kernel_bytes, const_cast<T *>(kernel.data()), nullptr)};
    cl_handler<cl_mem> output_buffer{clCreateBuffer(
        context.get(), CL_MEM_WRITE_ONLY, output_bytes, nullptr, nullptr)};

    const char *source{kernel_source_v<T>};
    cl_handler<cl_program> program{
        clCreateProgramWithSource(context.get(), 1, &source, nullptr, nullptr)};
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel_handle{
        clCreateKernel(program.get(), "convolution_dense", nullptr)};

    const cl_uint input_width_arg{static_cast<cl_uint>(input_width)};
    const cl_uint input_height_arg{static_cast<cl_uint>(input_height)};
    const cl_uint kernel_width_arg{static_cast<cl_uint>(KERNEL_WIDTH)};
    const cl_uint kernel_height_arg{static_cast<cl_uint>(KERNEL_HEIGHT)};
    set_kernel_args(kernel_handle.get(), input_buffer.get_ptr(),
                    kernel_buffer.get_ptr(), output_buffer.get_ptr(),
                    &input_width_arg, &input_height_arg, &kernel_width_arg,
                    &kernel_height_arg);

    const std::size_t global_size[2]{output_width, output_height};
    clFinish(queue.get());
    const auto start{std::chrono::high_resolution_clock::now()};
    clEnqueueNDRangeKernel(queue.get(), kernel_handle.get(), 2, nullptr,
                           global_size, nullptr, 0, nullptr, nullptr);
    clFinish(queue.get());
    const auto end{std::chrono::high_resolution_clock::now()};
    const auto elapsed{end - start};

    std::vector<T> output(output_width * output_height);
    clEnqueueReadBuffer(queue.get(), output_buffer.get(), CL_TRUE, 0,
                        output_bytes, output.data(), 0, nullptr, nullptr);
    benchmark_sink ^= checksum(output);
    return elapsed;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: convolution_opencl <N> [--benchmark]" << std::endl;
        return 1;
    }

    std::size_t input_width{};
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), input_width);
    input_width += 2 * (KERNEL_WIDTH / 2);

    benchmark_case<std::int32_t>(1024); // Warmup
    const auto time_int8{benchmark_case<std::int8_t>(input_width)};
    const auto time_int16{benchmark_case<std::int16_t>(input_width)};
    const auto time_int32{benchmark_case<std::int32_t>(input_width)};
    const auto time_float16{benchmark_case<half_t>(input_width)};
    const auto time_float32{benchmark_case<float>(input_width)};
    const auto time_float64{benchmark_case<double>(input_width)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int8.count() << std::endl
                  << time_int16.count() << std::endl
                  << time_int32.count() << std::endl
                  << time_float16.count() << std::endl
                  << time_float32.count() << std::endl
                  << time_float64.count() << std::endl;
    }
    else
    {
        std::cout << "Input width: " << input_width << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(std::int8_t)
                  << ": " << time_int8.count() << " ms" << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(std::int16_t)
                  << ": " << time_int16.count() << " ms" << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(std::int32_t)
                  << ": " << time_int32.count() << " ms" << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(half_t) << ": "
                  << time_float16.count() << " ms" << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(float) << ": "
                  << time_float32.count() << " ms" << std::endl;
        std::cout << "OpenCL dense time using " << TYPE_NAME(double) << ": "
                  << time_float64.count() << " ms" << std::endl;
    }
}
