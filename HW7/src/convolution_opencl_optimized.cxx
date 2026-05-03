#include "base.hxx"
#include "utils.hxx"

#include <charconv>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define KERNEL_SOURCE(TYPE)                                                    \
    "__kernel void convolution_sparse(__global const " #TYPE " *input,\n"      \
    "                                 __global const " #TYPE " *kernel_,\n"    \
    "                                 __global const uint *row_offsets,\n"     \
    "                                 __global const uint *col_offsets,\n"     \
    "                                 __global " #TYPE " *output,\n"           \
    "                                 const uint tap_count,\n"                 \
    "                                 const uint input_width,\n"               \
    "                                 const uint input_height,\n"              \
    "                                 const uint kernel_width,\n"              \
    "                                 const uint kernel_height)\n"             \
    "{\n"                                                                      \
    "    const uint local_x = get_local_id(0);\n"                              \
    "    const uint local_y = get_local_id(1);\n"                              \
    "    const uint group_x = get_group_id(0) * get_local_size(0);\n"          \
    "    const uint group_y = get_group_id(1) * get_local_size(1);\n"          \
    "    const uint global_x = group_x + local_x;\n"                           \
    "    const uint global_y = group_y + local_y;\n"                           \
    "    const uint output_width = input_width - kernel_width + 1;\n"          \
    "    const uint output_height = input_height - kernel_height + 1;\n"       \
    "    __local " #TYPE " tile[22][22];\n"                                    \
    "    const uint tile_width = 22;\n"                                        \
    "    const uint tile_height = 22;\n"                                       \
    "    const uint local_area = get_local_size(0) * get_local_size(1);\n"     \
    "    const uint local_index = local_y * get_local_size(0) + local_x;\n"    \
    "    for (uint idx = local_index; idx < tile_width * tile_height;\n"       \
    "         idx += local_area)\n"                                            \
    "    {\n"                                                                  \
    "        const uint ty = idx / tile_width;\n"                              \
    "        const uint tx = idx % tile_width;\n"                              \
    "        const uint input_x = group_x + tx;\n"                             \
    "        const uint input_y = group_y + ty;\n"                             \
    "        tile[ty][tx] = (input_x < input_width &&\n"                       \
    "                        input_y < input_height)\n"                        \
    "                           ? input[input_y * input_width + input_x]\n"    \
    "                           : 0;\n"                                        \
    "    }\n"                                                                  \
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"                                      \
    "    if (global_x < output_width && global_y < output_height)\n"           \
    "    {\n"                                                                  \
    "        " #TYPE " sum = 0;\n"                                             \
    "        for (uint tap = 0; tap < tap_count; tap++)\n"                     \
    "        {\n"                                                              \
    "            const uint tap_y = row_offsets[tap];\n"                       \
    "            const uint tap_x = col_offsets[tap];\n"                       \
    "            sum += kernel_[tap] *\n"                                      \
    "                   tile[local_y + tap_y][local_x + tap_x];\n"             \
    "        }\n"                                                              \
    "        output[global_y * output_width + global_x] = sum;\n"              \
    "    }\n"                                                                  \
    "}"

std::size_t align_up(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

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
std::chrono::duration<double, std::milli>
benchmark_case(std::size_t input_width)
{
    constexpr std::size_t local_size_x{16};
    constexpr std::size_t local_size_y{16};
    const std::size_t input_height{input_width};
    const std::size_t output_width{input_width - KERNEL_WIDTH + 1};
    const std::size_t output_height{input_height - KERNEL_HEIGHT + 1};
    const std::size_t input_bytes{input_width * input_height * sizeof(T)};
    const std::size_t kernel_bytes{KERNEL_WIDTH * KERNEL_HEIGHT * sizeof(T)};
    const auto input{make_input<T>(input_width, input_height)};
    const auto kernel{make_kernel<T>()};
    const auto stencil{
        make_sparse_stencil(kernel, KERNEL_WIDTH, KERNEL_HEIGHT)};

    cl_device_id device{get_device()};
    cl_handler<cl_context> context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_handler<cl_command_queue> queue{clCreateCommandQueueWithProperties(
        context.get(), device, nullptr, nullptr)};

    cl_handler<cl_mem> input_buffer{
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       input_bytes, const_cast<T *>(input.data()), nullptr)};
    cl_handler<cl_mem> kernel_buffer{clCreateBuffer(
        context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel_bytes,
        const_cast<T *>(stencil.weights.data()), nullptr)};
    cl_handler<cl_mem> row_offsets_buffer{clCreateBuffer(
        context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        stencil.row_offsets.size() * sizeof(std::uint32_t),
        const_cast<std::uint32_t *>(stencil.row_offsets.data()), nullptr)};
    cl_handler<cl_mem> col_offsets_buffer{clCreateBuffer(
        context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        stencil.col_offsets.size() * sizeof(std::uint32_t),
        const_cast<std::uint32_t *>(stencil.col_offsets.data()), nullptr)};
    cl_handler<cl_mem> output_buffer{clCreateBuffer(
        context.get(), CL_MEM_WRITE_ONLY,
        output_width * output_height * sizeof(T), nullptr, nullptr)};

    const std::string source{kernel_source<T>::value};
    const char *source_ptr{source.c_str()};
    cl_handler<cl_program> program{clCreateProgramWithSource(
        context.get(), 1, &source_ptr, nullptr, nullptr)};
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel_handle{
        clCreateKernel(program.get(), "convolution_sparse", nullptr)};

    const cl_uint tap_count_arg{static_cast<cl_uint>(stencil.weights.size())};
    const cl_uint input_width_arg{static_cast<cl_uint>(input_width)};
    const cl_uint input_height_arg{static_cast<cl_uint>(input_height)};
    const cl_uint kernel_width_arg{static_cast<cl_uint>(KERNEL_WIDTH)};
    const cl_uint kernel_height_arg{static_cast<cl_uint>(KERNEL_HEIGHT)};
    set_kernel_args(kernel_handle.get(), input_buffer.get_ptr(),
                    kernel_buffer.get_ptr(), row_offsets_buffer.get_ptr(),
                    col_offsets_buffer.get_ptr(), output_buffer.get_ptr(),
                    &tap_count_arg, &input_width_arg, &input_height_arg,
                    &kernel_width_arg, &kernel_height_arg);
    clFinish(queue.get());
    const std::size_t global_size[2]{align_up(output_width, local_size_x),
                                     align_up(output_height, local_size_y)};
    const std::size_t local_size[2]{local_size_x, local_size_y};
    const auto elapsed{measure_ms(
        [&]()
        {
            clEnqueueNDRangeKernel(queue.get(), kernel_handle.get(), 2, nullptr,
                                   global_size, local_size, 0, nullptr,
                                   nullptr);
            clFinish(queue.get());
        })};

    const std::size_t size{output_width * output_height};
    std::vector<T> output(size);
    clEnqueueReadBuffer(queue.get(), output_buffer.get(), CL_TRUE, 0,
                        size * sizeof(T), output.data(), 0, nullptr, nullptr);
    benchmark_sink ^= checksum(output);
    return elapsed;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: convolution_opencl_optimized <N> [--benchmark]"
                  << std::endl;
        return 1;
    }

    std::size_t input_width{};
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), input_width);

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
        std::cout << "OpenCL optimized time using " << TYPE_NAME(std::int8_t)
                  << ": " << time_int8.count() << " ms" << std::endl;
        std::cout << "OpenCL optimized time using " << TYPE_NAME(std::int16_t)
                  << ": " << time_int16.count() << " ms" << std::endl;
        std::cout << "OpenCL optimized time using " << TYPE_NAME(std::int32_t)
                  << ": " << time_int32.count() << " ms" << std::endl;
        std::cout << "OpenCL optimized time using " << TYPE_NAME(half_t) << ": "
                  << time_float16.count() << " ms" << std::endl;
        std::cout << "OpenCL optimized time using " << TYPE_NAME(float) << ": "
                  << time_float32.count() << " ms" << std::endl;
        std::cout << "OpenCL optimized time using " << TYPE_NAME(double) << ": "
                  << time_float64.count() << " ms" << std::endl;
    }
}
