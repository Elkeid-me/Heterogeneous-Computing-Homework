#include "base.hxx"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>

#if defined(_WIN32)
#define ROTATE_IMAGE_OPENCL_API __declspec(dllexport)
#else
#define ROTATE_IMAGE_OPENCL_API [[gnu::visibility("default")]]
#endif

namespace
{
    constexpr const char *KERNEL_SOURCE{
        R"(
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void rotate_image(__read_only image2d_t src_img,
                           __write_only image2d_t dst_img, float angle)
{
    int width = get_image_width(src_img);
    int height = get_image_height(src_img);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    float sinma = sin(angle);
    float cosma = cos(angle);
    int hwidth = width / 2;
    int hheight = height / 2;
    int xt = x - hwidth;
    int yt = y - hheight;
    float2 read_coord;
    read_coord.x = cosma * xt - sinma * yt + hwidth;
    read_coord.y = sinma * xt + cosma * yt + hheight;
    float4 value = read_imagef(src_img, sampler, read_coord);
    write_imagef(dst_img, (int2)(x, y), value);
}
)"};
}

extern "C" ROTATE_IMAGE_OPENCL_API void rotate_image_cpu(std::uint32_t *src_ptr,
                                                         std::uint32_t *dst_ptr,
                                                         int width, int height,
                                                         float degrees)
{
    auto start{std::chrono::high_resolution_clock::now()};
    const float radians{degrees * std::numbers::pi_v<float> / 180.0f};
    float sin_value{std::sin(radians)};
    float cos_value{std::cos(radians)};
    float hwidth{width * 0.5f};
    float hheight{height * 0.5f};
    for (int x{0}; x < width; x++)
    {
        for (int y{0}; y < height; y++)
        {
            float xt{x - hwidth};
            float yt{y - hheight};
            float read_x{cos_value * xt - sin_value * yt + hwidth};
            float read_y{sin_value * xt + cos_value * yt + hheight};
            int read_xi = static_cast<int>(std::round(read_x));
            int read_yi = static_cast<int>(std::round(read_y));
            if ((read_xi >= 0 && read_xi < width) &&
                (read_yi >= 0 && read_yi < height))
                dst_ptr[y * width + x] = src_ptr[read_yi * width + read_xi];
            else
                dst_ptr[y * width + x] = 0;
        }
    }
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> time{end - start};
    std::cout << time.count() << std::endl;
}

extern "C" ROTATE_IMAGE_OPENCL_API void
rotate_image_opencl(std::uint32_t *src_ptr, std::uint32_t *dst_ptr, int width,
                    int height, float degrees)
{

    const std::size_t width_size{static_cast<std::size_t>(width)};
    const std::size_t height_size{static_cast<std::size_t>(height)};
    cl_device_id device{get_device()};
    cl_handler<cl_context> context(
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr),
        clReleaseContext);
    cl_handler<cl_command_queue> queue(
        clCreateCommandQueueWithProperties(context.get(), device, nullptr,
                                           nullptr),
        clReleaseCommandQueue);
    const char *sources{KERNEL_SOURCE};
    cl_handler<cl_program> program(
        clCreateProgramWithSource(context.get(), 1, &sources, nullptr, nullptr),
        clReleaseProgram);
    clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel(
        clCreateKernel(program.get(), "rotate_image", nullptr),
        clReleaseKernel);

    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width_size;
    desc.image_height = height_size;

    cl_handler<cl_mem> src_image(
        clCreateImage(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      &format, &desc, const_cast<std::uint32_t *>(src_ptr),
                      nullptr),
        clReleaseMemObject);

    cl_handler<cl_mem> dst_image{clCreateImage(context.get(), CL_MEM_WRITE_ONLY,
                                               &format, &desc, nullptr,
                                               nullptr),
                                 clReleaseMemObject};

    const float radians{degrees * std::numbers::pi_v<float> / 180.0f};
    set_kernel_args(kernel.get(), src_image.get_ptr(), dst_image.get_ptr(),
                    &radians);
    clFinish(queue.get());
    auto start{std::chrono::high_resolution_clock::now()};
    const std::size_t global_size[2]{width_size, height_size};
    clEnqueueNDRangeKernel(queue.get(), kernel.get(), 2, nullptr, global_size,
                           nullptr, 0, nullptr, nullptr);
    clFinish(queue.get());
    auto end{std::chrono::high_resolution_clock::now()};
    std::chrono::duration<double, std::milli> time{end - start};
    std::cout << time.count() << std::endl;
    const std::size_t origin[3]{0u, 0u, 0u};
    const std::size_t region[3]{width_size, height_size, 1u};
    clEnqueueReadImage(queue.get(), dst_image.get(), CL_TRUE, origin, region, 0,
                       0, dst_ptr, 0, nullptr, nullptr);
    clFinish(queue.get());
}
