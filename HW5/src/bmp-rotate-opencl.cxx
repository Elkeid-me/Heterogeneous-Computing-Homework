#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#pragma pack(1)
struct BmpFileHeader
{
    std::uint16_t bfType;
    std::uint32_t bfSize;
    std::uint16_t bfReserved1;
    std::uint16_t bfReserved2;
    std::uint32_t bfOffBits;
};

#pragma pack(1)
struct BmpInfoHeader
{
    std::uint32_t biSize;
    std::int32_t biWidth;
    std::int32_t biHeight;
    std::uint16_t biPlanes;
    std::uint16_t biBitCount;
    std::uint32_t biCompression;
    std::uint32_t biSizeImage;
    std::int32_t biXPelsPerMeter;
    std::int32_t biYPelsPerMeter;
    std::uint32_t biClrUsed;
    std::uint32_t biClrImportant;
};

struct Image24
{
    int width;
    int height;
    std::vector<std::uint8_t> rgb;
};

bool read_bmp24(const std::string &path, Image24 &image)
{
    std::ifstream ifs(path, std::ios::binary);
    BmpFileHeader file_header{};
    BmpInfoHeader info_header{};

    ifs.read(reinterpret_cast<char *>(&file_header), sizeof(file_header));
    ifs.read(reinterpret_cast<char *>(&info_header), sizeof(info_header));

    if (file_header.bfType != 0x4D42)
    {
        std::cerr << "Input is not a BMP file." << std::endl;
        return false;
    }

    if (info_header.biSize != 40 || info_header.biPlanes != 1 ||
        info_header.biBitCount != 24 || info_header.biCompression != 0)
    {
        std::cerr << "Only uncompressed 24-bit BMP is supported." << std::endl;
        return false;
    }

    const int width{info_header.biWidth};
    const int height_abs{(info_header.biHeight < 0) ? -info_header.biHeight
                                                    : info_header.biHeight};
    const bool top_down{info_header.biHeight < 0};
    if (width <= 0 || height_abs <= 0)
    {
        std::cerr << "Invalid BMP dimensions." << std::endl;
        return false;
    }

    const std::size_t row_bytes{static_cast<std::size_t>(width) * 3};
    const std::size_t bmp_stride{(row_bytes + 3u) & ~3u};

    image.width = width;
    image.height = height_abs;
    image.rgb.assign(static_cast<std::size_t>(width) *
                         static_cast<std::size_t>(height_abs) * 3u,
                     0);

    ifs.seekg(static_cast<std::streamoff>(file_header.bfOffBits),
              std::ios::beg);
    std::vector<std::uint8_t> row(bmp_stride);
    for (int y{0}; y < height_abs; y++)
    {
        ifs.read(reinterpret_cast<char *>(row.data()),
                 static_cast<std::streamsize>(bmp_stride));

        const int dst_y{top_down ? y : (height_abs - 1 - y)};
        unsigned char *dst{image.rgb.data() +
                           static_cast<std::size_t>(dst_y) * row_bytes};

        for (int x{0}; x < width; x++)
        {
            const std::size_t bmp_index{static_cast<std::size_t>(x) * 3u};
            const std::size_t dst_index{bmp_index};
            dst[dst_index + 0] = row[bmp_index + 2];
            dst[dst_index + 1] = row[bmp_index + 1];
            dst[dst_index + 2] = row[bmp_index + 0];
        }
    }

    return true;
}

static bool write_bmp24(const std::string &path, const Image24 &image)
{
    if (image.width <= 0 || image.height <= 0)
    {
        std::cerr << "Invalid output image dimensions." << std::endl;
        return false;
    }

    const std::size_t row_bytes{static_cast<std::size_t>(image.width) * 3u};
    const std::size_t bmp_stride{(row_bytes + 3u) & ~3u};
    const std::size_t pixel_bytes{bmp_stride *
                                  static_cast<std::size_t>(image.height)};

    BmpFileHeader file_header{};
    file_header.bfType = 0x4D42;
    file_header.bfOffBits = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);
    file_header.bfSize =
        static_cast<std::uint32_t>(file_header.bfOffBits + pixel_bytes);

    BmpInfoHeader info_header{};
    info_header.biSize = 40;
    info_header.biWidth = image.width;
    info_header.biHeight = image.height;
    info_header.biPlanes = 1;
    info_header.biBitCount = 24;
    info_header.biCompression = 0;
    info_header.biSizeImage = static_cast<std::uint32_t>(pixel_bytes);

    std::ofstream ofs(path, std::ios::binary);

    ofs.write(reinterpret_cast<const char *>(&file_header),
              sizeof(file_header));
    ofs.write(reinterpret_cast<const char *>(&info_header),
              sizeof(info_header));

    std::vector<std::uint8_t> row(bmp_stride, 0);
    for (int y{image.height - 1}; y >= 0; y--)
    {
        const auto *src =
            image.rgb.data() + static_cast<std::size_t>(y) * row_bytes;
        for (int x{0}; x < image.width; x++)
        {
            const std::size_t src_index{static_cast<std::size_t>(x) * 3u};
            const std::size_t bmp_index{src_index};
            row[bmp_index + 0] = src[src_index + 2];
            row[bmp_index + 1] = src[src_index + 1];
            row[bmp_index + 2] = src[src_index + 0];
        }

        ofs.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(bmp_stride));
    }

    return true;
}

#ifdef USE_CPU
#define DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif
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
        if (clGetDeviceIDs(platforms[i], DEVICE_TYPE, 1, &device, nullptr) ==
            CL_SUCCESS)
            break;
    }
    return device;
}

static const char *kRotateKernelSource = R"(
__kernel void rotate24(read_only image2d_t src, write_only image2d_t dst,
                       const int width, const int height, const float sinv,
                       const float cosv, const float cx, const float cy)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    if (x >= width || y >= height)
        return;

    const float dx = (float)x - cx;
    const float dy = (float)y - cy;

    // Inverse mapping: destination pixel samples from source.
    const float sx = cosv * dx + sinv * dy + cx;
    const float sy = -sinv * dx + cosv * dy + cy;

    const int ix = (int)floor(sx + 0.5f);
    const int iy = (int)floor(sy + 0.5f);

    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    if (ix >= 0 && ix < width && iy >= 0 && iy < height)
    {
        const uint4 p = read_imageui(src, smp, (int2)(ix, iy));
        write_imageui(dst, (int2)(x, y), (uint4)(p.x, p.y, p.z, 255));
    }
    else
        write_imageui(dst, (int2)(x, y), (uint4)(0, 0, 0, 255));
}
)";

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr
            << "Usage: bmp-rotate-opencl <input.bmp> <output.bmp> <degrees>"
            << std::endl;
        return EXIT_FAILURE;
    }

    const std::string input_path{argv[1]};
    const std::string output_path{argv[2]};

    float degrees{std::stof(argv[3])};

    Image24 input_image{};
    if (!read_bmp24(input_path, input_image))
        return EXIT_FAILURE;

    const std::size_t pixel_count{static_cast<std::size_t>(input_image.width) *
                                  static_cast<std::size_t>(input_image.height)};
    const std::size_t rgba_bytes{pixel_count * 4u};
    std::vector<std::uint8_t> input_rgba(rgba_bytes);
    for (std::size_t i{0}; i < pixel_count; i++)
    {
        input_rgba[i * 4 + 0] = input_image.rgb[i * 3 + 0];
        input_rgba[i * 4 + 1] = input_image.rgb[i * 3 + 1];
        input_rgba[i * 4 + 2] = input_image.rgb[i * 3 + 2];
        input_rgba[i * 4 + 3] = 255;
    }

    cl_device_id device{get_device()};

    cl_context context{
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr)};
    cl_command_queue queue{
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr)};
    cl_program program{clCreateProgramWithSource(
        context, 1, &kRotateKernelSource, nullptr, nullptr)};
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel{clCreateKernel(program, "rotate24", nullptr)};

    cl_image_format image_format{};
    image_format.image_channel_order = CL_RGBA;
    image_format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc src_image_desc{};
    src_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_image_desc.image_width = static_cast<std::size_t>(input_image.width);
    src_image_desc.image_height = static_cast<std::size_t>(input_image.height);
    src_image_desc.image_row_pitch =
        static_cast<std::size_t>(input_image.width) * 4u;

    cl_image_desc dst_image_desc{};
    dst_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dst_image_desc.image_width = static_cast<std::size_t>(input_image.width);
    dst_image_desc.image_height = static_cast<std::size_t>(input_image.height);
    dst_image_desc.image_row_pitch = 0;

    cl_mem src_image{clCreateImage(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &image_format,
        &src_image_desc, input_rgba.data(), nullptr)};
    cl_mem dst_image{clCreateImage(context, CL_MEM_WRITE_ONLY, &image_format,
                                   &dst_image_desc, nullptr, nullptr)};
    const float radians{degrees * 3.14159265358979323846f / 180.0f};
    const float sinv{std::sin(radians)};
    const float cosv{std::cos(radians)};
    const float cx{(static_cast<float>(input_image.width) - 1.0f) * 0.5f};
    const float cy{(static_cast<float>(input_image.height) - 1.0f) * 0.5f};

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_image);
    clSetKernelArg(kernel, 2, sizeof(int), &input_image.width);
    clSetKernelArg(kernel, 3, sizeof(int), &input_image.height);
    clSetKernelArg(kernel, 4, sizeof(float), &sinv);
    clSetKernelArg(kernel, 5, sizeof(float), &cosv);
    clSetKernelArg(kernel, 6, sizeof(float), &cx);
    clSetKernelArg(kernel, 7, sizeof(float), &cy);

    const std::size_t global_size[2]{
        static_cast<std::size_t>(input_image.width),
        static_cast<std::size_t>(input_image.height)};

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0,
                           nullptr, nullptr);
    clFinish(queue);

    Image24 output_image{};
    output_image.width = input_image.width;
    output_image.height = input_image.height;
    output_image.rgb.resize(pixel_count * 3u);

    std::vector<std::uint8_t> output_rgba(rgba_bytes);
    const std::size_t origin[3]{0, 0, 0};
    const std::size_t region[3]{static_cast<std::size_t>(input_image.width),
                                static_cast<std::size_t>(input_image.height),
                                1};
    const std::size_t row_pitch{static_cast<std::size_t>(input_image.width) *
                                4u};
    clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, row_pitch, 0,
                       output_rgba.data(), 0, nullptr, nullptr);

    for (std::size_t i{0}; i < pixel_count; i++)
    {
        output_image.rgb[i * 3 + 0] = output_rgba[i * 4 + 0];
        output_image.rgb[i * 3 + 1] = output_rgba[i * 4 + 1];
        output_image.rgb[i * 3 + 2] = output_rgba[i * 4 + 2];
    }

    clReleaseMemObject(dst_image);
    clReleaseMemObject(src_image);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    if (!write_bmp24(output_path, output_image))
        return EXIT_FAILURE;

    std::cout << "Rotation completed: " << input_path << " -> " << output_path
              << ", angle = " << degrees << " deg" << std::endl;
    return 0;
}
