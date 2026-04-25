#include <CL/cl.h>
#include <memory>
#include <span>

cl_device_id get_device()
{
    cl_uint platform_count;
    clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_count == 0)
        return nullptr;

    auto platforms{std::make_unique<cl_platform_id[]>(platform_count)};
    clGetPlatformIDs(platform_count, platforms.get(), nullptr);
    std::span<cl_platform_id> platform_span(platforms.get(), platform_count);
    cl_device_id device{nullptr};
    for (auto platform : platform_span)
    {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) ==
            CL_SUCCESS)
            break;
    }
    return device;
}
