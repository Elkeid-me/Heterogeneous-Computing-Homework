#define CL_TARGET_OPENCL_VERSION 300 // avoid warning

#include <CL/cl.h>
#include <cstdlib>
#include <iostream>

#define check_cl_error(err, msg)                                               \
    do                                                                         \
    {                                                                          \
        if (err != CL_SUCCESS)                                                 \
        {                                                                      \
            std::cerr << msg << std::endl;                                     \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (false)

void print_device_type(cl_device_type type)
{
    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-types-table
    switch (type)
    {
    case CL_DEVICE_TYPE_CPU:
        std::cout << "        Type: CPU" << std::endl;
        break;
    case CL_DEVICE_TYPE_GPU:
        std::cout << "        Type: GPU" << std::endl;
        break;
    case CL_DEVICE_TYPE_ACCELERATOR:
        std::cout << "        Type: Accelerator" << std::endl;
        break;
    case CL_DEVICE_TYPE_CUSTOM:
        std::cout << "        Type: Custom" << std::endl;
        break;
    case CL_DEVICE_TYPE_DEFAULT:
        std::cout << "        Type: Default" << std::endl;
        break;
    default:
        std::cout << "        Type: Unknown" << std::endl;
        break;
    }
}

int main()
{
    cl_uint platform_count;
    // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformIDs.html
    clGetPlatformIDs(0, nullptr, &platform_count);
    auto platforms{std::make_unique<cl_platform_id[]>(platform_count)};
    clGetPlatformIDs(platform_count, platforms.get(), nullptr);
    for (cl_uint i{0}; i < platform_count; i++)
    {
        char buffer[128];
        // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformIDs.html
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer),
                          buffer, nullptr);
        std::cout << "Platform " << i << ": " << buffer << std::endl;
        cl_uint device_count;
        // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr,
                        &device_count);

        auto devices{std::make_unique<cl_device_id[]>(device_count)};
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count,
                             devices.get(), nullptr);

        for (cl_uint j{0}; j < device_count; j++)
        {
            // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer,
                            nullptr);
            std::cout << "    Device " << j << ": " << buffer << std::endl;

            cl_device_type device_type;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type),
                            &device_type, nullptr);
            print_device_type(device_type);

            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer),
                            buffer, nullptr);
            std::cout << "        Version: " << buffer << std::endl;

            cl_uint compute_units;
            cl_uint clock_freq;
            cl_ulong global_mem;
            cl_uint native_width;
            cl_uint preferred_width;

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(compute_units), &compute_units, nullptr);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(clock_freq), &clock_freq, nullptr);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(global_mem), &global_mem, nullptr);
            clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                            sizeof(native_width), &native_width, nullptr);
            clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                            sizeof(preferred_width), &preferred_width, nullptr);

            std::cout << "        Compute Units: " << compute_units
                      << std::endl;
            std::cout << "        Clock: " << clock_freq << " MHz" << std::endl;
            std::cout << "        Global Memory: " << global_mem / (1024 * 1024)
                      << " MB" << std::endl;
            std::cout << "        Native Vector Width (Float): " << native_width
                      << std::endl;
            std::cout << "        Preferred Vector Width (Float): "
                      << preferred_width << std::endl;
        }
    }
}
