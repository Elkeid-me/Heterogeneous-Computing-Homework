#include "base.hxx"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

constexpr const char KERNEL_SOURCE[]{R"(
__kernel void kernel1_test(__global int *pDest, __global int *pSrc1,
                           __global int *pSrc2)
{
    int index = get_global_id(0);
    pDest[index] = pSrc1[index] + pSrc2[index];
}

__kernel void kernel2_test(__global int *pDest, __global int *pSrc1,
                           __global int *pSrc2)
{
    int index = get_global_id(0);
    pDest[index] = pDest[index] * pSrc1[index] - pSrc2[index];
}
)"};

int main()
{
    cl_device_id device_id{get_device()};
    cl_handler<cl_context> context(
        clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr),
        clReleaseContext);
    cl_handler<cl_command_queue> command_queue(
        clCreateCommandQueueWithProperties(context.get(), device_id, nullptr,
                                           nullptr),
        clReleaseCommandQueue);
    constexpr std::size_t ELE_NUMS{16 * 1024 * 1024};
    cl_handler<cl_mem> src1MemObj(
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                       sizeof(std::int32_t) * ELE_NUMS, nullptr, nullptr),
        clRetainMemObject);
    cl_handler<cl_mem> src2MemObj(
        clCreateBuffer(context.get(), CL_MEM_READ_ONLY,
                       sizeof(std::int32_t) * ELE_NUMS, nullptr, nullptr),
        clRetainMemObject);
    cl_handler<cl_mem> destMemObj(
        clCreateBuffer(context.get(), CL_MEM_READ_WRITE,
                       sizeof(std::int32_t) * ELE_NUMS, nullptr, nullptr),
        clRetainMemObject);
    const char *kernelSource{KERNEL_SOURCE};
    cl_handler<cl_event> events[2];
    auto pHostBuffer{std::make_unique<std::int32_t[]>(ELE_NUMS)};
    for (std::size_t i{0}; i < ELE_NUMS; i++)
        pHostBuffer[i] = static_cast<std::int32_t>(i);
    clEnqueueWriteBuffer(command_queue.get(), src1MemObj.get(), CL_FALSE, 0,
                         sizeof(std::int32_t) * ELE_NUMS, pHostBuffer.get(), 0,
                         nullptr, events[0].get_ptr());
    clEnqueueWriteBuffer(command_queue.get(), src2MemObj.get(), CL_FALSE, 0,
                         sizeof(std::int32_t) * ELE_NUMS, pHostBuffer.get(), 1,
                         events[0].get_ptr(), events[1].get_ptr());
    cl_handler<cl_program> program(clCreateProgramWithSource(context.get(), 1,
                                                             &kernelSource,
                                                             nullptr, nullptr),
                                   clReleaseProgram);
    clBuildProgram(program.get(), 1, &device_id, nullptr, nullptr, nullptr);
    cl_handler<cl_kernel> kernel(
        clCreateKernel(program.get(), "kernel1_test", nullptr),
        clReleaseKernel);
    set_kernel_args(kernel.get(), destMemObj.get_ptr(), src1MemObj.get_ptr(),
                    src2MemObj.get_ptr());
    std::size_t maxWorkGroupSize = 0;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

    cl_event evts[2]{events[0].get(), events[1].get()};
    clWaitForEvents(2, evts);

    events[0].reset();
    events[1].reset();
    clEnqueueNDRangeKernel(command_queue.get(), kernel.get(), 1, nullptr,
                           &ELE_NUMS, &maxWorkGroupSize, 0, nullptr,
                           events[0].get_ptr());
    cl_handler<cl_kernel> kernel2(
        clCreateKernel(program.get(), "kernel2_test", nullptr),
        clReleaseKernel);
    set_kernel_args(kernel2.get(), destMemObj.get_ptr(), src1MemObj.get_ptr(),
                    src2MemObj.get_ptr());
    clEnqueueNDRangeKernel(command_queue.get(), kernel2.get(), 1, nullptr,
                           &ELE_NUMS, &maxWorkGroupSize, 1, events[0].get_ptr(),
                           events[1].get_ptr());
    auto pDeviceBuffer{std::make_unique<std::int32_t[]>(ELE_NUMS)};
    clEnqueueReadBuffer(command_queue.get(), destMemObj.get(), CL_TRUE, 0,
                        ELE_NUMS * sizeof(std::int32_t), pDeviceBuffer.get(), 1,
                        events[1].get_ptr(), nullptr);
    for (std::size_t i{0}; i < ELE_NUMS; i++)
    {
        int testData{pHostBuffer[i] + pHostBuffer[i]};
        testData = testData * pHostBuffer[i] - pHostBuffer[i];
        if (pDeviceBuffer[i] != testData)
        {
            std::cerr << "Test failed at index " << i << ": expected "
                      << testData << ", got " << pDeviceBuffer[i] << ".\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "Result is OK!" << std::endl;
    return 0;
}
