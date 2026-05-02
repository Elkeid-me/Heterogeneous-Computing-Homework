#ifndef OPENCL_HANDLER_HXX
#define OPENCL_HANDLER_HXX
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <utility>

template <typename T>
void __destroy(T obj);

#define __DEFINE_DELETER(TYPE, RELEASE_FUNC)                                   \
    template <>                                                                \
    void __destroy(TYPE obj)                                                   \
    {                                                                          \
        RELEASE_FUNC(obj);                                                     \
    }

__DEFINE_DELETER(cl_context, clReleaseContext);
__DEFINE_DELETER(cl_command_queue, clReleaseCommandQueue);
__DEFINE_DELETER(cl_mem, clReleaseMemObject);
__DEFINE_DELETER(cl_program, clReleaseProgram);
__DEFINE_DELETER(cl_kernel, clReleaseKernel);
__DEFINE_DELETER(cl_event, clReleaseEvent);

template <typename H>
class cl_handler
{
public:
    cl_handler() = default;
    cl_handler(H handler) : m_handler{handler} {}
    ~cl_handler()
    {
        if (m_handler)
            __destroy<H>(m_handler);
    }
    cl_handler(const cl_handler &) = delete;
    cl_handler &operator=(const cl_handler &) = delete;
    cl_handler(cl_handler &&other) noexcept
    {
        std::swap(m_handler, other.m_handler);
    }
    cl_handler &operator=(cl_handler &&other) noexcept
    {
        std::swap(m_handler, other.m_handler);
        return *this;
    }
    cl_handler &operator=(H handler)
    {
        reset(handler);
        return *this;
    }
    H get() const { return m_handler; }
    H *get_ptr() { return &m_handler; }
    explicit operator bool() const { return m_handler != nullptr; }
    cl_handler *operator&() = delete;
    void reset(H handler = nullptr)
    {
        if (m_handler)
            __destroy<H>(m_handler);
        m_handler = handler;
    }

private:
    H m_handler{nullptr};
};

cl_device_id get_device();

template <typename... Args, std::size_t... Is>
void set_kernel_args_impl(cl_kernel kernel, std::index_sequence<Is...>,
                          Args... args)
{
    (clSetKernelArg(kernel, Is, sizeof(decltype(*args)), args), ...);
}

template <typename... Args>
void set_kernel_args(cl_kernel kernel, Args... args)
{
    set_kernel_args_impl(kernel, std::index_sequence_for<Args...>{}, args...);
}

#endif // OPENCL_HANDLER_HXX
