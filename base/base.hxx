#ifndef OPENCL_HANDLER_HXX
#define OPENCL_HANDLER_HXX
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <utility>

template <typename H>
class cl_handler
{
public:
    using deleter = int (*)(H);
    cl_handler() = default;
    cl_handler(H handler, deleter del) : m_handler{handler}, m_deleter{del} {}
    ~cl_handler()
    {
        if (m_handler && m_deleter)
            m_deleter(m_handler);
    }
    cl_handler(const cl_handler &) = delete;
    cl_handler &operator=(const cl_handler &) = delete;
    cl_handler(cl_handler &&other) noexcept
    {
        std::swap(m_handler, other.m_handler);
        std::swap(m_deleter, other.m_deleter);
    }
    cl_handler &operator=(cl_handler &&other) noexcept
    {
        std::swap(m_handler, other.m_handler);
        std::swap(m_deleter, other.m_deleter);
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
        if (m_handler && m_deleter)
            m_deleter(m_handler);
        m_handler = handler;
    }

private:
    H m_handler{nullptr};
    deleter m_deleter{nullptr};
};

cl_device_id get_device();

#endif // OPENCL_HANDLER_HXX
