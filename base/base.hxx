#ifndef OPENCL_HANDLER_HXX
#define OPENCL_HANDLER_HXX
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

    H get() const { return m_handler; }
    H *get_ptr() { return &m_handler; }
    explicit operator bool() const { return m_handler != nullptr; }
    // operator H() const { return m_handler; }
    cl_handler *operator&() = delete;

private:
    H m_handler{nullptr};
    deleter m_deleter{nullptr};
};
#endif // OPENCL_HANDLER_HXX
