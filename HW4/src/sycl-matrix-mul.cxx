#include <sycl/sycl.hpp>

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>

template <typename T>
std::chrono::duration<double, std::milli> test(std::size_t n)
{
    const std::size_t elem_count{n * n};
    sycl::queue queue;

    std::vector<T> a(n * n);
    std::vector<T> b(n * n);
    std::vector<T> o(n * n, T{});

    sycl::buffer a_buffer(a);
    sycl::buffer b_buffer(b);
    sycl::buffer o_buffer(o);
    auto start{std::chrono::high_resolution_clock::now()};
    queue
        .submit(
            [&](sycl::handler &h)
            {
                sycl::accessor a_accessor(a_buffer, h, sycl::read_only);
                sycl::accessor b_accessor(b_buffer, h, sycl::read_only);
                sycl::accessor o_accessor(o_buffer, h);
                h.parallel_for(sycl::range<2>{n, n},
                               [=](sycl::id<2> index)
                               {
                                   for (int k = 0; k < n; ++k)
                                       o_accessor[index[0] * n + index[1]] +=
                                           a_accessor[index[0] * n + k] *
                                           b_accessor[k * n + index[1]];
                               });
            })
        .wait();
    // sycl::host_accessor h_a(vectorR_buffer, read_only);
    auto end{std::chrono::high_resolution_clock::now()};
    return end - start;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: sycl-matrix-mul <N> [--benchmark]" << std::endl;
        return 1;
    }

    std::size_t n;
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), n);

    sycl::queue queue;
    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    auto time_int{test<int>(n)};
    auto time_long{test<long>(n)};
    auto time_float{test<int>(n)};
    auto time_double{test<long>(n)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int.count() << std::endl
                  << time_long.count() << std::endl
                  << time_float.count() << std::endl
                  << time_double.count() << std::endl;
    }
    else
    {
        std::cout << "Matrix width: " << n << std::endl;
        std::cout << "SYCL time using `int`: " << time_int.count() << " ms"
                  << std::endl;
        std::cout << "SYCL time using `long`: " << time_long.count() << " ms"
                  << std::endl;
        std::cout << "SYCL time using `float`: " << time_float.count() << " ms"
                  << std::endl;
        std::cout << "SYCL time using `double`: " << time_double.count()
                  << " ms" << std::endl;
    }

    return 0;
}
