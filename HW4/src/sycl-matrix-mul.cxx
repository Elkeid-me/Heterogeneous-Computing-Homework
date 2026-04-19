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

    T *a{sycl::malloc_shared<T>(elem_count, queue)};
    T *b{sycl::malloc_shared<T>(elem_count, queue)};
    T *o{sycl::malloc_shared<T>(elem_count, queue)};

    auto start{std::chrono::high_resolution_clock::now()};
    queue
        .parallel_for(sycl::range<2>{n, n},
                      [=](sycl::id<2> index)
                      {
                          o[index[0] * n + index[1]] = T{};
                          for (int k = 0; k < n; ++k)
                              o[index[0] * n + index[1]] +=
                                  a[index[0] * n + k] * b[k * n + index[1]];
                      })
        .wait();
    auto end{std::chrono::high_resolution_clock::now()};
    sycl::free(a, queue);
    sycl::free(b, queue);
    sycl::free(o, queue);
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

    if (argc < 3)
    {
        sycl::queue queue;
        std::cout << "Running on "
                  << queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;
    }
    test<int>(1024); // warm up
    auto time_int{test<int>(n)};
    auto time_float{test<int>(n)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int.count() << std::endl
                  << time_float.count() << std::endl;
    }
    else
    {
        std::cout << "Matrix width: " << n << std::endl;
        std::cout << "SYCL time using `int`: " << time_int.count() << " ms"
                  << std::endl;
        std::cout << "SYCL time using `float`: " << time_float.count() << " ms"
                  << std::endl;
    }

    return 0;
}
