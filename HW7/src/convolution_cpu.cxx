#include "utils.hxx"

#include <charconv>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

template <typename T>
std::chrono::duration<double, std::milli>
benchmark_case(const std::size_t input_width)
{
    const std::size_t input_height{input_width};
    std::vector<T> input{make_input<T>(input_width, input_height)};
    std::vector<T> kernel{make_kernel<T>()};
    std::vector<T> output;
    const auto func{[&]()
                    {
                        output = convolve_cpu_unoptimized(
                            input, input_width, input_height, kernel,
                            KERNEL_WIDTH, KERNEL_HEIGHT);
                    }};
    const auto elapsed{measure_ms(func)};
    benchmark_sink ^= checksum(output);
    return elapsed;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: convolution_cpu <N> [--benchmark]" << std::endl;
        return 1;
    }

    std::size_t input_width;
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), input_width);

    const auto time_int8{benchmark_case<std::int8_t>(input_width)};
    const auto time_int16{benchmark_case<std::int16_t>(input_width)};
    const auto time_int32{benchmark_case<std::int32_t>(input_width)};
    const auto time_float32{benchmark_case<float>(input_width)};
    const auto time_float64{benchmark_case<double>(input_width)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int8.count() << std::endl
                  << time_int16.count() << std::endl
                  << time_int32.count() << std::endl
                  << time_float32.count() << std::endl
                  << time_float64.count() << std::endl;
    }
    else
    {
        std::cout << "Input width: " << input_width << std::endl;
        std::cout << "Kernel size: " << KERNEL_WIDTH << " x " << KERNEL_HEIGHT
                  << std::endl;
        std::cout << "CPU dense time using " << TYPE_NAME(std::int8_t) << ": "
                  << time_int8.count() << " ms" << std::endl;
        std::cout << "CPU dense time using " << TYPE_NAME(std::int16_t) << ": "
                  << time_int16.count() << " ms" << std::endl;
        std::cout << "CPU dense time using " << TYPE_NAME(std::int32_t) << ": "
                  << time_int32.count() << " ms" << std::endl;
        std::cout << "CPU dense time using " << TYPE_NAME(float) << ": "
                  << time_float32.count() << " ms" << std::endl;
        std::cout << "CPU dense time using " << TYPE_NAME(double) << ": "
                  << time_float64.count() << " ms" << std::endl;
    }
}
