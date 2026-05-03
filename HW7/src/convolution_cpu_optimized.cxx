#include "utils.hxx"

#include <immintrin.h>

#include <charconv>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

static_assert(__AVX2__,
              "This file should only be compiled when AVX2 is available.");

template <typename T>
struct avx2_traits;

template <>
struct avx2_traits<std::int32_t>
{
    using vector_type = __m256i;
    static constexpr std::size_t vector_width{8};
    static vector_type load(const std::int32_t *ptr)
    {
        return _mm256_loadu_si256(reinterpret_cast<const vector_type *>(ptr));
    }
    static vector_type setzero() { return _mm256_setzero_si256(); }
    static vector_type set1(std::int32_t value)
    {
        return _mm256_set1_epi32(value);
    }
    static vector_type mul(vector_type a, vector_type b)
    {
        return _mm256_mullo_epi32(a, b);
    }
    static vector_type add(vector_type a, vector_type b)
    {
        return _mm256_add_epi32(a, b);
    }
    static void store(std::int32_t *ptr, vector_type value)
    {
        _mm256_storeu_si256(reinterpret_cast<vector_type *>(ptr), value);
    }
};
template <>
struct avx2_traits<std::int16_t>
{
    using vector_type = __m256i;
    static constexpr std::size_t vector_width{16};
    static vector_type load(const std::int16_t *ptr)
    {
        return _mm256_loadu_si256(reinterpret_cast<const vector_type *>(ptr));
    }
    static vector_type setzero() { return _mm256_setzero_si256(); }
    static vector_type set1(std::int16_t value)
    {
        return _mm256_set1_epi16(value);
    }
    static vector_type mul(vector_type a, vector_type b)
    {
        return _mm256_mullo_epi16(a, b);
    }
    static vector_type add(vector_type a, vector_type b)
    {
        return _mm256_add_epi16(a, b);
    }
    static void store(std::int16_t *ptr, vector_type value)
    {
        _mm256_storeu_si256(reinterpret_cast<vector_type *>(ptr), value);
    }
};
template <>
struct avx2_traits<float>
{
    using vector_type = __m256;
    static constexpr std::size_t vector_width{8};
    static vector_type load(const float *ptr) { return _mm256_loadu_ps(ptr); }
    static vector_type setzero() { return _mm256_setzero_ps(); }
    static vector_type set1(float value) { return _mm256_set1_ps(value); }
    static vector_type mul(vector_type a, vector_type b)
    {
        return _mm256_mul_ps(a, b);
    }
    static vector_type add(vector_type a, vector_type b)
    {
        return _mm256_add_ps(a, b);
    }
    static void store(float *ptr, vector_type value)
    {
        _mm256_storeu_ps(ptr, value);
    }
};

template <>
struct avx2_traits<double>
{
    using vector_type = __m256d;
    static constexpr std::size_t vector_width{4};
    static vector_type load(const double *ptr) { return _mm256_loadu_pd(ptr); }
    static vector_type setzero() { return _mm256_setzero_pd(); }
    static vector_type set1(double value) { return _mm256_set1_pd(value); }
    static vector_type mul(vector_type a, vector_type b)
    {
        return _mm256_mul_pd(a, b);
    }
    static vector_type add(vector_type a, vector_type b)
    {
        return _mm256_add_pd(a, b);
    }
    static void store(double *ptr, vector_type value)
    {
        _mm256_storeu_pd(ptr, value);
    }
};

template <typename T>
std::vector<T>
convolve_optimized(const std::vector<T> &input, const std::size_t input_width,
                   const std::size_t input_height, const std::vector<T> &kernel,
                   const std::size_t kernel_width,
                   const std::size_t kernel_height)
{
    const std::size_t output_width{input_width - kernel_width + 1};
    const std::size_t output_height{input_height - kernel_height + 1};
    std::vector<T> output(output_width * output_height);

    const std::size_t vector_width{avx2_traits<T>::vector_width};
    const std::size_t vector_limit{output_width / vector_width * vector_width};

    for (std::size_t out_y{0}; out_y < output_height; out_y++)
    {
        for (std::size_t out_x{0}; out_x < vector_limit; out_x += vector_width)
        {
            typename avx2_traits<T>::vector_type sum{avx2_traits<T>::setzero()};
            for (std::size_t kernel_y{0}; kernel_y < kernel_height; kernel_y++)
            {
                for (std::size_t kernel_x{0}; kernel_x < kernel_width;
                     kernel_x++)
                {
                    const std::size_t input_index{
                        (out_y + kernel_y) * input_width + out_x + kernel_x};
                    const typename avx2_traits<T>::vector_type source{
                        avx2_traits<T>::load(input.data() + input_index)};
                    const typename avx2_traits<T>::vector_type weight{
                        avx2_traits<T>::set1(
                            kernel[kernel_y * kernel_width + kernel_x])};
                    sum = avx2_traits<T>::add(
                        sum, avx2_traits<T>::mul(weight, source));
                }
            }
            avx2_traits<T>::store(output.data() + out_y * output_width + out_x,
                                  sum);
        }

        for (std::size_t out_x{vector_limit}; out_x < output_width; out_x++)
        {
            T acc{0};
            for (std::size_t kernel_y{0}; kernel_y < kernel_height; kernel_y++)
            {
                for (std::size_t kernel_x{0}; kernel_x < kernel_width;
                     kernel_x++)
                {
                    const std::size_t input_index{
                        (out_y + kernel_y) * input_width + out_x + kernel_x};
                    acc += kernel[kernel_y * kernel_width + kernel_x] *
                           input[input_index];
                }
            }
            output[out_y * output_width + out_x] = acc;
        }
    }

    return output;
}

template <typename T>
std::chrono::duration<double, std::milli>
benchmark_case(std::size_t input_width)
{
    const std::size_t input_height{input_width};
    std::vector<T> input{make_input<T>(input_width, input_height)};
    std::vector<T> kernel{make_kernel<T>()};
    std::vector<T> output;
    const auto elapsed{measure_ms(
        [&]()
        {
            output = convolve_optimized(input, input_width, input_height,
                                        kernel, KERNEL_WIDTH, KERNEL_HEIGHT);
        })};
    benchmark_sink ^= checksum(std::move(output));
    return elapsed;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: convolution_cpu_optimized <N> [--benchmark]"
                  << std::endl;
        return 1;
    }

    std::size_t input_width{};
    std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), input_width);

    const auto time_int16{benchmark_case<std::int16_t>(input_width)};
    const auto time_int32{benchmark_case<std::int32_t>(input_width)};
    const auto time_float32{benchmark_case<float>(input_width)};
    const auto time_float64{benchmark_case<double>(input_width)};

    if (argc >= 3 && std::strcmp(argv[2], "--benchmark") == 0)
    {
        std::cout << time_int16.count() << std::endl
                  << time_int32.count() << std::endl
                  << time_float32.count() << std::endl
                  << time_float64.count() << std::endl;
    }
    else
    {
        std::cout << "Input width: " << input_width << std::endl;
        std::cout << "CPU (with AVX2) time using " << TYPE_NAME(std::int16_t)
                  << ": " << time_int16.count() << " ms" << std::endl;
        std::cout << "CPU (with AVX2) time using " << TYPE_NAME(std::int32_t)
                  << ": " << time_int32.count() << " ms" << std::endl;
        std::cout << "CPU (with AVX2) time using " << TYPE_NAME(float) << ": "
                  << time_float32.count() << " ms" << std::endl;
        std::cout << "CPU (with AVX2) time using " << TYPE_NAME(double) << ": "
                  << time_float64.count() << " ms" << std::endl;
    }
}
