#include "utils.hxx"

#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#if !(defined(__AVX2__) || defined(__AVX512F__))
#error "This file should only be compiled when AVX2/AVX-512 is available."
#endif

template <typename T>
T place_holder(T, T, T)
{
    return T{};
}

template <typename T>
struct avx_traits;

#ifdef __FMA__
constexpr bool cpu_support_fma{true};
#else
constexpr bool cpu_support_fma{false};
#endif

#define DEFINE_AVX_TRAIT(TYPE, VEC_TYPE, LOAD, SETZERO, SET1, MUL, ADD,        \
                         MUL_ADD, STORE)                                       \
    template <>                                                                \
    struct avx_traits<TYPE>                                                    \
    {                                                                          \
        using vector_type = VEC_TYPE;                                          \
        static constexpr bool type_support_fma{                                \
            std::is_floating_point_v<TYPE>};                                   \
        static constexpr std::size_t vector_width{sizeof(VEC_TYPE) /           \
                                                  sizeof(TYPE)};               \
        template <std::integral U = TYPE>                                      \
        static vector_type load(const U *ptr)                                  \
        {                                                                      \
            return LOAD(reinterpret_cast<const vector_type *>(ptr));           \
        }                                                                      \
        template <std::floating_point U = TYPE>                                \
        static vector_type load(const U *ptr)                                  \
        {                                                                      \
            return LOAD(ptr);                                                  \
        }                                                                      \
        static vector_type setzero() { return SETZERO(); }                     \
        static vector_type set1(TYPE val) { return SET1(val); }                \
        static vector_type mul(vector_type a, vector_type b)                   \
        {                                                                      \
            return MUL(a, b);                                                  \
        }                                                                      \
        static vector_type add(vector_type a, vector_type b)                   \
        {                                                                      \
            return ADD(a, b);                                                  \
        }                                                                      \
        static vector_type mul_add(vector_type a, vector_type b,               \
                                   vector_type c)                              \
        {                                                                      \
            if constexpr (type_support_fma && cpu_support_fma)                 \
                return MUL_ADD(a, b, c);                                       \
            else                                                               \
                return add(mul(a, b), c);                                      \
        }                                                                      \
        template <std::integral U = TYPE>                                      \
        static void store(U *ptr, vector_type val)                             \
        {                                                                      \
            STORE(reinterpret_cast<vector_type *>(ptr), val);                  \
        }                                                                      \
        template <std::floating_point U = TYPE>                                \
        static void store(U *ptr, vector_type val)                             \
        {                                                                      \
            STORE(ptr, val);                                                   \
        }                                                                      \
    }

#ifdef __AVX512F__
DEFINE_AVX_TRAIT(std::int32_t, __m512i, _mm512_loadu_si512,
                 _mm512_setzero_si512, _mm512_set1_epi32, _mm512_mullo_epi32,
                 _mm512_add_epi32, place_holder, _mm512_storeu_si512);
DEFINE_AVX_TRAIT(float, __m512, _mm512_loadu_ps, _mm512_setzero_ps,
                 _mm512_set1_ps, _mm512_mul_ps, _mm512_add_ps, _mm512_fmadd_ps,
                 _mm512_storeu_ps);
DEFINE_AVX_TRAIT(double, __m512d, _mm512_loadu_pd, _mm512_setzero_pd,
                 _mm512_set1_pd, _mm512_mul_pd, _mm512_add_pd, _mm512_fmadd_pd,
                 _mm512_storeu_pd);
#else
DEFINE_AVX_TRAIT(std::int32_t, __m256i, _mm256_loadu_si256,
                 _mm256_setzero_si256, _mm256_set1_epi32, _mm256_mullo_epi32,
                 _mm256_add_epi32, place_holder, _mm256_storeu_si256);
DEFINE_AVX_TRAIT(float, __m256, _mm256_loadu_ps, _mm256_setzero_ps,
                 _mm256_set1_ps, _mm256_mul_ps, _mm256_add_ps, _mm256_fmadd_ps,
                 _mm256_storeu_ps);
DEFINE_AVX_TRAIT(double, __m256d, _mm256_loadu_pd, _mm256_setzero_pd,
                 _mm256_set1_pd, _mm256_mul_pd, _mm256_add_pd, _mm256_fmadd_pd,
                 _mm256_storeu_pd);
#endif

#ifdef __AVX512BW__
DEFINE_AVX_TRAIT(std::int16_t, __m512i, _mm512_loadu_si512,
                 _mm512_setzero_si512, _mm512_set1_epi16, _mm512_mullo_epi16,
                 _mm512_add_epi16, place_holder, _mm512_storeu_si512);
#else
DEFINE_AVX_TRAIT(std::int16_t, __m256i, _mm256_loadu_si256,
                 _mm256_setzero_si256, _mm256_set1_epi16, _mm256_mullo_epi16,
                 _mm256_add_epi16, place_holder, _mm256_storeu_si256);
#endif

template <typename T>
std::tuple<std::vector<T>, std::chrono::duration<double, std::milli>>
convolve_optimized(const std::vector<T> &input, const std::size_t input_width,
                   const std::size_t input_height, const std::vector<T> &kernel,
                   const std::size_t kernel_width,
                   const std::size_t kernel_height)
{
    const std::size_t output_width{input_width - kernel_width + 1};
    const std::size_t output_height{input_height - kernel_height + 1};
    std::vector<T> output(output_width * output_height);

    auto start{std::chrono::high_resolution_clock::now()};
    const std::size_t vector_width{avx_traits<T>::vector_width};
    const std::size_t vector_limit{output_width / vector_width * vector_width};
    for (std::size_t out_y{0}; out_y < output_height; out_y++)
    {
        for (std::size_t out_x{0}; out_x < vector_limit; out_x += vector_width)
        {
            typename avx_traits<T>::vector_type sum{avx_traits<T>::setzero()};
            for (std::size_t kernel_y{0}; kernel_y < kernel_height; kernel_y++)
            {
                for (std::size_t kernel_x{0}; kernel_x < kernel_width;
                     kernel_x++)
                {

                    const std::size_t input_index{
                        (out_y + kernel_y) * input_width + out_x + kernel_x};
                    const typename avx_traits<T>::vector_type source{
                        avx_traits<T>::load(input.data() + input_index)};
                    const typename avx_traits<T>::vector_type weight{
                        avx_traits<T>::set1(
                            kernel[kernel_y * kernel_width + kernel_x])};
                    sum = avx_traits<T>::mul_add(weight, source, sum);
                }
            }
            avx_traits<T>::store(output.data() + out_y * output_width + out_x,
                                 sum);
        }

        for (std::size_t out_x{vector_limit}; out_x < output_width; out_x++)
        {
            T sum{};
            for (std::size_t kernel_y{0}; kernel_y < kernel_height; kernel_y++)
            {
                for (std::size_t kernel_x{0}; kernel_x < kernel_width;
                     kernel_x++)
                {
                    const std::size_t input_index{
                        (out_y + kernel_y) * input_width + out_x + kernel_x};
                    sum += kernel[kernel_y * kernel_width + kernel_x] *
                           input[input_index];
                }
            }
            output[out_y * output_width + out_x] = sum;
        }
    }
    auto end{std::chrono::high_resolution_clock::now()};
    auto elapsed{end - start};
    return {std::move(output), elapsed};
}

template <typename T>
std::chrono::duration<double, std::milli>
benchmark_case(std::size_t input_width)
{
    const std::size_t input_height{input_width};
    std::vector<T> input{make_input<T>(input_width, input_height)};
    std::vector<T> kernel{make_kernel<T>()};
    auto [output, elapsed]{convolve_optimized(
        input, input_width, input_height, kernel, KERNEL_WIDTH, KERNEL_HEIGHT)};
    benchmark_sink ^= checksum(output);
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
    input_width += 2 * (KERNEL_WIDTH / 2);

    benchmark_case<std::int32_t>(1024); // Warmup
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
        std::cout << "CPU (with AVX2/AVX-512) time using "
                  << TYPE_NAME(std::int16_t) << ": " << time_int16.count()
                  << " ms" << std::endl;
        std::cout << "CPU (with AVX2/AVX-512) time using "
                  << TYPE_NAME(std::int32_t) << ": " << time_int32.count()
                  << " ms" << std::endl;
        std::cout << "CPU (with AVX2/AVX-512) time using " << TYPE_NAME(float)
                  << ": " << time_float32.count() << " ms" << std::endl;
        std::cout << "CPU (with AVX2/AVX-512) time using " << TYPE_NAME(double)
                  << ": " << time_float64.count() << " ms" << std::endl;
    }
}
