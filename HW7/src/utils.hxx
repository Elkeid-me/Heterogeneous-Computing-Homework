#ifndef UTILS_HXX
#define UTILS_HXX

#include <random>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr std::size_t KERNEL_WIDTH{7};
constexpr std::size_t KERNEL_HEIGHT{KERNEL_WIDTH};

template <typename T>
std::vector<T> make_input(const std::size_t width, const std::size_t height)
{
    constexpr std::size_t half_kernel_width{KERNEL_WIDTH / 2};
    constexpr std::size_t half_kernel_height{KERNEL_HEIGHT / 2};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 256.0);
    std::vector<T> values(width * height);
    for (std::size_t row{0}; row < height; row++)
    {
        for (std::size_t col{0}; col < width; col++)
        {
            if (row < half_kernel_height ||
                row >= height - half_kernel_height || col < half_kernel_width ||
                col >= width - half_kernel_width)
                values[row * width + col] = 0;
            else
                values[row * width + col] = dis(gen);
        }
    }
    return values;
}

template <typename T>
std::vector<T> make_kernel()
{
    std::vector<T> values(KERNEL_WIDTH * KERNEL_HEIGHT);
    const std::size_t center_x{KERNEL_WIDTH / 2};
    const std::size_t center_y{KERNEL_HEIGHT / 2};
    for (std::size_t row{0}; row < KERNEL_HEIGHT; row++)
    {
        for (std::size_t col{0}; col < KERNEL_WIDTH; col++)
        {
            const int dx{static_cast<int>(col - center_x)};
            const int dy{static_cast<int>(row - center_y)};
            const int distance{std::abs(dx) + std::abs(dy)};
            switch (distance)
            {
            case 0:
                values[row * KERNEL_WIDTH + col] = 8;
                break;
            case 1:
                values[row * KERNEL_WIDTH + col] = -2;
                break;
            case 2:
            case 3:
                values[row * KERNEL_WIDTH + col] = -1;
                break;
            default:
                values[row * KERNEL_WIDTH + col] = 0;
                break;
            }
        }
    }
    return std::move(values);
}

template <typename T>
struct sparse_stencil
{
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> col_offsets;
    std::vector<T> weights;
    std::size_t width{0};
    std::size_t height{0};
};

template <typename T>
sparse_stencil<T> make_sparse_stencil(const std::vector<T> &dense,
                                      const std::size_t width,
                                      const std::size_t height)
{
    sparse_stencil<T> stencil;
    stencil.width = width;
    stencil.height = height;
    for (std::size_t row{0}; row < height; row++)
    {
        for (std::size_t col{0}; col < width; col++)
        {
            const T value{dense[row * width + col]};
            if (value != T{})
            {
                stencil.row_offsets.push_back(static_cast<std::int32_t>(row));
                stencil.col_offsets.push_back(static_cast<std::int32_t>(col));
                stencil.weights.push_back(value);
            }
        }
    }
    return stencil;
}

template <typename T>
std::tuple<std::vector<T>, std::chrono::duration<double, std::milli>>
convolve_cpu_unoptimized(const std::vector<T> &input,
                         const std::size_t input_width,
                         const std::size_t input_height,
                         const std::vector<T> &kernel,
                         const std::size_t kernel_width,
                         const std::size_t kernel_height)
{
    const std::size_t output_width{input_width - kernel_width + 1};
    const std::size_t output_height{input_height - kernel_height + 1};
    std::vector<T> output(output_width * output_height);

    const auto start{std::chrono::high_resolution_clock::now()};
    for (std::size_t out_y{0}; out_y < output_height; out_y++)
    {
        for (std::size_t out_x{0}; out_x < output_width; out_x++)
        {
            T sum{0};
            for (std::size_t kernel_y{0}; kernel_y < kernel_height; kernel_y++)
            {
                for (std::size_t kernel_x{0}; kernel_x < kernel_width;
                     kernel_x++)
                {
                    const std::size_t input_index{
                        (out_y + kernel_y) * input_width + out_x + kernel_x};
                    const std::size_t kernel_index{kernel_y * kernel_width +
                                                   kernel_x};
                    sum += kernel[kernel_index] * input[input_index];
                }
            }
            output[out_y * output_width + out_x] = sum;
        }
    }
    const auto end{std::chrono::high_resolution_clock::now()};
    const auto elapsed{end - start};
    return {std::move(output), elapsed};
}

template <typename T>
struct type_name;
#define DEFINE_TYPE_NAME(TYPE, NAME)                                           \
    template <>                                                                \
    struct type_name<TYPE>                                                     \
    {                                                                          \
        constexpr static const char *value{NAME};                              \
    };

std::uint16_t double_to_half(double value);

class half_t
{
public:
    half_t() = default;
    half_t(double value) : m_bits{double_to_half(value)} {}
    half_t &operator=(double value)
    {
        *this = half_t(value);
        return *this;
    }
    std::uint16_t bits() const { return m_bits; }
    friend auto operator<=>(const half_t &lhs, const half_t &rhs) = default;

private:
    std::uint16_t m_bits{0};
};

DEFINE_TYPE_NAME(std::int8_t, "int8");
DEFINE_TYPE_NAME(std::int16_t, "int16");
DEFINE_TYPE_NAME(std::int32_t, "int32");
DEFINE_TYPE_NAME(float, "float");
DEFINE_TYPE_NAME(double, "double");
DEFINE_TYPE_NAME(half_t, "half");
#define TYPE_NAME(T) type_name<T>::value

template <typename T>
std::uint64_t checksum(const std::vector<T> &values)
{
    std::uint64_t state{0};
    for (const auto &value : values)
    {
        const long double sample{static_cast<long double>(value) * 1024.0l};
        const std::uint64_t mixed{
            static_cast<std::uint64_t>(std::llround(sample))};
        state = state * 1315423911u + mixed + 0x9e3779b97f4a7c15ull;
    }
    return state;
}

template <>
std::uint64_t checksum<half_t>(const std::vector<half_t> &values);

extern volatile std::uint64_t benchmark_sink;

#endif // UTILS_HXX
