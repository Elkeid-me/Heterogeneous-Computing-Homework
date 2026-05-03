#include "utils.hxx"
#include <bit>

std::uint16_t double_to_half(double value)
{
    const uint64_t bits{std::bit_cast<uint64_t>(value)};
    const uint64_t sign{bits & 0x8000000000000000};
    const uint64_t exponent{bits & 0x7FF0000000000000};
    const uint64_t mantissa{bits & 0x000FFFFFFFFFFFFF};

    if (exponent == 0x7FF0000000000000)
    {
        // Inf or NaN
        return static_cast<std::uint16_t>(sign >> 48) |
               static_cast<std::uint16_t>(0x7C00) |
               static_cast<std::uint16_t>(mantissa != 0);
    }

    const int32_t new_exponent =
        static_cast<int32_t>(exponent >> 52) - 1023 + 15;
    if (new_exponent >= 31)
    {
        // Overflow, return Inf
        return static_cast<std::uint16_t>(sign >> 48) |
               static_cast<std::uint16_t>(0x7C00);
    }
    else if (new_exponent <= 0)
    {
        // Underflow, return zero
        return static_cast<std::uint16_t>(sign >> 48);
    }

    const uint64_t new_mantissa = mantissa >> 42;
    return static_cast<std::uint16_t>(sign >> 48) |
           static_cast<std::uint16_t>(new_exponent << 10) |
           static_cast<std::uint16_t>(new_mantissa);
}

template <>
std::uint64_t checksum<half_t>(const std::vector<half_t> &values)
{
    std::uint64_t state{0};
    for (const auto &value : values)
    {
        const long double sample{static_cast<long double>(value.bits()) *
                                 1024.0l};
        const std::uint64_t mixed{
            static_cast<std::uint64_t>(std::llround(sample))};
        state = state * 1315423911u + mixed + 0x9e3779b97f4a7c15ull;
    }
    return state;
}

volatile std::uint64_t benchmark_sink{0};
