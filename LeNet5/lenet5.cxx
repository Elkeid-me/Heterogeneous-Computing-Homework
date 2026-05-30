#include "lenet5_weights.hxx"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>
#include <mdspan>
#include <memory>
#include <string>
#include <vector>

template <std::size_t... Extents>
using float_mdspan = std::mdspan<float, std::extents<std::size_t, Extents...>>;
template <std::size_t... Extents>
using const_float_mdspan =
    std::mdspan<const float, std::extents<std::size_t, Extents...>>;

// Just no padding and stride 1 for simplicity.
template <std::size_t input_channels, std::size_t input_height,
          std::size_t input_width, std::size_t output_channels,
          std::size_t kernel_size>
std::vector<float>
conv(const_float_mdspan<input_channels, input_height, input_width> input,
     const_float_mdspan<output_channels, input_channels, kernel_size,
                        kernel_size>
         kernels,
     const_float_mdspan<output_channels> bias)
{
    static_assert(kernel_size % 2 == 1, "Only odd kernel sizes are supported.");
    constexpr std::size_t output_height{input_height - kernel_size + 1};
    constexpr std::size_t output_width{input_width - kernel_size + 1};
    std::vector<float> output(output_channels * output_height * output_width);
    float_mdspan<output_channels, output_height, output_width> output_mdspan{
        output.data()};
    for (std::size_t oc{0}; oc < output_channels; oc++)
    {
        for (std::size_t oh{0}; oh < output_height; oh++)
        {
            for (std::size_t ow{0}; ow < output_width; ow++)
            {
                float sum{bias[oc]};
                for (std::size_t ic{0}; ic < input_channels; ic++)
                {
                    for (std::size_t kh{0}; kh < kernel_size; kh++)
                    {
                        for (std::size_t kw{0}; kw < kernel_size; kw++)
                        {
                            sum += input[ic, oh + kh, ow + kw] *
                                   kernels[oc, ic, kh, kw];
                        }
                    }
                }
                output_mdspan[oc, oh, ow] = sum;
            }
        }
    }
    return output;
}

std::vector<float> relu(std::vector<float> input)
{
    for (std::size_t i{0}; i < input.size(); i++)
        input[i] = std::max(0.0f, input[i]);
    return input;
}

// Just no padding, kernel size 2, and stride 2 for simplicity.
template <std::size_t input_channels, std::size_t input_height,
          std::size_t input_width>
std::vector<float>
max_pooling(const_float_mdspan<input_channels, input_height, input_width> input)
{
    constexpr std::size_t output_height{input_height / 2};
    constexpr std::size_t output_width{input_width / 2};
    std::vector<float> output(input_channels * output_height * output_width);
    float_mdspan<input_channels, output_height, output_width> output_mdspan{
        output.data()};
    for (std::size_t c{0}; c < input_channels; c++)
    {
        for (std::size_t h{0}; h < output_height; h++)
        {
            for (std::size_t w{0}; w < output_width; w++)
                output_mdspan[c, h, w] = std::max(
                    {input[c, h * 2, w * 2], input[c, h * 2, w * 2 + 1],
                     input[c, h * 2 + 1, w * 2],
                     input[c, h * 2 + 1, w * 2 + 1]});
        }
    }
    return output;
}

template <std::size_t input_size, std::size_t output_size>
std::vector<float> linear(std::vector<float> input,
                          const_float_mdspan<output_size, input_size> weights,
                          const_float_mdspan<output_size> bias)
{
    std::vector<float> output(output_size);
    for (std::size_t o{0}; o < output_size; o++)
    {
        float sum{bias[o]};
        for (std::size_t i{0}; i < input_size; i++)
            sum += input[i] * weights[o, i];
        output[o] = sum;
    }
    return output;
}

int lenet(const_float_mdspan<1, 32, 32> input)
{
    // 0
    const_float_mdspan<6, 1, 5, 5> conv0_weights{features_0_weight};
    const_float_mdspan<6> conv0_bias{features_0_bias};
    std::vector<float> conv0_output{conv(input, conv0_weights, conv0_bias)};
    std::vector<float> relu0_output{relu(std::move(conv0_output))};
    std::vector<float> pool0_output{
        max_pooling(const_float_mdspan<6, 28, 28>(relu0_output.data()))};
    // 1
    const_float_mdspan<16, 6, 5, 5> conv1_weights{features_3_weight};
    const_float_mdspan<16> conv1_bias{features_3_bias};
    std::vector<float> conv1_output{
        conv(const_float_mdspan<6, 14, 14>(pool0_output.data()), conv1_weights,
             conv1_bias)};
    std::vector<float> relu1_output{relu(std::move(conv1_output))};
    std::vector<float> pool1_output{
        max_pooling(const_float_mdspan<16, 10, 10>(relu1_output.data()))};
    // --- Flatten ---
    // 0
    const_float_mdspan<120, 400> fc0_weights{classifier_1_weight};
    const_float_mdspan<120> fc0_bias{classifier_1_bias};
    std::vector<float> fc0_output{
        relu(linear(std::move(pool1_output), fc0_weights, fc0_bias))};
    // 1
    const_float_mdspan<84, 120> fc1_weights{classifier_3_weight};
    const_float_mdspan<84> fc1_bias{classifier_3_bias};
    std::vector<float> fc1_output{
        relu(linear(std::move(fc0_output), fc1_weights, fc1_bias))};
    // 2
    const_float_mdspan<10, 84> fc2_weights{classifier_5_weight};
    const_float_mdspan<10> fc2_bias{classifier_5_bias};
    std::vector<float> fc2_output{
        linear(std::move(fc1_output), fc2_weights, fc2_bias)};
    return std::max_element(fc2_output.begin(), fc2_output.end()) -
           fc2_output.begin();
}

int main(int argc, char *argv[])
{
    static_assert(sizeof(float) == 4, "This code assumes float is 4 bytes.");
    static_assert(std::endian::native == std::endian::little,
                  "This code assumes little-endian architecture.");
    if (argc != 2 && argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <img_index>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::ifstream input_file(argv[1],
                             std::ios_base::in | std::ios_base::binary);
    if (!input_file)
    {
        std::cerr << "Error: Could not open input file." << std::endl;
        return EXIT_FAILURE;
    }
    auto input{std::make_unique<float[]>(32 * 32)};
    input_file.seekg(argc == 3 ? std::stoul(argv[2]) * 32 * 32 * sizeof(float)
                               : 0,
                     std::ios_base::beg);
    input_file.read(reinterpret_cast<char *>(input.get()),
                    32 * 32 * sizeof(float));
    std::cout << lenet(const_float_mdspan<1, 32, 32>(input.get())) << std::endl;
}
