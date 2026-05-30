#include "lenet5_weights.hxx"
#include <iostream>
#include <mdspan>
#include <ranges>
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
    std::vector<float> output;
    constexpr std::size_t output_height{input_height - kernel_size + 1};
    constexpr std::size_t output_width{input_width - kernel_size + 1};
    output.reserve(input_channels * output_height * output_width);
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
    std::vector<float> output;
    output.reserve(input_channels * output_height * output_width);
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



int lennet(const_float_mdspan<1, 28, 28> input)
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
        max_pooling(const_float_mdspan<16, 5, 5>(relu1_output.data()))};

    // // 2
    // const_float_mdspan<120, 16, 5, 5> conv2_weights{features_6_weight};
    // const_float_mdspan<120> conv2_bias{features_6_bias};
    // std::vector<float> conv2_output{
    //     conv(const_float_mdspan<16, 2, 2>(pool1_output.data()), conv2_weights,
    //          conv2_bias)};

    const_float_mdspan<84, 120> fc0_weights{classifier_1_weight};
    const_float_mdspan<84> fc0_bias{classifier_1_bias};
    const_float_mdspan<10, 84> fc1_weights{classifier_3_weight};
    const_float_mdspan<10> fc1_bias{classifier_3_bias};
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
