let results_file = if (sys cpu | get 0.brand) =~ 11370H {
    "benchmark_results_11370H.csv"
} else {
    "benchmark_results_125H.csv"
}

if ($results_file | path exists) {
    rm $results_file
}

def benchmark [len: int] {
    let $cpu_result = 1..10 | each { |item|
        let result = xmake r convolution-cpu $len --benchmark | lines
        let int8_time = $result.0 | into float
        let int16_time = $result.1 | into float
        let int32_time = $result.2 | into float
        let f32_time = $result.3 | into float
        let f64_time = $result.4 | into float
        {i8: $int8_time, i16: $int16_time, i32: $int32_time, f32: $f32_time, f64: $f64_time}
    }
    let $avx2_result = 1..10 | each { |item|
        let result = xmake r convolution-cpu-optimized $len --benchmark | lines
        let int16_time = $result.0 | into float
        let int32_time = $result.1 | into float
        let f32_time = $result.2 | into float
        let f64_time = $result.3 | into float
        {i16: $int16_time, i32: $int32_time, f32: $f32_time, f64: $f64_time}
    }
    let $opencl_result = 1..10 | each { |item|
        let result = xmake r convolution-opencl $len --benchmark | lines
        let int8_time = $result.0 | into float
        let int16_time = $result.1 | into float
        let int32_time = $result.2 | into float
        let f16_time = $result.3 | into float
        let f32_time = $result.4 | into float
        let f64_time = $result.5 | into float
        {i8: $int8_time, i16: $int16_time, i32: $int32_time, f16: $f16_time, f32: $f32_time, f64: $f64_time}
    }
    let $opencl_optimized_result = 1..10 | each { |item|
        let result = xmake r convolution-opencl-optimized $len --benchmark | lines
        let int8_time = $result.0 | into float
        let int16_time = $result.1 | into float
        let int32_time = $result.2 | into float
        let f16_time = $result.3 | into float
        let f32_time = $result.4 | into float
        let f64_time = $result.5 | into float
        {i8: $int8_time, i16: $int16_time, i32: $int32_time, f16: $f16_time, f32: $f32_time, f64: $f64_time}
    }
    let cpu_i8_time = $cpu_result.i8 | math avg
    let cpu_i16_time = $cpu_result.i16 | math avg
    let cpu_i32_time = $cpu_result.i32 | math avg
    let cpu_f32_time = $cpu_result.f32 | math avg
    let cpu_f64_time = $cpu_result.f64 | math avg
    let avx2_i16_time = $avx2_result.i16 | math avg
    let avx2_i32_time = $avx2_result.i32 | math avg
    let avx2_f32_time = $avx2_result.f32 | math avg
    let avx2_f64_time = $avx2_result.f64 | math avg
    let opencl_i8_time = $opencl_result.i8 | math avg
    let opencl_i16_time = $opencl_result.i16 | math avg
    let opencl_i32_time = $opencl_result.i32 | math avg
    let opencl_f16_time = $opencl_result.f16 | math avg
    let opencl_f32_time = $opencl_result.f32 | math avg
    let opencl_f64_time = $opencl_result.f64 | math avg
    let opencl_optimized_i8_time = $opencl_optimized_result.i8 | math avg
    let opencl_optimized_i16_time = $opencl_optimized_result.i16 | math avg
    let opencl_optimized_i32_time = $opencl_optimized_result.i32 | math avg
    let opencl_optimized_f16_time = $opencl_optimized_result.f16 | math avg
    let opencl_optimized_f32_time = $opencl_optimized_result.f32 | math avg
    let opencl_optimized_f64_time = $opencl_optimized_result.f64 | math avg
    print $"Input width: ($len)"
    print $"CPU i8 Time: ($cpu_i8_time), i16: ($cpu_i16_time), i32: ($cpu_i32_time), f32: ($cpu_f32_time), f64: ($cpu_f64_time)"
    print $"AVX2/AVX-512 i16 Time: ($avx2_i16_time), i32: ($avx2_i32_time), f32: ($avx2_f32_time), f64: ($avx2_f64_time)"
    print $"OpenCL i8 Time: ($opencl_i8_time), i16: ($opencl_i16_time), i32: ($opencl_i32_time), f16: ($opencl_f16_time), f32: ($opencl_f32_time), f64: ($opencl_f64_time)"
    print $"OpenCL Optimized i8 Time: ($opencl_optimized_i8_time), i16: ($opencl_optimized_i16_time), i32: ($opencl_optimized_i32_time), f16: ($opencl_optimized_f16_time), f32: ($opencl_optimized_f32_time), f64: ($opencl_optimized_f64_time)"

    $"($cpu_i8_time), 0, ($opencl_i8_time), ($opencl_optimized_i8_time)\n" | save --append $results_file
    $"($cpu_i16_time), ($avx2_i16_time), ($opencl_i16_time), ($opencl_optimized_i16_time)\n" | save --append $results_file
    $"($cpu_i32_time), ($avx2_i32_time), ($opencl_i32_time), ($opencl_optimized_i32_time)\n" | save --append $results_file
    $"0, 0, ($opencl_f16_time), ($opencl_optimized_f16_time)\n" | save --append $results_file
    $"($cpu_f32_time), ($avx2_f32_time), ($opencl_f32_time), ($opencl_optimized_f32_time)\n" | save --append $results_file
    $"($cpu_f64_time), ($avx2_f64_time), ($opencl_f64_time), ($opencl_optimized_f64_time)\n" | save --append $results_file
}

benchmark 8192

wolframscript -f data-process.wls
