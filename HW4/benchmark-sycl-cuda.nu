let results_file = "benchmark_results_sycl_cuda.csv"

if ($results_file | path exists) {
    rm $results_file
}

def build [] {
    acpp -O3 -o build/sycl-matrix-mul src/sycl-matrix-mul.cxx
    xmake b cuda-matrix-mul
}

let vec_len = [1024 2048 4096]

def benchmark [len: int] {
    let $sycl_result = 1..10 | each { |item|
        let result = ./build/sycl-matrix-mul $len --benchmark | lines
        let int_time = $result.0 | into int
        let float_time = $result.1 | into int
        {int: $int_time, float: $float_time}
    }
    let $cuda_result = 1..10 | each { |item|
        let result = xmake r cuda-matrix-mul $len --benchmark | lines
        let int_time = $result.0 | into int
        let float_time = $result.1 | into int
        {int: $int_time, float: $float_time}
    }
    let sycl_int_time = $sycl_result.int | math avg
    let sycl_float_time = $sycl_result.float | math avg
    let cuda_int_time = $cuda_result.int | math avg
    let cuda_float_time = $cuda_result.float | math avg
    print $"Matrix width: ($len), Int Time: ($sycl_int_time), Float Time: ($sycl_float_time), CUDA Int Time: ($cuda_int_time), CUDA Float Time: ($cuda_float_time)"
    let result_str = $"($len), ($sycl_int_time), ($sycl_float_time), ($cuda_int_time), ($cuda_float_time)\n"
    $result_str | save --append $results_file
}

build
$vec_len | each {|item| benchmark $item } | ignore

# wolframscript -f data-process.wls
