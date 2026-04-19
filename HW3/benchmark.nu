let results_file = "benchmark_results.csv"

if ($results_file | path exists) {
    rm $results_file
}

let vec_len = [1024 2048 4096 8192]

def benchmark [len: int] {
    let $simple_result = 1..10 | each { |item|
        let result = xmake r matrix-mul $len --benchmark | lines
        let float_time = $result.0 | into int
        let double_time = $result.1 | into int
        {float: $float_time, double: $double_time}
    }
    let $tiled_result = 1..10 | each { |item|
        let result = xmake r matrix-mul-optimized $len --benchmark | lines
        let float_time = $result.0 | into int
        let double_time = $result.1 | into int
        {float: $float_time, double: $double_time}
    }
    let float_time = $simple_result.float | math avg
    let double_time = $simple_result.double | math avg
    let tiled_float_time = $tiled_result.float | math avg
    let tiled_double_time = $tiled_result.double | math avg
    print $"Matrix width: ($len), Float Time: ($float_time), Double Time: ($double_time), Tiled Float Time: ($tiled_float_time), Tiled Double Time: ($tiled_double_time)"
    let result_str = $"($len), ($float_time), ($double_time), ($tiled_float_time), ($tiled_double_time)\n"
    $result_str | save --append $results_file
}

$vec_len | each {|item| benchmark $item } | ignore

wolframscript -f data-process.wls
