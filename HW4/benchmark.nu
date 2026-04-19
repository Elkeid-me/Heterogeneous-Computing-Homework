let results_file = "benchmark_results.csv"

if ($results_file | path exists) {
    rm $results_file
}

let vec_len = [1024 2048 4096 8192]

def benchmark [len: int] {
    let $result = 1..10 | each { |item|
        let result = ./build/windows/x64/release/matrix-mul $len --benchmark | lines
        let int_time = $result.0 | into int
        let long_time = $result.1 | into int
        let float_time = $result.2 | into int
        let double_time = $result.3 | into int
        {int: $int_time, long: $long_time, float: $float_time, double: $double_time}
    }
    let int_time = $result.int | math avg
    let long_time = $result.long | math avg
    let float_time = $result.float | math avg
    let double_time = $result.double | math avg
    print $"Matrix width: ($len), Int Time: ($int_time), Long Time: ($long_time), Float Time: ($float_time), Double Time: ($double_time)"
    let result_str = $"($len), ($int_time), ($long_time), ($float_time), ($double_time)\n"
    $result_str | save --append $results_file
}

$vec_len | each {|item| benchmark $item } | ignore

# wolframscript -f data-process.wls
