let results_file = "benchmark_results.csv"

if ($results_file | path exists) {
    rm $results_file
}

let vec_len = [10 100 1000 10000 100000 1000000 10000000 100000000 200000000]

def benchmark [len: int] {
    let $result = 1..10 | each { |item|
        let result = ./build/windows/x64/release/homework-2.exe $len --benchmark | lines
        let cpu_time = $result.0 | into int
        let gpu_time = $result.1 | into int
        {cpu: $cpu_time, gpu: $gpu_time}
    }
    let cpu_time = $result.cpu | math avg
    let gpu_time = $result.gpu | math avg
    print $"Vector length: ($len), CPU Time: ($cpu_time), GPU Time: ($gpu_time)"
    let result_str = $"($len), ($cpu_time), ($gpu_time)\n"
    $result_str | save --append $results_file
}

$vec_len | each {|item| benchmark $item } | ignore

wolframscript -f data-process.wls
