let results_file = "benchmark_results.csv"

if ($results_file | path exists) {
    rm $results_file
}

let root = $env.PWD

def bench_one [input output] {
    let opencl_times = 1..10 | each {
        ./Rotate.exe -i $input -o $output -d 60 -e opencl | into float
    }
    let cpu_times = 1..10 | each {
        ./Rotate.exe -i $input -o $output -d 60 -e cpu | into float
    }

    {
        input: $input,
        opencl_ms: (($opencl_times | math avg)),
        cpu_ms: (($cpu_times | math avg)),
    }
}

let results = [
    (bench_one ($root | path join "Libbie-Front.png") ($root | path join "Libbie-Front-Rotated.png"))
    (bench_one ($root | path join "Microsoft-Wallpaper.png") ($root | path join "Microsoft-Wallpaper-Rotated.png"))
]

$results | to csv | save -f $results_file
print $"Benchmark results saved to ($results_file)"
