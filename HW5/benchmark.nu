let results_file = "benchmark_results.csv"

if ($results_file | path exists) {
    rm $results_file
}

let input = $env.PWD | path join "Microsoft_Wallpaper.bmp"
let output = $env.PWD | path join "Microsoft_Wallpaper-rotated.bmp"
let result = 1..10 | each { |item|
    let opencl_time = timeit { xmake r bmp-rotate-opencl $input $output 60 | ignore }
    let cpu_open_cl_time = timeit { xmake r bmp-rotate-cpu-opencl $input $output 60 | ignore }
    {opencl: $opencl_time, cpu_open_cl: $cpu_open_cl_time}
}
let opencl_time = $result.opencl | math avg
let cpu_opencl_time = $result.cpu_open_cl | math avg
print $"OpenCL Time: ($opencl_time), CPU Time: ($cpu_opencl_time)"
let opencl_time_ms = $opencl_time / 1ms
let cpu_opencl_time_ms = $cpu_opencl_time / 1ms
let result_str = $"($opencl_time_ms), ($cpu_opencl_time_ms)\n"
$result_str | save --append $results_file
