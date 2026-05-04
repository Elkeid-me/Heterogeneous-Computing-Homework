data = Import["benchmark_results.csv"]

plot = BarChart[data,
                ChartLegends -> {"CPU", "CPU (AVX2/AVX-512)", "OpenCL", "OpenCL (Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag.png", plot]

plot = BarChart[data[[All,2;;4]],
                ChartLegends -> {"CPU (AVX2/AVX-512)", "OpenCL", "OpenCL (Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag-without-cpu.png", plot]

