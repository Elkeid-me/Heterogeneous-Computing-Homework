data = Import["benchmark_results_125H.csv"]

plot = BarChart[data,
                ChartLegends -> {"CPU (125H)", "CPU (125H, AVX2)", "OpenCL (Arc)", "OpenCL (Arc, Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag.png", plot]

plot = BarChart[data[[All, 2;;4]],
                ChartLegends -> {"CPU (125H, AVX2)", "OpenCL (Arc)", "OpenCL (Arc, Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag-without-cpu.png", plot]

data2 = Import["benchmark_results_11370H.csv"]

plot = BarChart[data2,
                ChartLegends -> {"CPU (11370H)", "CPU (11370H, AVX-512)", "OpenCL (MX450)", "OpenCL (MX450, Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag-2.png", plot]

plot = BarChart[data2[[All, 2;;4]],
                ChartLegends -> {"CPU (11370H, AVX-512)", "OpenCL (MX450)", "OpenCL (MX450, Optimized)"},
                ChartLabels -> {{"int8", "int16", "int32", "fp16", "fp32", "fp64"}, None}]
Export["diag-2-without-cpu.png", plot]
