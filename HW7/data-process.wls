data = Import["benchmark_results.csv"]
plot = BarChart[data,
                ChartLegends -> {"CPU", "CPU (AVX2)", "OpenCL", "OpenCL (Optimized)"},
                ChartLabels -> {{"i8", "i16", "i32", "f16", "f32", "f64"}, None}]
Export["diag.png", plot]
