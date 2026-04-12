lg[x_] := Log[10, x]
data = lg /@ Import["benchmark_results.csv"]

cpuData = data[[All, {1, 2}]]
gpuData = data[[All, {1, 3}]]
ratio = Transpose[{data[[All, 1]], data[[All, 2]] - data[[All, 3]]}]

plot = ListLinePlot[{cpuData, gpuData, ratio}, PlotRange -> All]

Export["diag.png", plot]