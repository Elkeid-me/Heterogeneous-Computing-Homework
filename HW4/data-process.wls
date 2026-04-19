lg[x_] := Log[2, x]
data = lg /@ Import["benchmark_results_opencl.csv"]

intData = data[[All, {1, 2}]]
longData = data[[All, {1, 3}]]
floatData = data[[All, {1, 4}]]
doubleData = data[[All, {1, 5}]]
plot = ListLinePlot[{intData, longData, floatData, doubleData},
                    PlotRange -> All,
                    PlotLegends -> {"Int", "Long", "Float", "Double"},
                    AxesLabel -> {"log (Matrix Width)", "log (Time)"}]

Export["diag.png", plot]

data = lg /@ Import["benchmark_results_sycl_cuda.csv"]

syclInt = data[[All, {1, 2}]]
syclFloat = data[[All, {1, 3}]]
cudaInt = data[[All, {1, 4}]]
cudaFloat = data[[All, {1, 5}]]
plot = ListLinePlot[{syclInt, syclFloat, cudaInt, cudaFloat},
                    PlotRange -> All,
                    PlotLegends -> {"SYCL Int", "SYCL Float", "CUDA Int", "CUDA Float"},
                    AxesLabel -> {"log (Matrix Width)", "log (Time)"}]

Export["diag-2.png", plot]
