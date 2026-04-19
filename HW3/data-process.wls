lg[x_] := Log[2, x]
data = lg /@ Import["benchmark_results.csv"]

floatData = data[[All, {1, 2}]]
doubleData = data[[All, {1, 3}]]
tiledFloatData = data[[All, {1, 4}]]
tiledDoubleData = data[[All, {1, 5}]]
plot = ListLinePlot[{floatData, doubleData, tiledFloatData, tiledDoubleData},
                    PlotRange -> All,
                    PlotLegends -> {"Float", "Double", "Tiled Float", "Tiled Double"},
                    AxesLabel -> {"log (Matrix Width)", "log (Time)"}]

Export["diag.png", plot]