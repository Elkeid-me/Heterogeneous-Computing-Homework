lg[x_] := Log[2, x]
data = lg /@ Import["benchmark_results.csv"]

intData = data[[All, {1, 2}]]
longData = data[[All, {1, 3}]]
floatData = data[[All, {1, 4}]]
doubleData = data[[All, {1, 5}]]
plot = ListLinePlot[{intData, longData, floatData, doubleData},
                    PlotRange -> All,
                    PlotLegends -> {"Int", "Long", "Float", "Double"},
                    AxesLabel -> {"log (Matrix Width)", "log (Time)"}]

Export["diag.png", plot]