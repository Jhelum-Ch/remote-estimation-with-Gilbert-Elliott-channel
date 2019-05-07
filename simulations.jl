# ---
# name: simulations.jl
# author: Jhelum Chakravorty, Aditya Mahajan
# date: 1 Feb, 2017
# license: BSD 3 clause
# ---
#

@everywhere begin
    using Dates
    using ProgressMeter
    include("algorithms.jl")
end

# Traces from multiple runs
# =========================
#
# To visualize how well the stochastic approximation algorithm (implemented in
# `algorithms.jl`) work, we run them multiple times for any particular choice
# of parameters. Since this task is an embarrassingly parallel one, we run
# each trace on a separate core and then combine all the traces. This is
# implemented in the `generateTraces` function below. 
#
# The output is an array of size 2, where each component is `iterations *
# numRuns`


function generateTraces(channel, cost, discount, iterations; 
                        numRuns = 100, initial=[1.0, 1.0],
                        c = 0.1, alpha=0.01, debug=false,
                        gradient=:OneSided)
    tuples = @showprogress 10 "Generating Traces ..." pmap(1:numRuns) do run
        sa_costly(channel, cost, discount; iterations=iterations, initial=initial,
                 c = c, alpha=alpha, debug=debug, gradient=gradient)
    end

    traces = [zeros(iterations, numRuns) for S in S0:S1] 

    for S in S0:S1, run in 1:numRuns
        traces[S][:,run] = tuples[run][:,S]
    end
    return traces
end

# Generating and saving results
# =============================
#
# The function `generateOutput` takes the same parameters as `generateTraces`
# plus an additional optional parameter: `saveRawData`, which defaults to
# `false`. It returns a data frame with six columns: mean
# value, mean + (standard deviation), mean - (standard deviation) for both
# values of thresholds. This data is saved to a tab
# separated file (in the `output/` directory). The filename includes the value
# of the cost and the discount factor.
#
# When `saveRawData` is set to `true`, the traces are saved to a
# `.jld` file. 

using  DataFrames
using  Statistics
import CSV
using  HDF5

function generateOutput(channel, cost, discount, iterations; initial=[1.0, 1.0],
    numRuns  = 100, saveSummaryData = true, saveRawData = false,
    c = 0.1, alpha=0.01, debug=false, gradient=:OneSided)

    traces = generateTraces(channel, cost, discount, iterations; 
                            initial = initial, numRuns = numRuns,
                            gradient = gradient)

    lower0 = zeros(iterations)
    lower1 = zeros(iterations)
    mean0  = zeros(iterations)
    mean1  = zeros(iterations)
    upper0 = zeros(iterations)
    upper1 = zeros(iterations)

    for i = 1:iterations
      lower0[i], mean0[i], upper0[i] = quantile(traces[S0][i,:], (0.25, 0.5, 0.75))
      lower1[i], mean1[i], upper1[i] = quantile(traces[S1][i,:], (0.25, 0.5, 0.75))
    end

    stats = DataFrame(mean0  = mean0, 
                      upper0 = upper0,
                      lower0 = lower0,
                      mean1  = mean1,
                      upper1 = upper1,
                      lower1 = lower1)

    # filename = string("output/", "cost_", cost, "__discount_" , discount, 
    #                   "__numruns_", numRuns )
    filename = string("output/", "cost_", cost, "__discount_" , discount, "__alpha1_", alpha1,
                      "__beta_", beta)

    if saveSummaryData
      CSV.write("$filename.tsv", stats, delim='\t')
    end

     if saveRawData
         h5open("$filename.h5", "w") do file
             @write file cost
             @write file discount
             @write file iterations
             @write file numRuns
             trace0 = traces[S0]
             trace1 = traces[S1]
             @write file trace0
             @write file trace1
             @write file mean0
             @write file mean1
             @write file std0
             @write file std1
         end
     end

    return stats
end

using Plots
# ENV["GKSwstype"]="100" 
# gr()
pyplot()


# A helper function to display labels

function labeltext(s, param)
  text(string('$', s, "=", param, '$'), 8, :bottom, :right)
end

function labeltext(s)
  text(string('$', s, '$'), 8, :bottom, :right)
end

function plotStats(stats)

    plt = plot(xlabel="Iterations", ylabel="Threshold", ylim=:auto)
    plot!(plt, stats[:upper0], linecolor=:lightblue, label="")
    plot!(plt, stats[:lower0], linecolor=:lightblue, label="",
          fillrange=stats[:upper0], fillcolor=:lightgray, fillalpha=0.8)
    plot!(plt, stats[:mean0], linecolor=:black, linewidth=2, label="")

    iterations=size(stats,1)

    x = div(3*iterations,4)
    y = stats[:upper0][x]
    label=labeltext("k(0)")

    plot!(plt, annotations=(x,y,label))

    plot!(plt, stats[:upper1], linecolor=:lightblue, label="")
    plot!(plt, stats[:lower1], linecolor=:lightblue, label="",
          fillrange=stats[:upper1], fillcolor=:lightgray, fillalpha=0.8)
    plot!(plt, stats[:mean1], linecolor=:black, linewidth=2, label="")

    y = stats[:upper1][x]
    label=labeltext("k(1)")

    plot!(plt, annotations=(x,y,label))

    return plt
end
