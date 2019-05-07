using Distributed

if nprocs() == 1
  addprocs(Sys.CPU_THREADS)
end

include("simulations.jl")

# @time stats = generateOutput(100., 0.9, 100; 
#                         numRuns=10, saveSummaryData=true,
#                        saveRawData = true)

const numRuns = 100
#const cost    = 100.0
const iterations = 50_000
#const discount = 0.9
const gradient = :TwoSided


const alpha = 1
const c = 0.1

#const Q = [0.3 0.7; 0.1 0.9]
const p = [0.7,0.2]

#const channel = Channel(Q,p)

function computeTimeSeries(sa, costValues, discountValues, dropProbValues;
                           iterations=1000, initial = [1.0, 1.0], numRuns=100, 
			               saveSummaryData=true, saveRawData = false, savePlot=false,
                           labeltext=label_costly, ylim=:auto)

	for discount in discountValues, dropProb in dropProbValues
		result_k = Array(DataFrame, length(costValues))
        #result_C = Array(DataFrame, length(paramValues))

        for i in 1:length(costValues)
            cost = costValues[i]
            alpha1 = dropProb[1]
            beta = dropProb[2]
            Q = [alpha1, 1-alpha1;beta, 1-beta]

            channel = Channel(Q,p)

            stats_k = generateOutput(channel, cost, discount, iterations; 
		                    numRuns=numRuns, c = c, alpha = alpha, 
		                    gradient = gradient,
		                    saveSummaryData=true, saveRawData=false)

            result_k[i] = stats_k
            #result_C[i] = rc


		

            if savePlot
				plt = plotStats(stats_k)

				filename = string("plots/", gradient, 
				                  "__cost_", cost, "__discount_", discount, "__p_00_", alpha1, "__p_10_", beta,
				                  "__numRuns_", numRuns, "__c_", c, "__alpha_", 
				                  alpha)

				savefig(plt, "$filename.png")
			end
		end
	end
end

@time computeTimeSeries(sa_costly, [100.0], [0.9,1.0], zip([0.3,0.5,0.7,0.9], [0.1,0.4,0.8,0.9]); 
                  iterations=50_000, initial = [1.0, 1.0], numRuns=100, saveSummaryData = true, saveRawData = false, 
                  savePlot=false, labeltext=label_costly, ylim=(0,13))
