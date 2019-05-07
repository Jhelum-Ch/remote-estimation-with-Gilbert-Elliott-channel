@everywhere begin
    include("algorithms.jl")
end

using DataFrames
import CSV

data = readtable("k_summary.csv", header = true)



@inline function sample_average_for_C(discount, cost; iterations=1000000)
   ellMean = zeros(length(data[:q1]))
   emmMean = zeros(length(data[:q1]))
   ellUpper = zeros(length(data[:q1]))
   emmUpper = zeros(length(data[:q1]))
   ellLower = zeros(length(data[:q1]))
   emmLower = zeros(length(data[:q1]))
   C_mean = zeros(length(data[:q1]))
   C_upper= zeros(length(data[:q1]))
   C_lower = zeros(length(data[:q1]))
   p = [0.7,0.2]

    for j in 1:length(data[:q1])
        for i in 1:iterations 
            q1 = data[:q1][j]
            q2 = data[:q2][j]
            Q = [q1, 1-q1;q2, 1-q2]
            channel = Channel(Q,p)

             
            threshold = [data[:k0Mean][j], data[:k1Mean][j]]
            thresholdUpper = [data[:k0Upper][j], data[:k1Upper][j]]
            thresholdLower = [data[:k0Lower][j], data[:k1Lower][j]]


            L_mean, M_mean= sample(channel, threshold, cost, discount)
            ellMean[j] += L_mean
            emmMean[j] += M_mean

            L_upper, M_upper= sample(channel, thresholdUpper, cost, discount)
            ellUpper[j] += L_upper
            emmUpper[j] += M_upper

            L_lower, M_lower= sample(channel, thresholdLower, cost, discount)
            ellLower[j] += L_lower
            emmLower[j] += M_lower
            #kay += K
        end
        ellMean[j] /= iterations
        emmMean[j] /= iterations
        ellUpper[j] /= iterations
        emmUpper[j]/= iterations
        ellLower[j] /= iterations
        emmLower[j] /= iterations
        #kay /= iterations

        C_mean[j] = ellMean[j]/emmMean[j]
        C_upper[j] = ellUpper[j]/emmUpper[j]
        C_lower[j] = ellLower[j]/emmLower[j]
    end

    statsC = DataFrame(L=data[:L], 
                     C_mean=C_mean, C_upper = C_upper, C_lower = C_lower)
    #writetable("C_summary.tsv", statsC, separator='\t', header = true)
    CSV.write("C_summary.tsv", statsC, delim='\t', header = true)

    return statsC
end

# @time sample_average_for_C(0.9, 0.3, 0.1)
@time sample_average_for_C(0.9, 100.0)
