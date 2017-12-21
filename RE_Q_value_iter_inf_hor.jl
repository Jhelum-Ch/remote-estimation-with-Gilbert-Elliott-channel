# Julia code by Jhelum Chakravorty
# This code is for Remote Estimation problem with one transmitter and one receiver with an erasure channel. For reference, see
# 'Chakravorty J. and Mahajan A., ``Sufficient conditions for the value function and optimal strategy to be even and quasi-convex''


using Distributions
using StatsBase
#using PyPlot

const a = 1.0       # AR process parameter
#const q = 0.9       # Packet-pass probability
const λ = 1.0       # Communication cost
const β = 0.9       # Discount
const ϵ = 10.0^(-6)   # tolerance for convergance

# We first truncate the state space to [-L, L] and then discretize it into
# N points. 

const N = 1001
const L = 5
const X = linspace(-L, L, N)

# Next we create the voronoi boundaries of these grid points.
# Note that there are 2N+1 grid points, so there will be 2N+2 boundaries.
# We follow the convention that the lower boundary of grid point n indexed by
# n and the upper boundary is indexed by n+1

boundary = zeros(N+1)

boundary[1]   = -Inf
boundary[N+1] = Inf

for n = 2:N
    boundary[n] = (X[n-1]+X[n])/2
end

# The action space is binary. 0 means don't transmit and 1 means transmit
const U0 = 1
const U1 = 2
const U = [U0, U1]

# The GE channel states
const S0 = 1
const S1 = 2
const S = [S0, S1]


# Now, we discretize the probability distribution. For every grid point x[i],
# we calculate the probability that the transition takes us to the interval
# (boundary[j], boundary[j+1])

const W = Normal(0, 1)

P = [zeros(N, N) for u in U, s in S]
P_ch = [0.3 0.7;0.1 0.9] # We use Gilbert-Elliott channel

for s in S, i in 1:N, j in 1:N
    P[U0,s][i,j] = cdf(W, boundary[j+1] - a*X[i]) - cdf(W, boundary[j] - a*X[i])
    P[U1,s][i,j] = P_ch[s,2] * (cdf(W, boundary[j+1]) - cdf(W, boundary[j])) + 
                    P_ch[s,1]*P[U0][i,j]
end


# Per-step cost (note that action is stored as as u+1. So we subtract one)
cost(x,s,u) = λ*(u-1) + (1 - (u-1)*P_ch[s,1])*x*x
C = [zeros(N, length(U)) for s in S]


g_beta = [zeros(Int, N) for s in S]
V_beta = [zeros(N) for s in S]
function findV(Q_beta) 
  for s in S, n in 1:N
    idx = Q_beta[s][n,U0] <= Q_beta[s][n,U1] ? U0 : U1
    g_beta[s][n] = idx - 1  # Optimal strategy
    V_beta[s][n] = Q_beta[s][n,idx]
  end
  return V_beta, g_beta


end


for u in U, n in 1:N, s in S
  C[s][n,u] = cost(X[n], s, u) #Per-step cost
end

Q_beta = copy(C) #[zeros(N, length(U)) for s in S] #C
Q_old = [zeros(N,length(U)) for s in S] #[zeros(N, length(U)) for s in S]
for s in S, i in 1:N, j in 1:length(U)
  Q_old[s][i,j] = Q_beta[s][i,j]
end

(V_beta, g_beta) = findV(Q_beta)
V_old = [zeros(N) for s in S]



function frobNorm(Q_beta,Q_old) #Frobenious norm of matrices
  a_max = 0
  for s in S
    diffMat = Q_beta[s] - Q_old[s]
    a = trace(ctranspose(diffMat)*diffMat)
    sqrt(a)
    a_max = max(a_max, a) # Take the max value for s=0 or s=1
  end
  return a_max
end


# Main loop - Bellman update 
err = 1
  while err > ϵ 
    for s in S, n in 1:N
      u = Q_beta[s][n,U0] <= Q_beta[s][n,U1] ? U0 : U1  # Choose the best action
      res = cost(X[n], s, u)
      for m in 1:N
        res += β *P[u,s][n,m]*V_beta[s][m] #Bellman update
      end
      Q_beta[s][n,u] = res
    end
    err = frobNorm(Q_beta,Q_old)
    #println(err)
    for s in S, i in 1:N, j in 1:length(U)
      Q_old[s][i,j] = Q_beta[s][i,j]
    end
    (V_beta, g_beta) = findV(Q_beta)
  end
  
