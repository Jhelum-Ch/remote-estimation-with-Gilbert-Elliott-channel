# ---
# name: algorihms.jl
# author: Jhelum Chakravorty, Aditya Mahajan
# date: 16 May, 2018
# license: BSD 3 clause
# ---

# Introduction
# ============
#
# Source Model
# ------------
#
# The source is a first-order auto-regressive process $\{E_t\}_{t \ge 0}$
# which evolves as follows:
# $$ E_{t+1} = a E_t + W_t, $$
# where $a$ is a real number and $\{W_t\}_{t \ge 0}$ is a i.i.d.\ process with
# symmetric and unimodal distribution. In our simulations, we restrict
# attention to $a = 1$ and $W_t$ distributed according to zero-mean Gaussian
# distribution with unit variance. These values are hard-coded in the
# `nextState` function defined below. 

@inline function nextState(E)
  a = 1.0
  σ = 1.0
  a*E + σ*randn()
end

# Channel Model
# ------------
#
# The channel is binary state Markov channel (also known as a Gilbert-Elliot
# channel. For conceptual simplicity, we label the two states as `0` and `1`.

const S0 = 1
const S1 = 2

struct Channel
  Q :: Matrix{Float64}  # Channel transition matrix
  p :: Vector{Float64}  # drop probability
end

# The `nextChannel` function generates a random next channel state according
# to the channel transition matrix. 

@inline function nextChannel(S, Q)
  S0 + Int(rand() <= Q[S, S1]) 
end

# The `successful` function generates a binary valued random variable
# indicating whether or not the transmission is successful when the channel
# state is `S`. 

@inline function successful(S, p)
  rand() > p[S]
end

# We assume that the system starts from $E^+_{-1} = 0$ and $S_{-1} = 0$. 

const E_ini = 0.0
const S_ini = S0

# Distortion
# ----------
# Whenever $|E_t| < k(S_t)$, where $k$ is a threshold vector that can be
# tuned, we incur a distortion $d(E_t)$. We hard-code $d(e) = e^2$ in the
# `distortion` function defined below. 

@inline distortion(E) = E^2

# Sampling functions
# ==================
# 
# Let $τ$ denote the stopping time when $\{(E^+_{t-1}, S^{t-1}) = (E_{ini},
# S_{ini})$. The code below performs the following sample-path calculations. 
# $$
# L = \sum_{t = 0}^{τ - 1} β^t (λ(U_t) + d(E^+_t), \quad
# M = \sum_{t = 0}^{τ -1} β^t.         
# $$
# where $β$ is the discount factor and $U_t = \IND\{|E_t| \ge k(S_t)\}$.
#
# Since the calculation is stochastic, there is a positive probability that
# $τ$ is a big number, which can slow down the calculations. So, we set a
# bound `maxIterations` on the maximum size of $τ$. We set the default value
# of `maxIterations` as `10_000`. This number may need to be increased when
# computing the performance for large `threshold`. 

@inline function sample(channel, threshold, cost, discount; maxIterations = 10_000)

    (E_post, S) = (E_ini, S_ini)

    L, M = 0.0, 0.0

    counter = 0
    scale   = 1.0
    while counter <= maxIterations
        E_pre = nextState(E_post)
        transmit  = !(-threshold[S] < E_pre < threshold[S])
        S         = nextChannel(S, channel.Q)
        received  = transmit && successful(S, channel.p) 

        E_post = received ? 0.0 : E_pre

        # println((counter, E_pre, transmit, S, received, E_post))

        L += scale * (distortion(E_post) + cost*transmit)
        M += scale

        if (E_post, S) == (E_ini, S_ini)
          break
        else
          scale *= discount
          counter += 1
        end
    end
    (L, M)
end

# Mini-batch averaging
# --------------------
# 
# The value obtained by one sample path is usually noisy. So, we smoothen it
# out by averaging over a mini-batch. The default size of the mini-batch is
# `100` iterations.

@inline function sample_average(channel, threshold, cost, discount; iterations::Int=100)
    ell, emm = 0.0, 0.0

    for i in 1:iterations
        L, M = sample(channel, threshold, cost, discount)
        ell += L
        emm += M
    end
    ell /= iterations
    emm /= iterations
    (ell, emm)
end

# Stochastic approximation
# ========================
# 
# The function `sa_costly` computes the optimal threshold for given values
# of `cost`, `discount`. It uses Kiefer Wolfowitz algorithm.
# In particular, the gradient is calculated using finite differences:
# $$ \nabla L \approx \frac{1}{2c} [ L(k + c) - L(k - c) ]. $$
# By default, `c` is set to `0.1`. If we were writing this code for higher
# dimensions, we would replace Kiefer-Wolfowitz with the simultaneous
# perturbation (SPSA) algorithm, which is more sample efficient for higher
# dimensions. 
#
# The stochastic approximation iteration starts from an initial guess (the
# parameter `initial`). Its default value is `[1.0,1.0]`. This initialization
# could be useful if we have a reasonable guess for optimal threshold (e.g.,
# the exact solution obtained by Fredholm integral equations for the case when
# `dropProb = 0`; see TAC 2017 paper for details).
#
# It is not possible to detect convergence of stochastic approximation
# algorithms. So we run the algorithm for a fixed number of iterations (the
# parameter `iterations`, whose default value is `1_000`).
#
# Stochastic approximation algorithms are sensitive to the choice of learning
# rates. We use ADAM to adapt the learning rates according to the sample
# path. The parameters `decay1`, `decay2`, `alpha`, and `epsilon` can be used
# to tune ADAM. In our experience, these should be left to their default
# values. 
#
# It is not possible to detect convergence of stochastic approximation
# algorithms. So we run the algorithm for a fixed number of iterations (the
# parameter `iterations`, whose default value is `1_000`).
#
# Sometimes it is useful to visualize the estimates (of the threshold) as the
# algorithm is running. To do so, set `debug` to `true`, which will print the
# current estimate of the threshold after every 100 iterations.
#
# The output of the function is a trace of the estimates of the
# threshold (therefore, it is a 2D array fo size `iterations`).

function one_sided(channel, threshold, δ, c, cost, discount)
    threshold_offset = max.(threshold + δ .* c, c)
    L, M = sample_average(channel, threshold, cost, discount)
    L_offset, M_offset = sample_average(channel, threshold_offset, cost, discount)

    δ * ( L_offset * M - L * M_offset ) / c
end

function two_sided(channel,threshold, δ, c, cost, discount)
    threshold_plus  = max.(threshold + δ .* c, c)
    threshold_minus = max.(threshold - δ .* c, c)
    L, M = sample_average(channel, threshold, cost, discount)
    L_plus, M_plus = sample_average(channel, threshold_plus, cost, discount)
    L_minus, M_minus = sample_average(channel, threshold_minus, cost, discount)

    L_delta = L_plus - L_minus
    M_delta = M_plus - M_minus

    δ * ( L_delta * M - L * M_delta ) / c
end

using Printf

@fastmath function sa_costly(channel, cost, discount;
                gradient  = :OneSided,
    iterations :: Int     = 1_000,
        initial :: Array{Float64,1} = [1.0, 1.0],
        decay1 :: Float64 = 0.9,
	    decay2 :: Float64 = 0.999,
	   epsilon :: Float64 = 1e-8,
        alpha  :: Float64 = 0.01,
             c :: Float64 = 0.1,
	     debug :: Bool    = false,
    )

    threshold = copy(initial)

    trace = zeros(iterations,2)

    moment1 = zeros(2)
    moment2 = zeros(2)

    weight1 = decay1
    weight2 = decay2

    counter = 1

    @inbounds for k in 1:iterations
        # For SFSA
        δ = randn(2)

        # For SPSA
        # Generate Rademacher random variables
        # δ = 2*round.(Int, rand(2) .<= 0.5) - 1


        if gradient == :OneSided
            gradientValue = one_sided(channel,threshold, δ, c, cost, discount)
        else
            gradientValue = two_sided(channel,threshold, δ, c, cost, discount)
        end

        moment1 = decay1 * moment1 + (1 - decay1) * gradientValue
        moment2 = decay2 * moment2 + (1 - decay2) * gradientValue.^2

        corrected1 = moment1/(1 - weight1)
        corrected2 = moment2/(1 - weight2)

        weight1 *= decay1
        weight2 *= decay2

        threshold_delta = corrected1 ./ ( sqrt.(corrected2) .+ epsilon)

        threshold .-= alpha .* threshold_delta 

        threshold  = max.(threshold, c)

        if debug && mod(k,100) == 0
          @printf("#:%8d, threshold[S0] =%0.6f, threshold[S1] = %0.6f\n", 
                  k, threshold[S0], threshold[S1])
        end

        trace[k,:] = threshold
    end

    return trace
end


