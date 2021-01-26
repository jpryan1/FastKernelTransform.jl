# random data generators
uniform_data(n, d) = [rand(d) for _ in 1:n]
gaussian_data(n, d) = [randn(d) for _ in 1:n]

# data size n per cluster center
# dimension d
# number of centers c
# σ is std of data around centers
function gaussian_mixture_data(n::Int, c::Int, d::Int, σ::Real)
    centers = gaussian_data(c, d)
    data = zeros(d, n*c)
    for (i, μ) in enumerate(centers)
        ind_i = n*(i-1)+1:n*i
        data[:, ind_i] = @. μ + σ * $randn(d, n)
    end
    return [copy(c) for c in eachcol(data)]
end

# embeds data in d dimensional space
function embedded_data(data::AbstractVector{<:AbstractVector{<:Real}}, d::Int)
    di = length(data[1])
    E = randn(d, di) # random linear embedding
    return [E*x for x in data]
end

using LinearAlgebra
using HDF5
using Statistics
using BenchmarkTools
using FastKernelTransform
using FastKernelTransform: FmmMatrix
using CovarianceFunctions
using TimerOutputs
to = TimerOutput()

# to save results
f = h5open("FKT_synthetic_experiments.h5", "w")

sizes = @. 2048 * 2^(1:2)
dimensions = [3, 4]
f["sizes"] = sizes
f["dimensions"] = dimensions

σ = .2
c = 8
gm_data(n, d) = gaussian_mixture_data(Int(floor(n/c)), d, c, σ)
create_group(f, "mixture parameters")
g = f["mixture parameters"]
g["c"] = 8 # number of centers
g["sigma"] = σ # std of clusters

generators = (uniform_data, gaussian_data) #,  gm_data)
gen_names = ["uniform", "gaussian"] #, "mixture"]
f["generators"] = gen_names

nexperiments = 1 # number of repetitions per experiment
f["nexperiments"] = nexperiments

using CovarianceFunctions
using CovarianceFunctions: Exp, EQ, MaternP, Matern, Cauchy
kernel = Exp()

# FKT parameters # IDEA could loop through hyper-parameters
max_dofs_per_leaf = 256  # When to stop in tree decomposition
precond_param     = 512  # Size of diag blocks to inv for preconditioner
trunc_param = 5
f["max_dofs_per_leaf"] = max_dofs_per_leaf
f["precond_param"] = precond_param
f["trunc_param"] = trunc_param

factor_times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))
fast_times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))
lazy_times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))
# dense_times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))

for k in eachindex(generators)
    gen = generators[k]
    println(gen_names[k])
    for j in eachindex(dimensions)
        d = dimensions[j]
        println("dim ", d)
        for i in eachindex(sizes)
            n = sizes[i]
            bl = zeros(n) # result vector for lazy matrix
            b = zeros(Complex{Float64}, n) # result vector
            y = randn(n) # "charge vector"
            for exp_i in 1:nexperiments
                points = gen(n, d) # generate data set

                # factor benchmark
                K = FmmMatrix(kernel, points, max_dofs_per_leaf, precond_param, trunc_param, to)
                bench = @benchmarkable fkt($K)
                factor_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time

                # fast multiply benchmark
                F = fkt(K)
                bench = @benchmarkable mul!($b, $F, $y)
                fast_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time

                # lazy multiply benchmark
                G = gramian(kernel, points)
                bench = @benchmarkable mul!($bl, $G, $y)
                lazy_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time
            end
        end
    end
end

nano = 1e9 # conversion to seconds from nano seconds
factor_times ./= nano
fast_times ./= nano
lazy_times ./= nano

f["factor_times"] = factor_times
f["fast_times"] = fast_times
f["lazy_times"] = lazy_times
close(f)
