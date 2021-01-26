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

# embeds
function embedded_data(data::AbstractVector{<:AbstractVector}, d::Int)
    return -1 # TODO
end

using LinearAlgebra
using HDF5
using Statistics
using BenchmarkTools
using FastKernelTransform
using FastKernelTransform: FmmMatrix
using TimerOutputs
to = TimerOutput()

# to save results
f = h5open("FKT_synthetic_experiments.h5", "w")

sizes = @. 2048 * 2^(1:2)
dimensions = [3, 4, 5]
f["sizes"] = sizes
f["dimensions"] = dimensions

σ = .2
c = 8
gm_data(n, d) = gaussian_mixture_data(Int(floor(n/c)), d, c, σ)
create_group(f, "mixture parameters")
g = f["mixture parameters"]
g["c"] = 8 # number of centers
g["sigma"] = σ # std of clusters

generators = (uniform_data, gaussian_data,  gm_data)
f["generators"] = ["uniform", "gaussian", "mixture"]

nexperiments = 1 # number of repetitions per experiment
f["nexperiments"] = nexperiments

kernel(r) = exp(-r) # IDEA could loop through kernels
kernel(x, y) = kernel(norm(x-y))

# FKT parameters # IDEA could loop through hyper-parameters
max_dofs_per_leaf = 128  # When to stop in tree decomposition
precond_param     = 256  # Size of diag blocks to inv for preconditioner
trunc_param = 5
f["max_dofs_per_leaf"] = max_dofs_per_leaf
f["precond_param"] = precond_param
f["trunc_param"] = trunc_param

# TODO: add dense and lazy multiply

times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))
for k in eachindex(generators)
    gen = generators[k]
    for j in eachindex(dimensions)
        d = dimensions[j]
        for i in eachindex(sizes)
            for exp_i in 1:nexperiments
                n = sizes[i]
                x = gen(n, d) # generate data set
                K = FmmMatrix(kernel, x, max_dofs_per_leaf, precond_param, trunc_param, to)
                times[exp_i, i, j, k] = minimum(@benchmark($fkt($K))).time
            end
        end
    end
end

f["times"] = times
close(f)
