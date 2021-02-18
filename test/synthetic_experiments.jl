using LinearAlgebra
using HDF5
using Statistics
using BenchmarkTools
using FastKernelTransform
using FastKernelTransform: FmmMatrix
using FastKernelTransform: uniform_data, gaussian_data, gaussian_mixture_data, embedded_data, unit_hypersphere
using CovarianceFunctions
using TimerOutputs
to = TimerOutput()

# to save results
# f = h5open("FKT_synthetic_experiments_N_scaling.h5", "w")
# sizes = @. 512 * 2^(1:6)
# dimensions = [3]

f = h5open("FKT_synthetic_experiments_scaling.h5", "w")
sizes = [25000, 50000, 100000]
dimensions = [3, 5, 7]
f["sizes"] = sizes
f["dimensions"] = dimensions

σ = .2
c = 10
gm_data(n, d) = unit_hypersphere(n,d)
create_group(f, "mixture parameters")
g = f["mixture parameters"]
g["c"] = c # number of centers
g["sigma"] = σ # std of clusters

generators = ( gm_data)
gen_names = [ "mixture"]
f["generators"] = gen_names

precond_param     = 0  # Size of diag blocks to inv for preconditioner
trunc_param = 4
max_dofs_per_leaf_multiplier = [2]  # When to stop in tree decomposition
max_dofs_fun(p, d) = 128 #2binomial(p + d, d)
f["max_dofs_per_leaf"] = "functional"

nexperiments = 3 # number of different random datasets for each size
f["nexperiments"] = nexperiments
nsamples = 1 # number of different runs for benchmarking results
f["nsamples"] = nsamples

using CovarianceFunctions
using CovarianceFunctions: Exp, EQ, MaternP, Matern, Cauchy
# kernels = [Exp(), EQ()]
kernel = Exp()

# FKT parameters # IDEA could loop through hyper-parameters

f["precond_param"] = precond_param
f["trunc_param"] = trunc_param

factor_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
fast_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
lazy_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
# dense_times = zeros(nexperiments, length(sizes), length(dimensions), length(generators))

gen = gm_data
nano = 1e9 # conversion to seconds from nano seconds

for k in eachindex(max_dofs_per_leaf_multiplier)
    for j in eachindex(dimensions)
        d = dimensions[j]
        println("dim ", d)
        mdpl = max_dofs_per_leaf_multiplier[k]*max_dofs_fun(trunc_param, d)
        println("max dofs ", mdpl)
        for i in eachindex(sizes)
            n = sizes[i]
            println("size ", n)
            bl = zeros(n) # result vector for lazy matrix
            b = zeros(Complex{Float64}, n) # result vector
            y = randn(n) # "charge vector"
            for exp_i in 1:nexperiments
                points = gen(n, d) # generate data set

                # factor benchmark
                K = FmmMatrix(kernel, points, mdpl, precond_param, trunc_param, to)
                bench = @benchmarkable fkt($K)
                factor_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time
                println("factor ", factor_times[exp_i, i, j, k] / nano )
                # fast multiply benchmark
                # F = fkt(K)
                # bench = @benchmarkable mul!($b, $F, $y, verbose = $true)
                # fast_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time
                # println("fast ", fast_times[exp_i, i, j, k] / nano )
                #
                # if n ≤ 2^15
                #     # lazy multiply benchmark
                #     G = gramian(kernel, points)
                #     bench = @benchmarkable mul!($bl, $G, $y)
                #     lazy_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time
                #     println("lazy ", fast_times[exp_i, i, j, k] / nano )
                # end
            end
        end
    end
end

factor_times ./= nano
fast_times ./= nano
lazy_times ./= nano

f["factor_times"] = factor_times
f["fast_times"] = fast_times
f["lazy_times"] = lazy_times
close(f)
