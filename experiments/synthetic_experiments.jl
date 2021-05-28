using LinearAlgebra
using HDF5
using Statistics
using BenchmarkTools
using FastKernelTransform
using FastKernelTransform: FmmMatrix
using FastKernelTransform: uniform_data, gaussian_data, gaussian_mixture_data,
        embedded_data, unit_hypersphere, interlocking_rings, FactorizationParameters
using CovarianceFunctions
using TimerOutputs
to = TimerOutput()

# to save results
# f = h5open("FKT_synthetic_experiments_N_scaling.h5", "w")
# sizes = @. 512 * 2^(1:6)
# dimensions = [3]

# generator(n, d) = unit_hypersphere(n, d)
# gen_name = "hypersphere"

generator(n, d) = interlocking_rings(n, d)
gen_name = "interlocking_rings"

trunc_param = 4
f = h5open("FKT_scaling_$(gen_name)_p$(trunc_param).h5", "w")
f["generator"] = gen_name

# sizes = collect(@. 40_000 * 2^(0:3))
sizes = collect(@. 1_000 * 2^(0:5)) # ^8 would be 250_000
dimensions = collect(3:5)
f["sizes"] = sizes
f["dimensions"] = dimensions

precond_param     = 0  # Size of diag blocks to inv for preconditioner
max_dofs_per_leaf_multiplier = [2]  # When to stop in tree decomposition
max_dofs_fun(p, d) = binomial(p + d, d)
f["max_dofs_per_leaf"] = "functional"

nexperiments = 5 # number of different random datasets for each size
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
dense_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))

nano = 1e9 # conversion to seconds from nano seconds

function compute_fast_times!(b, y, n, d, kernel, params, to)
    points = generator(n, d) # generate data set
    # factor benchmark
    K = FmmMatrix(kernel, points, params, to)
    bench = @benchmarkable fkt($K)
    factor_time = minimum(run(bench, samples = nsamples)).time
    # fast multiply benchmark
    F = fkt(K)
    bench = @benchmarkable mul!($b, $F, $y, verbose = $true)
    fast_time = minimum(run(bench, samples = nsamples)).time
    return factor_time, fast_time
end

function compute_dense_times!(b, y, n, d, kernel, params, to)
    points = generator(n, d) # generate data set
    # dense multiply benchmark, includes instantiation
    G = gramian(kernel, points)
    bench = @benchmarkable mul!($b, Matrix($G), $y)
    dense_time = minimum(run(bench, samples = nsamples)).time
    return dense_time
end

# lazy multiply benchmark
function compute_lazy_times!(b, y, n, d, kernel, params, to)
    points = generator(n, d) # generate data set
    G = gramian(kernel, points)
    bench = @benchmarkable mul!($b, $G, $y)
    lazy_time = minimum(run(bench, samples = nsamples)).time
    return lazy_time
end

max_lazy = 40_000
max_dense = 20_000

for k in eachindex(max_dofs_per_leaf_multiplier)
    for j in eachindex(dimensions)
        d = dimensions[j]
        println("dim ", d)
        mdpl = Int(round(max_dofs_per_leaf_multiplier[k]*max_dofs_fun(trunc_param, d)))
        println("max dofs ", mdpl)
        params = FactorizationParameters(max_dofs_per_leaf = mdpl, precond_param = precond_param,
                                         trunc_param = trunc_param, lazy = true)
        for i in eachindex(sizes)
            n = sizes[i]
            println("size ", n)
            b = zeros(Complex{Float64}, n) # result vector
            y = randn(n) # "charge vector"
            for exp_i in 1:nexperiments
                GC.gc()
                factor_time, fast_time = compute_fast_times!(b, y, n, d, kernel, params, to)
                factor_times[exp_i, i, j, k] = factor_time
                fast_times[exp_i, i, j, k] = fast_time
                println("factor ", factor_times[exp_i, i, j, k] / nano )
                println("fast ", fast_times[exp_i, i, j, k] / nano )

                if n < max_dense
                    GC.gc()
                    dense_time = compute_dense_times!(b, y, n, d, kernel, params, to)
                    dense_times[exp_i, i, j, k] = dense_time
                    println("dense ", dense_times[exp_i, i, j, k] / nano )
                end
                if n < max_lazy
                    lazy_time = compute_lazy_times!(b, y, n, d, kernel, params, to)
                    lazy_times[exp_i, i, j, k] = lazy_time
                    println("lazy ", lazy_times[exp_i, i, j, k] / nano )
                end
            end
        end
    end
end

factor_times ./= nano
fast_times ./= nano
dense_times ./= nano
lazy_times ./= nano

f["factor_times"] = factor_times
f["fast_times"] = fast_times
f["dense_times"] = dense_times
f["lazy_times"] = lazy_times
close(f)
