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

generator(n, d) = unit_hypersphere(n, d)
gen_name = "hypersphere"
trunc_param = 4
f = h5open("FKT_breakeven_$(gen_name)_p$(trunc_param).h5", "w")

# generator(n, d) = interlocking_rings(n, d)
# gen_name = "interlocking_rings"
f["generator"] = gen_name

sizes = collect(range(500, 2_000, step = 500))
dimensions = collect(3:6)
f["sizes"] = sizes
f["dimensions"] = dimensions

precond_param = 0  # Size of diag blocks to inv for preconditioner
max_dofs_per_leaf_multiplier = [2]  # When to stop in tree decomposition
max_dofs_fun(p, d) = binomial(p + d, d)
f["max_dofs_per_leaf"] = "functional"

nexperiments = 4 # number of different random datasets for each size
f["nexperiments"] = nexperiments
nsamples = 8 # number of different runs for benchmarking results
f["nsamples"] = nsamples

using CovarianceFunctions
using CovarianceFunctions: Exp, EQ, MaternP, Matern, Cauchy
kernel = Exp()

f["precond_param"] = precond_param
f["trunc_param"] = trunc_param

factor_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
fast_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
lazy_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))
dense_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))

nano = 1e9 # conversion to seconds from nano seconds

for k in eachindex(max_dofs_per_leaf_multiplier)
    for j in eachindex(dimensions)
        d = dimensions[j]
        println("dim ", d)
        mdpl = Int(round(max_dofs_per_leaf_multiplier[k]*max_dofs_fun(trunc_param, d)))
        println("max dofs ", mdpl)

        params = FactorizationParameters(max_dofs_per_leaf = mdpl, precond_param = precond_param, trunc_param = trunc_param)
        for i in eachindex(sizes)
            n = sizes[i]
            println("size ", n)
            bl = zeros(n) # result vector for lazy matrix
            bd = zeros(n) # result vector for dense matrix
            b = zeros(Complex{Float64}, n) # result vector
            y = randn(n) # "charge vector"
            for exp_i in 1:nexperiments
                println("experiment ", exp_i)
                points = generator(n, d) # generate data set

                # factor benchmark
                K = FmmMatrix(kernel, points, params, to)
                bench = @benchmarkable fkt($K)
                factor_times[exp_i, i, j, k] = minimum(run(bench, samples = nsamples)).time
                println("factor ", factor_times[exp_i, i, j, k] / nano )
                # fast multiply benchmark
                F = fkt(K)
                bench = @benchmarkable mul!($b, $F, $y, verbose = $true)
                fast_times[exp_i, i, j, k] = minimum(run(bench, samples = nsamples)).time
                println("fast ", fast_times[exp_i, i, j, k] / nano )

                # lazy multiply benchmark
                G = gramian(kernel, points)
                bench = @benchmarkable mul!($bl, $G, $y)
                lazy_times[exp_i, i, j, k] = minimum(run(bench, samples = nsamples)).time
                println("lazy ", lazy_times[exp_i, i, j, k] / nano )

                # dense multiply benchmark, includes instantiation
                bench = @benchmarkable mul!($bd, Matrix($G), $y)
                dense_times[exp_i, i, j, k] = minimum(run(bench, samples = nsamples)).time
                println("dense ", dense_times[exp_i, i, j, k] / nano )
            end
        end
    end
end

factor_times ./= nano
fast_times ./= nano
lazy_times ./= nano
dense_times ./= nano

f["factor_times"] = factor_times
f["fast_times"] = fast_times
f["lazy_times"] = lazy_times
f["dense_times"] = dense_times
close(f)
