# experiment to study tree construction
using LinearAlgebra
using HDF5
using Statistics
using BenchmarkTools
using FastKernelTransform
using FastKernelTransform: FmmMatrix, initialize_tree
using FastKernelTransform: uniform_data, gaussian_data, gaussian_mixture_data, embedded_data
using CovarianceFunctions
using TimerOutputs
to = TimerOutput()

# to save results
f = h5open("FKT_tree_experiment.h5", "w")
sizes = [15_000, 30_000, 60_000, 125_000, 250_000, 500_000, 1_000_000, 2_000_000]
dimensions = [3, 4, 5]
f["sizes"] = sizes
f["dimensions"] = dimensions

σ = .1
c = 2
gm_data(n, d) = gaussian_mixture_data(n, c, d, σ)
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
max_dofs_fun(p, d) = 2binomial(p + d, d)
f["max_dofs_per_leaf"] = "functional"

nexperiments = 1 # number of different random datasets for each size
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

tree_times = zeros(nexperiments, length(sizes), length(dimensions), length(max_dofs_per_leaf_multiplier))

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
            for exp_i in 1:nexperiments
                points = gen(n, d) # generate data set
                tgt_points = points
                src_points = points
                outgoing_length = 1

                # factor benchmark
                bench = @benchmarkable initialize_tree($tgt_points, $src_points, $mdpl, $outgoing_length)
                tree_times[exp_i, i, j, k] = minimum(run(bench, samples = 1)).time
                println("tree construction ", tree_times[exp_i, i, j, k] / nano)
                println()
            end
        end
    end
end

tree_times ./= nano
f["tree_times"] = tree_times
close(f)
