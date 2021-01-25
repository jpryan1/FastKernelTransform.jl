# module BenchmarkFastKernelTransform
# this creates a benchmark suite that can be used to keep track of performance
# TODO: add benchmarks for sub-components to optimize performance
using LinearAlgebra
using FastKernelTransform

using Plots
using SymEngine
using BenchmarkTools
using TimerOutputs

using FastKernelTransform: get_kernel_fun, FmmMatrix, factorize

# controls when the Woodbury representation is more efficient than dense
function FastKernelTransform_benchmark(N = 8000, max_dofs_per_leaf = 512, precond_param = 1024,
                            trunc_param = 10, dimension = 3)
    # Parameter used for Gegenbauer polynomials
    alpha = dimension/2 - 1
    # Lookup table for transformation coefficients
    transform_coef_table = -ones(trunc_param+1, trunc_param+1, trunc_param+1)
    scale   = 10
    points  = [scale .* rand(dimension) for i in 1:N]

    # define kernel
    k(r) = exp(-r)
    k(x, y) = k(norm(x-y))

    @vars r
    kern    = k(r)
    kernel_fun = get_kernel_fun(kern)

    # Start with random data values at each point
    x = rand(N)
    to = TimerOutput()

    suite = BenchmarkGroup()

    mat = FmmMatrix(k, points, max_dofs_per_leaf, precond_param, trunc_param, to)
    suite["factor"] = @benchmarkable factorize($mat)
    fact = factorize(mat)
    # suite["form dense matrix"] = @benchmarkable $kernel_fun.($points, permutedims($points))
    # kern_mat  = kernel_fun.(points, permutedims(points))

    suite["fast matvec"] = @benchmarkable $fact * $x
    # suite["dense matvec"] = @benchmarkable $kern_mat * $x
    # @timeit to "Fact approx solve"  xbar      = fact \ b
    # @timeit to "Dense solve "       xstar     = kern_mat \ b
    # println("Fact solve residual ", norm(kern_mat * xbar - b) / norm(b))
    return suite
end

N = 16000
max_dofs_per_leaf = 512
precond_param = 2048
trunc_param = 5
dimension = 3

suite = FastKernelTransform_benchmark(N, max_dofs_per_leaf, precond_param, trunc_param, dimension)
results = run(suite)
display(results)

path = "FastKernelTransform/test/"
file = "benchmark_results.json"
save = false # do this only once before making changes, then compare against results
if save
    BenchmarkTools.save(path * file, results)
end

compare = true
if compare
    old_results = BenchmarkTools.load(path * file)
    old_results = old_results[1]
    metric = minimum
    judge(metric(results), metric(old_results))
end

# end # BenchmarkFastKernelTransform
