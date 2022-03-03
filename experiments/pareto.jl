# WARNING: new version deprecates this, future profiling with @profile

using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: gaussian_mixture_data, two_bump_data, uniform_data, unit_hypersphere, FactorizationParameters
using TimerOutputs
using StaticArrays
using Profile
using CovarianceFunctions
using CovarianceFunctions: Exp, EQ, MaternP, Matern, Cauchy, difference
using FastKernelTransform: FmmMatrix, factorize
using NearestNeighbors

ek(r) = 1/(1+r^2) # with short lengthscale, not as accurate?
ek(x, y) = ek(norm(difference(x, y)))
FastKernelTransform.qrable(::typeof(ek)) = false
GC.gc()

function run_pareto_test(b, x, scale, kernel, points, param)
    params = FactorizationParameters(max_dofs_per_leaf = 128, precond_param = 0,
                                    trunc_param = param, neighbor_scale=scale, lazy = true)
    mat = FmmMatrix(kernel, points, params)

    elapsed_time = @elapsed begin
        fact = factorize(mat)
        bbar = *(fact, x, verbose = true)
    end
    rel_err = norm(b-bbar)/norm(b)
    return elapsed_time, rel_err
end

function run_time_trial()

    N = 32000  # Number of points
    dimension = 2
    to = TimerOutput()
    alpha = dimension/2 - 1
    data_generator(n, d) = uniform_data(n, d)
    points = data_generator(N, dimension)
    kernel = ek
    x = rand(N)

    kern_mat = zeros(length(points), length(points))
    kern_mat .= kernel.(points, permutedims(points))
    b = kern_mat * x
    for param in 0:2:6
        println("p=", param)
        for scale in [0.25 0.35 0.45 0.55 0.65 0.75]
    # bigger neighbor scale means more compression, less accuracy
            run_pareto_test(b, x, scale, kernel, points, param)
            times=0
            errs=0
            GC.gc()
            for k in 1:3
                time, err = run_pareto_test(b, x, scale, kernel, points, param)
                times+=time
                errs+=err
                GC.gc()
            end
            println(times/3.0,",", errs/3.0)
        end
    end
end
run_time_trial()
