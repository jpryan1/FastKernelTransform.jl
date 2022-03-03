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

ek(r) = exp(-r) # with short lengthscale, not as accurate?
ek(x, y) = ek(norm(difference(x, y)))
FastKernelTransform.qrable(::typeof(ek)) = false

function run_time_trial(N,dimension, show)
# N = 40000  # Number of points
# dimension = 5
params = FactorizationParameters(max_dofs_per_leaf = 4*128, precond_param = 0,
                                trunc_param = 4, lazy = true)
# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1
# Lookup table for transformation coefficients
data_generator(n, d) = unit_hypersphere(n, d)
points = data_generator(N, dimension)

# define kernel

kernel = ek
to = TimerOutput()

# Start with random data values at each point
x = rand(N)

# Convert from symbolic expression to Julia function

mat = FmmMatrix(kernel, points, params, to)

doprofile = false
if doprofile
    @profile factorize(mat)
    # Profile.print(format = :flat, sortedby = :count, mincount = 100, combine = true)
    Profile.print(format = :tree, maxdepth = 20, noisefloor = 2., mincount = 10, combine = true)
    Profile.clear()
end

# Create the timer object
dotime = true
if dotime
    @timeit to "Form factorization" fact = factorize(mat)
    @timeit to "Factorization matvec " bbar = *(fact, x, verbose = true)

    compare = false
    if compare
        @timeit to "Form dense matrix direct" begin
            @timeit to "allocation" kern_mat = zeros(length(points), length(points))
            @timeit to "computation" kern_mat .= kernel.(points, permutedims(points))
        end
        @timeit to "Dense matvec "      b         = kern_mat * x
        G = gramian(kernel, points)
        @timeit to "Lazy matvec " mul!(b, G, x)
    end
    if show
        display(to)
    end
end
# println("Factorization matvec error ", norm(b-bbar)/norm(b))
# @timeit to "Fact approx solve"  xbar      = fact \ b
# @timeit to "Dense solve "       xstar     = kern_mat \ b
# println("Fact solve residual ", norm(kern_mat * xbar - b) / norm(b))
end
GC.gc()

for d in 2:2
    for Npow in 2:2
        N = 10000*(2^Npow)
        N = 1000*Npow
        println("n=",N," d=",d)
        run_time_trial(N,d, false)
        run_time_trial(N,d, true)
    end
end
