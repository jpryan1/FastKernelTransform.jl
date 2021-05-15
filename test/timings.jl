# WARNING: new version deprecates this, future profiling with @profile

using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: gaussian_mixture_data, two_bump_data, uniform_data, unit_hypersphere, FactorizationParameters
using TimerOutputs
using StaticArrays
using Profile


N = 64000  # Number of points
dimension = 5

params = FactorizationParameters(max_dofs_per_leaf = 128, precond_param = 0,
                                trunc_param = 4, lazy = true)

# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1

# Lookup table for transformation coefficients
data_generator(n, d) = unit_hypersphere(n, d)
points = data_generator(N, dimension)
# S, T = length(points[1]), eltype(points[1])
# points = [SVector{S, T}(p) for p in points]
# A = rand(dimension, dimension)
# Q, R = qr(A)
# points = [Q*(vcat(point, zeros(dimension-3))) for point in points]

# define kernel
using CovarianceFunctions
using CovarianceFunctions: Exp, EQ, MaternP, Matern, Cauchy, difference

ek(r) = exp(-r) # with short lengthscale, not as accurate?
ek(x, y) = ek(norm(difference(x, y)))
FastKernelTransform.qrable(::typeof(ek)) = false
kernel = ek

# Start with random data values at each point
x = rand(N)

# Convert from symbolic expression to Julia function
using FastKernelTransform: FmmMatrix, factorize

mat = FmmMatrix(kernel, points, params)

doprofile = true
if doprofile
    @profile factorize(mat)
    # Profile.print(format = :flat, sortedby = :count, mincount = 100, combine = true)
    Profile.print(format = :tree, maxdepth = 5, noisefloor = 2., mincount = 10, combine = true)
    Profile.clear()
end

# Create the timer object
dotime = false
if dotime
    to = TimerOutput()
    @timeit to "Form factorization" fact = factorize(mat)
    # @timeit to "Factorization matvec " bbar = *(fact, x, verbose = true)

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
    display(to)
end
# println("Factorization matvec error ", norm(b-bbar)/norm(b))
# @timeit to "Fact approx solve"  xbar      = fact \ b
# @timeit to "Dense solve "       xstar     = kern_mat \ b
# println("Fact solve residual ", norm(kern_mat * xbar - b) / norm(b))
