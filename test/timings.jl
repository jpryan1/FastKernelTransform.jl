# WARNING: new version deprecates this, future profiling with @profile 

using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: gaussian_mixture_data, two_bump_data, uniform_data, unit_hypersphere
using TimerOutputs
using StaticArrays

# Create the timer object
to = TimerOutput()

N                 = 16000  # Number of points
max_dofs_per_leaf = 140  # When to stop in tree decomposition
precond_param     = 0  # Size of diag blocks to inv for preconditioner

trunc_param = 4
dimension   = 5

# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1

# Lookup table for transformation coefficients
σ = .25
c = 10
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
# kernel = Exp()

ek(r) = exp(-r) # with short lengthscale, not as accurate?
ek(x, y) = ek(norm(difference(x, y)))
FastKernelTransform.qrable(::typeof(ek)) = false
kernel = ek

# Start with random data values at each point
x = rand(N)

# Convert from symbolic expression to Julia function
using FastKernelTransform: FmmMatrix, factorize

mat = FmmMatrix(kernel, points, max_dofs_per_leaf, precond_param, trunc_param, to)
@timeit to "Form factorization" begin
    @profile fact = factorize(mat)
end

# @timeit to "Factorization matvec "       bbar      = *(fact, x, verbose = true)
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

# println("Factorization matvec error ", norm(b-bbar)/norm(b))
# @timeit to "Fact approx solve"  xbar      = fact \ b
# @timeit to "Dense solve "       xstar     = kern_mat \ b
# println("Fact solve residual ", norm(kern_mat * xbar - b) / norm(b))
display(to)
