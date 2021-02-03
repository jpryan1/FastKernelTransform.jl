# this is the original FastKernelTransform file, separated it out from the test suite for rapid testing
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: gaussian_mixture_data, two_bump_data, uniform_data
using TimerOutputs

# Create the timer object
to = TimerOutput()

N                 = 2*8192  # Number of points
max_dofs_per_leaf = 256  # When to stop in tree decomposition
precond_param     = max_dofs_per_leaf  # Size of diag blocks to inv for preconditioner

trunc_param = 5
dimension   = 3

# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1

# Lookup table for transformation coefficients
# σ = .25
# c = 8
# data_generator(n, d) = gaussian_mixture_data(n, c, d, σ)
# points = data_generator(N, dimension)
# points = uniform_data(N, dimension)
points = two_bump_data(N, dimension)

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
    fact = factorize(mat)
end

@timeit to "Factorization matvec "       bbar      = fact * x
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
