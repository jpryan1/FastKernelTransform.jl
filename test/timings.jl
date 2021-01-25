# this is the original FastKernelTransform file, separated it out from the test suite for rapid testing
using LinearAlgebra
using FastKernelTransform
using TimerOutputs

# Create the timer object
to = TimerOutput()

N                 = 8000  # Number of points
max_dofs_per_leaf = 256  # When to stop in tree decomposition
precond_param     = 512  # Size of diag blocks to inv for preconditioner

trunc_param = 10
dimension   = 3
# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1
# Lookup table for transformation coefficients
scale   = 10
points  = [scale .* rand(dimension) for i in 1:N]

# define kernel
kernel(r) = exp(-r)
kernel(x, y) = kernel(norm(x-y))

# Start with random data values at each point
x = rand(N)

# Convert from symbolic expression to Julia function
using FastKernelTransform: FmmMatrix, factorize

mat = FmmMatrix(kernel, points, max_dofs_per_leaf, precond_param, trunc_param, to)
@timeit to "Form factorization" begin
    fact = factorize(mat)
end
# on my machine, has 2x overhead compared to using k
# using FastKernelTransform: get_kernel_fun
# @vars r
# kern = k(r)
# kernel_fun = get_kernel_fun(kern)
# @timeit to "Form dense matrix symbolic" kern_mat  = kernel_fun.(points, permutedims(points))
@timeit to "Form dense matrix direct" kern_mat  = kernel.(points, permutedims(points))
@timeit to "Factorization matvec "       bbar      = fact * x
@timeit to "Dense matvec "      b         = kern_mat * x
# println("Factorization matvec error ", norm(b-bbar)/norm(b))
# @timeit to "Fact approx solve"  xbar      = fact \ b
# @timeit to "Dense solve "       xstar     = kern_mat \ b
# println("Fact solve residual ", norm(kern_mat * xbar - b) / norm(b))
display(to)
