module TestFKT

using LinearAlgebra
using FastKernelTransform
using TimerOutputs
using Test

# Create the timer object
to = TimerOutput()
N                 = 4000  # Number of points
max_dofs_per_leaf = 128  # When to stop in tree decomposition
precond_param     = 256  # Size of diag blocks to inv for preconditioner

trunc_param = 10
dimension   = 3
# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1
# Lookup table for transformation coefficients
scale   = 10
x  = [scale .* rand(dimension) for i in 1:N]

# Start with random data values at each point
y = rand(N)

# Convert from symbolic expression to Julia function
using FastKernelTransform: FmmMatrix, factorize

# define kernels
se(r) = exp(-r^2)
se(x, y) = se(norm(x-y))
ek(r) = exp(-r) # exponential kernel
ek(x, y) = ek(norm(x-y))
es(r) = r == 0 ? typeof(r)(1e3) : inv(r)
es(x, y) = es(norm(x-y))

cauchy(r) = inv(1+r^2)
cauchy(x, y) = cauchy(norm(x-y))

kernels = (es, se, ek, cauchy)
names = ["Electro", "SE", "Exp", "Cauchy"]
tol = 5e-5
@testset "MultipoleFactorization" begin
    for k in kernels
        mat = FmmMatrix(k, x, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        kern_mat  = k.(x, permutedims(x))
        bbar = fact * y
        b = kern_mat * y
        @test norm(b-bbar)/norm(b) < tol
    end
end

end # TestFastKernelTransform
