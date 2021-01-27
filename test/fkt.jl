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
dimension   = 2
# Parameter used for Gegenbauer polynomials
alpha = dimension/2 - 1
# Lookup table for transformation coefficients
scale   = 10
x  = [scale .* rand(dimension) for i in 1:N]

# Start with random data values at each point
y = rand(N)

# Convert from symbolic expression to Julia function
using FastKernelTransform: FmmMatrix, factorize
using CovarianceFunctions
using CovarianceFunctions: EQ, Exp, Cauchy, Lengthscale

# define kernels
es(r) = r == 0 ? typeof(r)(1e3) : inv(r)
es(x, y) = es(norm(x-y))

eq = Lengthscale(EQ(), 1/sqrt(2)) # with short lengthscale, not as accurate?

kernels = [ Exp(), Cauchy()]
names = [ "Exp", "Cauchy"]
if(dimension > 2)
    kernels = (es, eq, Exp(), Cauchy())
    names = ["Electro", "EQ", "Exp", "Cauchy"]
end
tol = 5e-5
@testset "factorize and mul!" begin
    fact = 0
    for (i, k) in enumerate(kernels)
        # println(names[i])
        mat = FmmMatrix(k, x, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)

        kern_mat  = k.(x, permutedims(x))
        bbar = fact * y
        b = kern_mat * y
        @test norm(b-bbar) / norm(b) < tol
    end
    @test size(fact) == (N, N)
    @test size(fact, 1) == N
    @test size(fact, 2) == N
    @test size(fact, 3) == 1
end

end # TestFastKernelTransform
