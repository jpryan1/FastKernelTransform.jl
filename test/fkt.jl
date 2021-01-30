module TestFKT

using LinearAlgebra
using FastKernelTransform
using TimerOutputs
using Test

# Create the timer object
to = TimerOutput()
trunc_param = 10

scale = 10 # scale for random point distribution

using FastKernelTransform: FmmMatrix, factorize
using CovarianceFunctions
using CovarianceFunctions: EQ, Exp, Cauchy, Lengthscale, difference

# define kernels
es(r) = r == 0 ? typeof(r)(1e3) : inv(r)
es(x, y) = es(norm(difference(x, y)))
eq = Lengthscale(EQ(), 1/sqrt(2)) # with short lengthscale, not as accurate?

atol = 1e-4
rtol = 1e-4
# test driver
function fkt_test(kernels, x1, x2, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    for (i, k) in enumerate(kernels)
        mat = FmmMatrix(k, x1, x2, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        kern_mat  = k.(x1, permutedims(x2))
        bbar = fact * y
        b = kern_mat * y
        @test isapprox(b, bbar, rtol = rtol, atol = atol)
    end
end

@testset "factorize and mul!" begin
    @testset "basic properties" begin
        n, d = 128, 2
        max_dofs_per_leaf = 4 # When to stop in tree decomposition
        precond_param     = 8  # Size of diag blocks to inv for preconditioner
        x1  = [scale .* rand(d) for i in 1:n]
        x2  = [scale .* rand(d) for i in 1:n]
        mat = FmmMatrix(Exp(), x1,x2, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        @test size(fact) == (n, n)
        @test size(fact, 1) == n
        @test size(fact, 2) == n
        @test size(fact, 3) == 1
    end

    @testset "2d" begin
        n, d = 1024, 2
        max_dofs_per_leaf = 128 # When to stop in tree decomposition
        precond_param     = 256  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x1  = [scale .* rand(d) for i in 1:n]
        x2  = [scale .* rand(d) for i in 1:n]
        y = rand(n) # Start with random data values at each point
        kernels = [Exp(), Cauchy()]
        names = ["Exp", "Cauchy"]
        fkt_test(kernels, x1, x2, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    end

    @testset "3d" begin
        n, d = 1024, 3
        max_dofs_per_leaf = 128  # When to stop in tree decomposition
        precond_param     = 256  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x1  = [scale .* rand(d) for i in 1:n]
        x2  = [scale .* rand(d) for i in 1:n]
        y = rand(n) # Start with random data values at each point
        kernels = (eq, Exp(), Cauchy()) # (es, eq, Exp(), Cauchy())
        names = ["EQ", "Exp", "Cauchy"] # ["Electro", "EQ", "Exp", "Cauchy"]
        fkt_test(kernels, x1, x2, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    end
end

end # TestFastKernelTransform
