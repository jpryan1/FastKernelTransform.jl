module TestFKT

using LinearAlgebra
using FastKernelTransform
using TimerOutputs
using Test

# Create the timer object
to = TimerOutput()

using FastKernelTransform: FmmMatrix, factorize
using FastKernelTransform: uniform_data, gaussian_mixture_data, two_bump_data
using CovarianceFunctions
using CovarianceFunctions: EQ, Exp, Cauchy, Lengthscale, difference

σ = .25
# c = 2
# data_generator(n, d) = gaussian_mixture_data(n, c, d, σ)
data_generator(n, d) = two_bump_data(n, d, σ)

# define kernels
es(r) = r == 0 ? typeof(r)(1e3) : inv(r)
es(x, y) = es(norm(difference(x, y)))
eq = Lengthscale(EQ(), 1/sqrt(2)) # with short lengthscale, not as accurate?
ek(r) = exp(-r) # with short lengthscale, not as accurate?
ek(x, y) = ek(norm(difference(x, y)))

FastKernelTransform.qrable(::typeof(ek)) = true

atol = 1e-4
rtol = 1e-4
# test driver
function fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    for (i, k) in enumerate(kernels)
        mat = FmmMatrix(k, x,x, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        kern_mat  = k.(x, permutedims(x))
        bbar = fact * y
        b = kern_mat * y
        println("relative error = $(norm(b-bbar)/norm(b))")
        if norm(b-bbar)/norm(b) < 1e-10
            println("Warning: far-field probably not called")
        end
        @test isapprox(b, bbar, rtol = rtol, atol = atol)
    end
end

@testset "factorize and mul!" begin
    @testset "basic properties" begin
        n, d = 1024, 2
        max_dofs_per_leaf = 64 # When to stop in tree decomposition
        precond_param     = 128  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x = data_generator(n, d)
        mat = FmmMatrix(Exp(), x, x, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        @test size(fact) == (n, n)
        @test size(fact, 1) == n
        @test size(fact, 2) == n
        @test size(fact, 3) == 1
    end

    @testset "2d" begin

        n, d = 4096, 2
        max_dofs_per_leaf = 256 # When to stop in tree decomposition
        precond_param     = 512  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x = data_generator(n, d)
        y = rand(n) # Start with random data values at each point
        kernels = [ek, Cauchy()]
        names = ["Exp", "Cauchy"]
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    end

    @testset "3d" begin

        n, d = 4096, 3
        max_dofs_per_leaf = 256  # When to stop in tree decomposition
        precond_param     = 512  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x = data_generator(n, d)
        y = rand(n) # Start with random data values at each point
        kernels = (eq, ek, Cauchy()) # (es, eq, ek, Cauchy())
        names = ["EQ", "Exp", "Cauchy"] # ["Electro", "EQ", "Exp", "Cauchy"]
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, to)
    end
end

end # TestFastKernelTransform
