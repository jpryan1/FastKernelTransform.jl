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
ek(r) = exp(-10r)
ek(x, y) = ek(norm(difference(x, y)))

FastKernelTransform.qrable(::typeof(ek)) = true
FastKernelTransform.get_correction(::typeof(ek)) = ek

atol = 1e-3
rtol = 1e-3
verbose = true
# test driver
function fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param,
                  to, names = nothing; verbose::Bool = true)
    for (i, k) in enumerate(kernels)
        verbose && println(names[i])
        mat = FmmMatrix(k, x, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        kern_mat  = k.(x, permutedims(x))
        bbar = *(fact, y, verbose = verbose)
        b = kern_mat * y
        verbose && println("relative error = $(norm(b-bbar)/norm(b))")
        @test isapprox(b, bbar, rtol = rtol, atol = atol)
    end
end

@testset "factorize and mul!" begin
    @testset "basic properties" begin
        n, d = 2048, 2
        max_dofs_per_leaf = 256 # When to stop in tree decomposition
        precond_param     = 512  # Size of diag blocks to inv for preconditioner
        trunc_param = 7
        x = data_generator(n, d)
        variance = exp.(randn(n)) # additive diagonal
        k = Exp()
        mat = FmmMatrix(k, x, max_dofs_per_leaf, precond_param, trunc_param, to, variance)
        fact = factorize(mat)
        @test size(fact) == (n, n)
        @test size(fact, 1) == n
        @test size(fact, 2) == n
        @test size(fact, 3) == 1
        y = randn(n)
        b = *(fact, y, verbose = true)
        @test eltype(b) <: Real
        # checking diagonal matrix was incorporated
        K = gramian(k, x)
        c = K * y
        @. c += variance * y
        @test isapprox(b, c, rtol = rtol)
        # indexing
        for _ in 1:16
            i, j = rand(1:n, 2)
            println(i, j)
            println(fact[i, j])
            println(K[i, j])
            @test fact[i, j] ≈ K[i, j]
        end
    end

    @testset "2d" begin
        n, d = 4096, 2
        max_dofs_per_leaf = 512 # When to stop in tree decomposition
        precond_param     = 1024  # Size of diag blocks to inv for preconditioner
        trunc_param = 7
        x = data_generator(n, d)
        y = rand(n) # Start with random data values at each point
        kernels = (ek, Cauchy())
        names = ("Exp", "Cauchy")
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, to, names)
        # matrix-matrix multiply
        m = 2
        Y = rand(n, m) # Start with random data values at each point
        fkt_test(kernels, x, Y, max_dofs_per_leaf, precond_param, trunc_param, to, names)
    end

    @testset "3d" begin
        n, d = 4096, 3
        max_dofs_per_leaf = 512  # When to stop in tree decomposition
        precond_param     = 1024  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x = data_generator(n, d)
        y = rand(n) # Start with random data values at each point
        kernels = (eq, ek, Cauchy()) # (es, eq, ek, Cauchy())
        names = ("EQ", "Exp", "Cauchy") # ["Electro", "EQ", "Exp", "Cauchy"]
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, to, names)

        # matrix-matrix multiply
        m = 2
        Y = rand(n, m) # Start with random data values at each point
        fkt_test(kernels, x, Y, max_dofs_per_leaf, precond_param, trunc_param, to, names)
    end

    @testset "rectangle" begin
        n, d = 4096, 3
        max_dofs_per_leaf = 512  # When to stop in tree decomposition
        precond_param     = 1024  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        x1 = data_generator(n, d)
        x2 = data_generator(n+4, d)
        y = rand(n+4) # Start with random data values at each point
        kernels = (eq, ek, Cauchy()) # (es, eq, ek, Cauchy())
        names = ("EQ", "Exp", "Cauchy") # ["Electro", "EQ", "Exp", "Cauchy"]
        mat = FmmMatrix(ek, x1, x2, max_dofs_per_leaf, precond_param, trunc_param, to)
        fact = factorize(mat)
        kern_mat  = ek.(x1, permutedims(x2))
        bbar = *(fact, y, verbose = verbose)
        b = kern_mat * y
        verbose && println("relative error = $(norm(b-bbar)/norm(b))")
        @test isapprox(b, bbar, rtol = rtol, atol = atol)
    end
end

using FastKernelTransform: conj_grad
@testset "linear solves" begin
    n, d = 8096, 3
    max_dofs_per_leaf = 512  # When to stop in tree decomposition
    precond_param     = 1024  # Size of diag blocks to inv for preconditioner
    trunc_param = 5
    x = data_generator(n, d)
    y = rand(n) # Start with random data values at each point
    k = ek
    names = "Exp"
    variance = fill(1e-4, n)
    mat = FmmMatrix(k, x, max_dofs_per_leaf, precond_param, trunc_param, to, variance)
    fact = factorize(mat)
    b = fact * y
    rtol = 1e-3
    y_cg = conj_grad(fact, b, tol = rtol, max_iter = 256)
    @test isapprox(y, y_cg, rtol = rtol) # error
    b_cg = fact * y_cg
    # println(norm(y_cg-y)/norm(y))
    # println(norm(b_cg-b)/norm(b))
    @test isapprox(b, b_cg, rtol = rtol)
end

end # TestFastKernelTransform
