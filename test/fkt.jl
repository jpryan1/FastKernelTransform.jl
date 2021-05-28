module TestFKT

using LinearAlgebra
using FastKernelTransform
using TimerOutputs
using Test

using FastKernelTransform: FmmMatrix, factorize, FactorizationParameters
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
verbose = false
lazy = true # TODO: test both lazy and dense versions

# test driver
function fkt_test(kernels, x, y::AbstractVecOrMat, max_dofs_per_leaf::Int,
    precond_param::Int, trunc_param::Int, names = kernels; verbose::Bool = true)
    x_tgt, x_src = x, x
    fkt_test(kernels, x_tgt, x_src, y, max_dofs_per_leaf, precond_param, trunc_param, names, verbose = verbose)
end
function fkt_test(kernels, x_tgt, x_src, y::AbstractVecOrMat, max_dofs_per_leaf::Int,
    precond_param::Int, trunc_param::Int, names = kernels; verbose::Bool = true)
    for (i, k) in enumerate(kernels)
        verbose && println(names[i])
        params = FactorizationParameters(max_dofs_per_leaf = max_dofs_per_leaf,
                precond_param = precond_param, trunc_param = trunc_param, lazy = lazy, verbose = verbose)
        mat = FmmMatrix(k, x_tgt, x_src, params)
        fact = factorize(mat)
        kern_mat  = k.(x_tgt, permutedims(x_src))
        bbar = *(fact, y, verbose = verbose)
        b = kern_mat * y
        verbose && println("relative error = $(norm(b-bbar)/norm(b))")
        @test isapprox(b, bbar, rtol = rtol, atol = atol)
        # testing 5-arg mul!
        α, β = randn(2)
        n = length(x_tgt)
        b = y isa AbstractVector ? randn(n) : randn(n, size(y, 2))
        c = α * (kern_mat * y) + β * b
        mul!(b, fact, y, α, β)
        @test isapprox(b, c, rtol = rtol, atol = atol)
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
        params = FactorizationParameters(max_dofs_per_leaf = max_dofs_per_leaf,
                precond_param = precond_param, trunc_param = trunc_param, lazy = lazy, verbose = verbose)
        mat = FmmMatrix(k, x, variance, params)

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
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, names)
        # matrix-matrix multiply
        m = 2
        Y = rand(n, m) # Start with random data values at each point
        fkt_test(kernels, x, Y, max_dofs_per_leaf, precond_param, trunc_param, names)
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
        fkt_test(kernels, x, y, max_dofs_per_leaf, precond_param, trunc_param, names)
        # matrix-matrix multiply
        m = 2
        Y = rand(n, m) # Start with random data values at each point
        fkt_test(kernels, x, Y, max_dofs_per_leaf, precond_param, trunc_param, names)
    end

    @testset "rectangular matrix" begin
        n, d = 4096, 3
        max_dofs_per_leaf = 512  # When to stop in tree decomposition
        precond_param     = 1024  # Size of diag blocks to inv for preconditioner
        trunc_param = 5
        m = 128
        x_tgt = data_generator(n, d)
        x_src = data_generator(m, d)
        y = rand(m) # Start with random data values at each point
        kernels = (eq, ek, Cauchy()) # (es, eq, ek, Cauchy())
        names = ("EQ", "Exp", "Cauchy") # ["Electro", "EQ", "Exp", "Cauchy"]
        fkt_test(kernels, x_tgt, x_src, y, max_dofs_per_leaf, precond_param, trunc_param, names)

        # matrix-matrix multiply
        k = 2
        Y = rand(m, k) # Start with random data values at each point
        fkt_test(kernels, x_tgt, x_src, Y, max_dofs_per_leaf, precond_param, trunc_param, names)
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
    params = FactorizationParameters(max_dofs_per_leaf = max_dofs_per_leaf,
            precond_param = precond_param, trunc_param = trunc_param, lazy = lazy, verbose = verbose)
    mat = FmmMatrix(k, x, variance, params)
    fact = factorize(mat)
    b = fact * y
    M =  k.(x, permutedims(x)) + Diagonal(variance)
    @test isapprox(b, M*y, atol = atol, rtol = rtol)
    y_cg = conj_grad(fact, b, tol = rtol, max_iter = 256)
    @test isapprox(y, y_cg, rtol = rtol) # error
    b_cg = fact * y_cg
    @test isapprox(b, b_cg, rtol = rtol)
end

@testset "gramian" begin # testing factorization of gramian type with fkt
    n, d = 8096, 3
    x = data_generator(n, d)
    y = rand(n) # Start with random data values at each point
    k = ek
    G = gramian(k, x)

    params = FactorizationParameters(neighbor_scale = 1/2 + 1e-5randn(), lazy = true, verbose = false)
    F = fkt(G, params)
    @test F isa MultipoleFactorization
    @test F.params.neighbor_scale == params.neighbor_scale
    @test F.params.lazy == params.lazy
    @test F.params.verbose == params.verbose

    F = fkt(G) # also execute with default options
    @test F isa MultipoleFactorization
end

@testset "small data test" begin
    d = 2
    n = 64
    x = [randn(d) for _ in 1:n] # random input
    σ = 1e-2
    variance = fill(σ^2, n) # variance of output
    params = FactorizationParameters(; max_dofs_per_leaf = 8, precond_param = 0, trunc_param = 5)
    k = CovarianceFunctions.RQ(1.0)
    K = FmmMatrix(k, x, variance, params)
    F = fkt(K)
    @test F isa MultipoleFactorization

    l = 2
    xs1 = range(-l, l, length = 64)
    xs = [[xi, xj] for xi in xs1, xj in xs1]
    xs = vec(xs)
    ns = length(xs)

    for _ in 1:16 # this sometimes, but not always errors
        x = [randn(d) for _ in 1:n] # random input
        KS = FmmMatrix(k, xs, x, params)
        FS = fkt(KS)
        @test F isa MultipoleFactorization
        a = randn(n)
        MS = k.(xs, permutedims(x))
        @test isapprox(MS*a, FS*a, atol = atol, rtol = rtol)
        b = zeros(ns)
        @test isapprox(MS*a, mul!(b, FS, a), atol = atol, rtol = rtol)
    end
end

end # TestFastKernelTransform
