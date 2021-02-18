module TestUtil
using FastKernelTransform
using FastKernelTransform: LazyMultipoleMatrix, diagonal_correction!
using CovarianceFunctions: Gramian
using Test
using LinearAlgebra

@testset "util" begin
    n = 3
    A = randn(n, n)
    generator = ()->A
    L = LazyMultipoleMatrix{eltype(A)}(generator, n, n)
    x = randn(n)
    y = randn(n)
    mul!(y, L, x)
    @test y ≈ A*x

    # diagonal correction
    A = zeros(n, 2n)
    var = randn(n)
    A = diagonal_correction!(A, var, 1:n)
    @test diag(A) ≈ var

    x = [randn() for _ in 1:n]
    y = [randn() for _ in 1:2n]
    K = Gramian(dot, x, y)
    D = diagonal_correction!(K, var, 1:n)
    @test diag(D) ≈ @. x * y[1:n] + var
    MK = Matrix(K)
    for i in 1:length(var)
        MK[i, i] += var[i]
    end
    @test Matrix(D) ≈ MK
end

using FastKernelTransform: kcenters
@testset "k centers clustering" begin
    n, k, d = 128, 2, 5
    σ = .1
    x1 = [σ * randn(d) .+ 1 for _ in 1:n]
    x2 = [σ * randn(d) .- 1 for _ in 1:n]
    x = vcat(x1, x2)
    c, d, i = kcenters(x, k)
    @test length(c) == 2
    isin(c) = c in x
    @test all(isin, c) # centers are points in the data
    @test d[1] ≈ min(norm(x[1] - c[1]), norm(x[1] - c[2]))
    @test all(==(i[1]), i[1:n]) # first n points belong to the same cluster
    @test all(==(i[n+1]), i[n+1:end]) # last n points belong to the same cluster
end

# TODO: tests for rational qr, doublfact, real_imag_views

end
