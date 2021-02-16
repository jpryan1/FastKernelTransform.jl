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

# TODO: tests for rational qr, doublfact, real_imag_views

end
