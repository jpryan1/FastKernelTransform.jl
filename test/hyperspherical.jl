module TestHyperspherical
using Test
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: hyperspherical, get_multiindices, hyper_normalizer,
                            cart2hyp, hyp2cart, gegenbauer, gegenbauer_normalizer
@testset "hyperspherical" begin

    # testing conversion from cartesian to hyper-spherical coordinates
    @testset "coordinates" begin
        d = 3
        x = randn(d)
        h = cart2hyp(x)
        @test h[1] ≈ norm(x)
        @test x[3] ≈ norm(x) * cos(h[2])
        for d in 3:10
            x = randn(d)
            @test x ≈ hyp2cart(cart2hyp(x))
        end
    end

    # testing hyper-spherical harmonics
    @testset "harmonics" begin
        for d in 3:10 # testing addition theorem
            k = 0
            α = d/2 - 1
            M = get_multiindices(d, k)

            x, y = randn(d), randn(d)
            x /= norm(x); y /= norm(y)
            cosγ = dot(x, y)

            C = gegenbauer(α, k, cosγ)
            N = gegenbauer_normalizer(d, k)

            h, hp = cart2hyp(x), cart2hyp(y)
            H = [hyperspherical(h, k, μ) for μ in M]
            Hp = [hyperspherical(hp, k, μ) for μ in M]
            # println([hyperspherical(hp, k, μ, Val(false)) for μ in M])
            @test C / N ≈ dot(H, Hp)
        end
    end
end

# benchmarking against SpecialPolynomials
# using SpecialPolynomials: Gegenbauer
# using BenchmarkTools
# α = 1/2
# n = 0
# x = randn()
# println("existing implementation")
# @btime $Gegenbauer{$α}($[0, 0, 0, 1])($x)
# println("costum implementation")
# t = @btime $gegenbauer($α, 4, $x) # ~300 x on my machine

end # module
