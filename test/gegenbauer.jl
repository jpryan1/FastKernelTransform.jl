module TestGegenbauer
using Test
using FastKernelTransform: gegenbauer, chebyshev
using SpecialPolynomials: Gegenbauer

α = exp(randn())
n = 0
x = randn()
@testset "gegenbauer" begin
    @test gegenbauer(α, 0, x) ≈ 1
    @test gegenbauer(α, 1, x) ≈ 2α*x
    @test gegenbauer(α, 2, x) ≈ Gegenbauer{α}([0, 0, 1])(x)
    @test gegenbauer(α, 3, x) ≈ Gegenbauer{α}([0, 0, 0, 1])(x)
end

@testset "chebyshev" begin
    @test chebyshev(0, x) ≈ 1
    @test chebyshev(1, x) ≈ x
    @test chebyshev(2, x) ≈ 2x*chebyshev(1, x) - chebyshev(0, x)
    @test chebyshev(3, x) ≈ 2x*chebyshev(2, x) - chebyshev(1, x)

    @test chebyshev(1, x, Val(2)) ≈ 2x
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
