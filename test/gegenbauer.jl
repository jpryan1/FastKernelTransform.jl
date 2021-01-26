module TestGegenbauer
using Test
using FastKernelTransform: gegenbauer

α = exp(randn())
n = 0
x = randn()
@testset "gegenbauer" begin
    @test gegenbauer(α, 0, x) ≈ 1
    @test gegenbauer(α, 1, x) ≈ 2α*x
    @test gegenbauer(α, 2, x) ≈ Gegenbauer{α}([0, 0, 1])(x)
    @test gegenbauer(α, 3, x) ≈ Gegenbauer{α}([0, 0, 0, 1])(x)
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
