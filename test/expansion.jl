module TestExpansion
using Test
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: transformation_coefficients, get_F, G, gegenbauer

# TODO: loop test over kernels
# kernel(r) = 1/r
# kernel(r) = exp(-r^2)
kernel(r) = exp(-r)

d = 3
x = randn(d)
xp = randn(d)
xu = x / norm(x)
xpu = xp / norm(xp)
x = 3*xu
xp = xpu
order = 16
f = transformation_coefficients(d, order);

F = get_F(kernel, order, f)
r = norm(x)
rp = norm(xp)
FM = zeros(order+1, order+1)
for k in 0:order
    for i in k:2:order
        FM[k+1, i+1] = F(k, i, r)
    end
end

GM = zeros(size(FM))
for k in 0:order
    for i in k:2:order
        GM[k+1, i+1] = G(k, i)(rp)
    end
end

FG = vec(sum(FM .* GM, dims = 2))

cosγ = dot(xu, xpu)
k = 0:length(FG)-1
C = @. gegenbauer(1/2, k, cosγ) * rp^k / r^(k+1)
result = dot(C, FG)

@testset "expansion" begin
    @test result ≈ kernel(norm(x-xp))
end

end
