module TestExpansion
using Test
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: transformation_coefficients, init_F_G, init_F, gegenbauer

@testset "expansion" begin
    # TODO: loop test over kernels
    kernel(r) = 1/r
    # kernel(r) = exp(-r^2)
    # kernel(r) = exp(-r)
    d = 3
    x = randn(d)
    xp = randn(d)
    xu = x / norm(x)
    xpu = xp / norm(xp)

    x = 3xu
    xp = xpu

    order = 10

    r = norm(x)
    rv = [r]
    rp = norm(xp)
    rpv = [rp]

    for doQR in (false, true)
        get_F, get_G = init_F_G(kernel, d, order, Val(doQR))
        FM = zeros(order+1, order+1)
        F = get_F(rv)
        for k in 0:order
            for i in k:2:order
                FM[k+1, i+1] = F(k, i)[1]
            end
        end

        G = get_G(rpv)
        GM = zeros(size(FM))
        for k in 0:order
            for i in k:2:order
                GM[k+1, i+1] = G(k, i)[1]
            end
        end

        FG = vec(sum(FM .* GM, dims = 2))
        cosγ = dot(xu, xpu)
        k = 0:length(FG)-1
        C = @. gegenbauer(1/2, k, cosγ) * rp^k / r^(k+1)
        result = dot(C, FG)

        atol = 1e-5
        @testset "qr = $doQR" begin
            @test isapprox(result, kernel(norm(x-xp)), atol = atol)
        end
    end

    # test that QR and generic implementation yield same results
    orders = 1:8
    error = zeros(length(orders), 2)
    for (i_qr, doQR) in enumerate((false, true))
        for (i_order, order) in enumerate(orders)
            get_F, get_G = init_F_G(kernel, d, order, Val(doQR))
            FM = zeros(order+1, order+1)
            F = get_F(rv)
            for k in 0:order
                for i in k:2:order
                    FM[k+1, i+1] = F(k, i)[1]
                end
            end

            G = get_G(rpv)
            GM = zeros(size(FM))
            for k in 0:order
                for i in k:2:order
                    GM[k+1, i+1] = G(k, i)[1]
                end
            end

            FG = vec(sum(FM .* GM, dims = 2))
            cosγ = dot(xu, xpu)
            k = 0:length(FG)-1
            C = @. gegenbauer(1/2, k, cosγ) * rp^k / r^(k+1)
            result = dot(C, FG)

            error[i_order, i_qr] = abs(result - kernel(norm(x-xp))) / kernel(norm(x-xp))
        end
    end
    generic_error, qr_error = @views error[:, 1], error[:, 2]
    rtol = 1e-6
    @testset "qr == generic" begin
        @test isapprox(generic_error, qr_error, rtol = rtol)
    end
    # using Plots
    # plot(orders, error, yscale = :log10)
    # gui()
end

end

# using SymEngine
# @vars sr
# sf = get_F([sr])(0, 0)
# for i in 0:5
#     for j in 0:5
#         println(get_F([sr])(i,j), " f for i=", i, " for j=",j)
#         println(get_G([sr])(i,j), " g for i=", i, " for j=",j)
#     end
# end
