module TestExpansion
using Test
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: transformation_coefficients, init_F_G, init_F, gegenbauer
using Printf
# using Plots

@testset "expansion" begin
    # TODO: loop test over kernels
    # kernel(r) = 1/r
    # kernel(r) = exp(-r^2)
    kernel(r) = 1/(1+r^2)
    println("cauchy")
    for d in [3 6 9 12]
        orders = collect(3:3:18)
        worst_errors = []
        N = 1000
        for order in orders
            get_F, get_G = init_F_G(kernel, d, order, Val(false))

            worst_err = 0
            for step in 1:N
                errors = []
                x = randn(d)
                xp = randn(d)
                xu = x / norm(x)
                xpu = xp / norm(xp)

                x = 2xu
                xp = xpu

                r = norm(x)
                rv = [r]
                rp = norm(xp)
                rpv = [rp]

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
                C = @. gegenbauer((d/2)-1, k, cosγ) * rp^k / r^(k+1)
                result = dot(C, FG)

                worst_err = max(worst_err,  abs(result-kernel(norm(x-xp))))
            end
            print("d=",d,",p=",order,",err=")
            @printf "%.2e" worst_err
            println("")
            push!(worst_errors,worst_err)
        end
        # println(d)
        # for e in worst_errors
        #     print(e)
        #     print(" ")
        # end
        # println("")
        # if d==3
        #     plot(orders, worst_errors, seriestype = :scatter, yscale = :log10, label="d=3")
        # else
        #     plot!(orders, worst_errors, seriestype = :scatter, yscale = :log10, label="d=8")
        # end
    end
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
