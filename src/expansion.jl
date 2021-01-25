# m is
function F(k::Int, j::Int, kernel, m::Int, f::AbstractArray{<:Real, 3})
    dkernel = zeros(m+1)
    fjk = @view f[j, k, :]
    function (r::Real)
        derivatives!(dkernel, kernel, r)
        @. dkernel *= r^((0:m)-j) * fjk
        return sum(dkernel)
    end
end

function F(fact::MultipoleFactorization, k::Int, j::Int)
    F(i, j, fact.k, fact.trunc_param, fact.transform_coef_table)
end

function get_F(kernel, r::AbstractVector{<:Real}, m::Int, f::AbstractArray{<:Real, 3})
    dkernel = derivatives(kernel, r, m)
    @. dkernel *= r'^(0:m)
    s, fik = similar(r), similar(r, m+1)
    function (k::Int, i::Int)
        if i == 0
            return @. s = kernel(r)*r
        else
            mi = 1:i
            dki = @view dkernel[mi .+ 1, :]
            fik = @view f[i+1, k+1, mi .+ 1]
            mul!(s, dki', fik)
            return @. s *= r^(k+1-i)
        end
    end
end
#
function get_F(fact::MultipoleFactorization, r::AbstractVector)
    get_F(fact.kernel, r, fact.trunc_param, fact.transform_coef_table)
end

function get_F(kernel, m::Int, f::AbstractArray{<:Real, 3})
    dkernel = zeros(m+1)
    function (k::Int, i::Int, r::Real)
        if i == 0
            return kernel(r)*r
        else
            mi = 1:i
            fik = @view f[i+1, k+1, mi .+ 1]
            derivatives!(dkernel, kernel, r)
            dk = @view dkernel[mi .+ 1]
            @. dk *= r^mi
            return r^(k+1-i) * dot(dk, fik)
        end
    end
end

function get_F(fact::MultipoleFactorization)
    get_F(fact.kernel, fact.trunc_param, fact.transform_coef_table)
end

# correctness?
# for F: @. dk *= r^((0:m)-j+i+1) * fji
# for G: (r::Real)->r^(j-i)


# function F(i::Int, j::Int, k, m::Int, r::Real, f::AbstractArray{<:Real, 3})
#     dk = derivatives(k, r, m)
#     fji = @view f[j, i, :]
#     @. dk *= r^((0:m)-j) * fji
#     return sum(dk)
# end
#
# function F(fact::MultipoleFactorization, i::Int, j::Int, r::Real)
#     F(i, j, fact.k, fact.trunc_param, r, fact.transform_coef_table)
# end

# function G(i::Int, j::Int, r::Real)
#     r^j
# end
function G(k::Int, i::Int)
    (r::Real)->r^(i-k)
end
# function G(i::Int, j::Int, r::Real)
#     r^j
# end

# potential micro-optimizations:
# c = zero(r)
# for mi in 0:m # IDEA: @simd @inbounds @fastmath
#     c += dk[mi+1] * r^(mi-j) * f[mi+1]
# end
# return c

# ri = (1/r)^j
# for mi in 0:m
#     ri =
#     @. dk * r^((0:mi) )
# end
