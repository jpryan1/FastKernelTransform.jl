# generic implementation of F and G expansion coefficients,
# works for any kernel that can be auto-differentiated with derivatives (fallback TaylorSeries.jl)

# initializing generic get_F and get_G functions
function init_F_G(kernel, dimension::Int, trunc_param::Int, doQR::Val{false} = Val(false))
    f = transformation_coefficients(dimension, trunc_param)
    get_F = init_F(kernel, trunc_param, f)
    radial_fun_ranks = zeros(Int, trunc_param + 1) # generic expansion is full rank
    for k in 0:trunc_param
        radial_fun_ranks[k+1] = floor(Int, (trunc_param - k) / 2 + 1)
    end
    return get_F, get_G, radial_fun_ranks
end

# this is the most efficient one
# f is transform_coef_table
function init_F(kernel, trunc_param::Int, f::AbstractArray{<:Real, 3})
    function get_F(r::AbstractVector{<:Real})
        dkernel = derivatives(kernel, r, trunc_param)
        @. dkernel *= r'^(0:trunc_param)
        s, fik = similar(r), similar(r, trunc_param+1)
        function F(k::Int, i::Int)
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
end

function get_G(r::AbstractVector{<:Real})
    s = similar(r)
    function G(k::Int, i::Int)
        @. s = r^(i-k)
    end
end
