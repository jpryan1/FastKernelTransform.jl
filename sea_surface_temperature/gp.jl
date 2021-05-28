# const VecOfVec{T} = AbstractVector{<:AbstractVector{T}}
using FastKernelTransform
using FastKernelTransform: FmmMatrix, FactorizationParameters
using CovarianceFunctions
using CovarianceFunctions: difference, EQ, RQ, MaternP, Cauchy, Exp
using LinearAlgebra

############################### Conditional Mean ###############################
# k is kernel of prior, (x, y) is data, variance is diagonal of noise covariance matrix or nothing
struct ConditionalMean{K, U<:AbstractVector, V<:AbstractVector}
    k::K
    x::U
    α::V
end

function ConditionalMean(kernel, x::AbstractVector, y::AbstractVector, var::AbstractVector;
    max_dofs_per_leaf = 8, precond_param = 2max_dofs_per_leaf, trunc_param = 5, fast::Bool = true)
    params = FactorizationParameters(; max_dofs_per_leaf = max_dofs_per_leaf,
                        precond_param = precond_param, trunc_param = trunc_param)
    F = nothing
    if fast
        K = FmmMatrix(kernel, x, var, params)
        F = fkt(K)
    else
        K = gramian(kernel, x)
        F = Matrix(K) + Diagonal(var)
    end
    α = F \ y
    ConditionalMean(k, x, α)
end

# for scalar predictions
function (M::ConditionalMean)(x::AbstractVector{<:Real})
    value = zero(eltype(x))
    for i in eachindex(M.α)
        value += M.k(x, M.x[i]) * M.α[i]
    end
    return value
end

# prediction for set of points in amortized quasi-constant time
function marginal(M::ConditionalMean, x::AbstractVector; max_dofs_per_leaf = 8,
                  precond_param = 0, trunc_param = 5, fast::Bool = true)
    μ = zeros(length(x))
    params = FactorizationParameters(; max_dofs_per_leaf = max_dofs_per_leaf,
                        precond_param = precond_param, trunc_param = trunc_param)
    F = nothing
    if fast
        K = FmmMatrix(M.k, x, M.x, params)
        F = fkt(K)
    else
        K = gramian(kernel, x)
        F = Matrix(K) + Diagonal(var)
    end
    return mul!(μ, F, M.α)
end
