mutable struct FmmMatrix{K, V<:AbstractVector{<:AbstractVector{<:Real}}, VT, PAR} # IDEA: use CovarianceFunctions.Gramian
    kernel::K
    tgt_points::V
    src_points::V
    variance::VT
    params::PAR
end

function FmmMatrix(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                   params::FactorizationParameters = FactorizationParameters())
    variance = nothing
    FmmMatrix(kernel, tgt_points, src_points, variance, params)
end

function FmmMatrix(kernel, points::AbstractVecOfVec{<:Real}, params::FactorizationParameters = FactorizationParameters())
    variance = nothing
    FmmMatrix(kernel, points, points, variance, params)
end

function FmmMatrix(kernel, points::AbstractVecOfVec{<:Real}, variance,
                    params::FactorizationParameters = FactorizationParameters())
    FmmMatrix(kernel, points, points, variance, params)
end

# fast kernel transform
function fkt(mat::FmmMatrix)
    MultipoleFactorization(mat.kernel, mat.tgt_points, mat.src_points, mat.variance, mat.params)
end

# factorize only calls fkt if it is worth it
function LinearAlgebra.factorize(mat::FmmMatrix)
    if max(length(mat.tgt_points), length(mat.src_points)) < mat.params.max_dofs_per_leaf
        x, y = mat.tgt_points, mat.src_points
        K = mat.kernel.(x, permutedims(y))
        if !isnothing(mat.variance)
            K += Diagonal(mat.variance)
        end
        return factorize(K)
    else
        return fkt(mat)
    end
end
