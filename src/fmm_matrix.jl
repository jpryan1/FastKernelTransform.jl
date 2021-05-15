mutable struct FmmMatrix{K, V<:AbstractVector{<:AbstractVector{<:Real}}, VT} # IDEA: use CovarianceFunctions.Gramian
    kernel::K
    tgt_points::V
    src_points::V
    max_dofs_per_leaf::Int64
    precond_param::Int64
    trunc_param::Int64
    variance::VT
end

function FmmMatrix(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                   max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int)
    variance = nothing
    return FmmMatrix(kernel, tgt_points, src_points, max_dofs_per_leaf, precond_param,
                     trunc_param, variance)
end

function FmmMatrix(kernel, points::AbstractVecOfVec{<:Real},
                   max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int,
                   variance = nothing)
    return FmmMatrix(kernel, points, points, max_dofs_per_leaf, precond_param,
                     trunc_param, variance)
end

# fast kernel transform
function fkt(mat::FmmMatrix)
    return MultipoleFactorization(mat.kernel, mat.tgt_points, mat.src_points,
        mat.max_dofs_per_leaf, mat.precond_param, mat.trunc_param, mat.variance)
end

# factorize only calls fkt if it is worth it
function LinearAlgebra.factorize(mat::FmmMatrix)
    if max(length(mat.tgt_points),length(mat.src_points)) < mat.max_dofs_per_leaf
        x, y = mat.tgt_points, mat.src_points
        return factorize(mat.kernel.(x, permutedims(y)))
    else
        return fkt(mat)
    end
end
