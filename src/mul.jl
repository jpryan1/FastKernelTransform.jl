# matrix-vector multiplication and solves for MultipoleFactorization type
import LinearAlgebra: *, mul!, \
function *(fact::MultipoleFactorization, x::AbstractVector; verbose::Bool = false)
    b = similar(x, size(fact, 1))
    mul!(b, fact, x, verbose = verbose)
end
function *(fact::MultipoleFactorization, X::AbstractMatrix; verbose::Bool = false)
    B = similar(X, size(fact, 1), size(X, 2))
    mul!(B, fact, X, verbose = verbose)
end
\(fact::MultipoleFactorization, b::AbstractVector) = conj_grad(fact, b)

function mul!(Y::AbstractVector, A::MultipoleFactorization, X::AbstractVector,
              α::Real = 1, β::Real = 0; verbose::Bool = false)
    _mul!(Y, A, X, α, β, verbose = verbose)
end
function mul!(Y::AbstractMatrix, A::MultipoleFactorization, X::AbstractMatrix,
              α::Real = 1, β::Real = 0; verbose::Bool = false)
    _mul!(Y, A, X, α, β, verbose = verbose)
end

# this is not thread-safe
# ASSUMES node.outgoing is pre-allocated
# function mul!(y::AbstractVector, fact::MultipoleFactorization, x::AbstractVector,
#         thread_safe::Val{false}, α::Real = 1, β::Real = 0; verbose::Bool = false)
#     _checksizes(y, fact, x)
#     num_multipoles = length(keys(fact.multi_to_single))
#     total_compressed = 0
#     total_not_compressed = 0
#
#     @sync for node in fact.tree.allnodes # computes all multipoles
#         if !isempty(node.s2o)
#             @spawn begin
#                 x_far_src = @view x[node.src_point_indices]
#                 mul!(node.outgoing, node.s2o, x_far_src) # WARNING: this is not thread-safe (multiply several vectors in parallel)
#             end
#         end
#     end
#
#     @sync for leaf in fact.tree.allleaves
#         if isempty(leaf.tgt_points) continue end
#         @spawn begin
#             xi = @view x[leaf.near_indices]
#             yi = @view y[leaf.tgt_point_indices]
#             mul!(yi, leaf.near_mat, xi, α, β) # near field interaction
#             tot_far_points = get_tot_far_points(leaf)
#             for far_node_idx in eachindex(leaf.far_nodes)
#                 far_node = leaf.far_nodes[far_node_idx]
#                 if isempty(far_node.src_points) continue end
#                 far_leaf_points = far_node.far_leaf_points
#                 far_src_points = length(far_node.src_points)
#                 if (num_multipoles * (far_src_points + far_leaf_points)) < (far_src_points * far_leaf_points) # only use multipoles if it is efficient
#                     total_compressed += 1
#                     xi = far_node.outgoing
#                 else
#                     total_not_compressed += 1
#                     xi = @view x[far_node.src_point_indices]
#                 end
#                 o2i = leaf.o2i[far_node_idx]
#                 multiply_helper!(yi, o2i, xi, α)
#             end
#         end
#     end
#     verbose && println("Compressed: ",total_compressed," not compressed: ", total_not_compressed)
#     return y
# end

# IDEA: could pass data structure that reports how many compressions took place
function _mul!(y::AbstractVecOrMat, fact::MultipoleFactorization, x::AbstractVecOrMat,
              α::Real = 1, β::Real = 0, thread_safe::Union{Val{true}, Val{false}} = Val(true);
              verbose::Bool = false)
    _checksizes(y, fact, x)
    num_multipoles = length(keys(fact.multi_to_single))
    total_compressed = 0
    total_not_compressed = 0

    compute_multipoles!(y, fact, x, thread_safe)

    @sync for leaf in fact.tree.allleaves
        if isempty(leaf.tgt_points) continue end
        @spawn begin
            xi = @view x[leaf.near_indices, :]
            yi = @view y[leaf.tgt_point_indices, :]
            mul!(yi, leaf.near_mat, xi, α, β) # near field interaction
            tot_far_points = get_tot_far_points(leaf)
            for far_node_idx in eachindex(leaf.far_nodes)
                far_node = leaf.far_nodes[far_node_idx]
                if isempty(far_node.src_points) continue end
                far_leaf_points = far_node.far_leaf_points
                far_src_points = length(far_node.src_points)
                if (num_multipoles * (far_src_points + far_leaf_points)) < (far_src_points * far_leaf_points) # only use multipoles if it is efficient
                    total_compressed += 1
                    xi = far_node.outgoing
                else
                    total_not_compressed += 1
                    xi = @view x[far_node.src_point_indices, :]
                end
                o2i = leaf.o2i[far_node_idx]
                multiply_helper!(yi, o2i, xi, α)
            end
        end
    end
    verbose && println("Compressed: ",total_compressed," not compressed: ", total_not_compressed)
    return y
end

function compute_multipoles!(y, fact, x, thread_safe::Union{Val{false}, Val{true}} = Val(true))
    @sync for node in fact.tree.allnodes # computes all multipoles
        if !isempty(node.s2o)
            @spawn begin
                x_far_src = @views x isa AbstractVector ? x[node.src_point_indices] : x[node.src_point_indices, :]
                if thread_safe isa Val{true}
                    node.outgoing = node.s2o * x_far_src # thread safe
                else
                    mul!(node.outgoing, node.s2o, x_far_src) # WARNING: this is not thread-safe (multiply several vectors in parallel)
                end
            end
        end
    end
end

# fallback
function multiply_helper!(yi::AbstractVecOrMat, o2i::AbstractMatrix, xi::AbstractVecOrMat, α::Real)
    mul!(yi, o2i, xi, α, 1) # yi is mathematically real
end
# only carries out relevant MVMs if target is real
# o2i is complex
function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::AbstractMatrix{<:Complex}, xi::AbstractVecOrMat{<:Real}, α::Real)
    Re, Im = real_imag_views(o2i)
    mul!(yi, Re, xi, α, 1)
end

function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::LazyMultipoleMatrix, xi::AbstractVecOrMat{<:Complex}, α::Real)
    multiply_helper!(yi, Matrix(o2i), xi, α)
end

function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::AbstractMatrix{<:Complex}, xi::AbstractVecOrMat{<:Complex}, α::Real)
    Re, Im = real_imag_views(o2i)
    re_xi, im_xi = real_imag_views(xi)
    mul!(yi, Re, re_xi, α, 1)
    mul!(yi, Im, im_xi, -α, 1)
end

# makes sure sizes of arguments for matrix multiplication agree
function _checksizes(y::AbstractVecOrMat, fact::MultipoleFactorization, x::AbstractVecOrMat)
    if size(fact, 2) ≠ size(x, 1)
        s = "second dimension of fact, $(size(fact, 2)), does not match length of x, $(length(x))"
        throw(DimensionMismatch(s))
    elseif size(y, 1) ≠ size(fact, 1)
         s = "first dimension of fact, $(size(fact, 1)), does not match length of y, $(length(y))"
         throw(DimensionMismatch(s))
    elseif size(y, 2) ≠ size(x, 2)
        s = "second dimension of y, $(size(y, 2)), does not match second dimension of x, $(size(x, 2)))"
    end
end

function conj_grad(fact::MultipoleFactorization, b::Array{Float64, 1};
                   max_iter::Int = 128, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = false)
    conj_grad!(zero(b), fact, b, max_iter = max_iter, tol = tol, precond = precond, verbose = verbose)
end

function conj_grad!(x::AbstractVector, fact::MultipoleFactorization, b::AbstractVector;
                    max_iter::Int = 128, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = false)
    Ax = all(==(0), b) ? zero(x) : fact * x
    r = b - Ax
    z = approx_inv(fact, r)
    p = copy(z)
    Ap = similar(p)
    rsold = dot(r, z)

    for i in 1:max_iter
        @timeit fact.to "CG MV" mul!(Ap, fact, p)

        alpha = rsold / dot(p, Ap)
        @. x = x + alpha * p
        @. r = r - alpha * Ap

        # if precond # what else changes?
        @timeit fact.to "CG LS" approx_inv!(z, fact, r)
        # end
        rsnew = dot(r, z)
        verbose && println(i, " res ", rsnew)
        sqrt(rsnew) > tol || break
        @. p = z + (rsnew / rsold) * p;
        rsold = rsnew
    end
    return x
end

approx_inv(fact::MultipoleFactorization, b::AbstractVector) = approx_inv!(zero(b), fact, b)
function approx_inv!(total::AbstractVector, fact::MultipoleFactorization, b::AbstractVector)
    @sync for cell in fact.tree.allnodes
        A = cell.diag_block
        if A isa Factorization && prod(size(A)) > 0
            @spawn begin # IDEA: cell.tgt_point_indices ≡ cell.src_point_indices || throw()
                ind = cell.tgt_point_indices
                x = b[ind]
                ldiv!(A, x)
                @. total[ind] = x
            end
        end
    end
    return total
end
