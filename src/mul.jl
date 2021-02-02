# matrix-vector multiplication and solves for MultipoleFactorization type
import LinearAlgebra: *, mul!, \
function *(fact::MultipoleFactorization, x::AbstractVector)
    b = similar(x, size(fact, 1))
    mul!(b, fact, x)
end
\(fact::MultipoleFactorization, b::AbstractVector) = conj_grad(fact, b)

function mul!(y::AbstractVector, fact::MultipoleFactorization, x::AbstractVector, α::Real = 1, β::Real = 0)
    _checksizes(y, fact, x)
    num_multipoles = binomial(fact.trunc_param + fact.tree.dimension, fact.trunc_param)
    total_compressed = 0
    total_not_compressed = 0
    @sync for leaf in fact.tree.allleaves
        if isempty(leaf.tgt_points) continue end
        @spawn begin
            xi = x[leaf.near_indices]
            yi = @view y[leaf.tgt_point_indices]
            mul!(yi, leaf.near_mat, xi, α, β) # near field interaction
            tot_far_points = sum([length(far_node.src_points) for far_node in leaf.far_nodes])

            for far_node_idx in eachindex(leaf.far_nodes)
                far_node = leaf.far_nodes[far_node_idx]
                if isempty(far_node.src_points) continue end
                m = length(leaf.tgt_points)
                if (num_multipoles * (m + tot_far_points)) < (m * tot_far_points)
                    total_compressed += 1
                    if isempty(far_node.outgoing) # IDEA: have this pre-allocated in compute_transformation_mats
                        far_node.outgoing = far_node.s2o * x[far_node.src_point_indices]
                    end
                    xi = far_node.outgoing
                else
                    total_not_compressed += 1
                    xi = x[far_node.src_point_indices]
                end
                o2i = leaf.o2i[far_node_idx]
                multiply_helper!(yi, o2i, xi, α)
            end
        end
    end
    println("Compressed: ",total_compressed," not compressed: ", total_not_compressed)
    # clean up IDEA: have this pre-allocated in compute_transformation_mats
    for cell in fact.tree.allnodes
        cell.outgoing = []
    end
    return y
end

# fallback for complex target
function multiply_helper!(yi::AbstractVector{<:Complex}, o2i::AbstractMatrix, xi::AbstractVector, α::Real)
    mul!(yi, leaf.o2i[far_node_idx], xi, α, 1) # yi is mathematically real
end
# only carries out relevant MVMs if target is real
# o2i is complex
function multiply_helper!(yi::AbstractVector{<:Real}, o2i::AbstractMatrix, xi::AbstractVector, α::Real)
    Re, Im = real_imag_views(o2i)
    if eltype(xi) <: Real
        mul!(yi, Re, xi, α, 1)
    else
        re_xi, im_xi = real_imag_views(xi)
        mul!(yi, Re, re_xi, α, 1)
        mul!(yi, Im, im_xi, -α, 1)
    end
    return yi
end

# makes sure sizes of arguments for matrix multiplication agree
function _checksizes(y::AbstractVector, fact::MultipoleFactorization, x::AbstractVector)
    if size(fact, 2) ≠ length(x)
        s = "second dimension of fact, $(size(fact, 2)), does not match length of x, $(length(x))"
        throw(DimensionMismatch(s))
    elseif length(y) ≠ size(fact, 1)
         s = "first dimension of fact, $(size(fact, 1)), does not match length of y, $(length(y))"
         throw(DimensionMismatch(s))
    end
end

function conj_grad(fact::MultipoleFactorization, b::Array{Float64, 1})
    x = rand(length(b))
    Ax = fact * x
    r = b - Ax
    z = approx_inv(fact, r)
    p = z
    rsold = dot(r,z)
    for i in 1:100
        @timeit fact.to "CG MV" Ap = fact * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        @timeit fact.to "CG LS" z = approx_inv(fact, r)
        rsnew = dot(r,z)
        println(i, " res ", rsnew)
        if sqrt(rsnew) < 1e-5
              break
        end
        p = z + (rsnew / rsold) * p;
        rsold = rsnew
    end
    return x
end

function approx_inv(fact::MultipoleFactorization, b)
    total = zero(b)
    for cell in fact.tree.allnodes
        if !isa(cell.diag_block, Factorization)
            continue
        end
        total[cell.point_indices] =  cell.diag_block \ b[cell.point_indices]
    end
    return total
end
