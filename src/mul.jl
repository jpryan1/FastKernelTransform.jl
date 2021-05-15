# matrix-vector multiplication and solves for MultipoleFactorization type
import LinearAlgebra: *, mul!, \
function *(F::MultipoleFactorization, x::AbstractVector; verbose::Bool = false)
    b = similar(x, size(F, 1))
    mul!(b, F, x, verbose = verbose)
end
function *(F::MultipoleFactorization, X::AbstractMatrix; verbose::Bool = false)
    B = similar(X, size(F, 1), size(X, 2))
    mul!(B, F, X, verbose = verbose)
end
\(F::MultipoleFactorization, b::AbstractVector) = conj_grad(F, b)

function mul!(Y::AbstractVector, A::MultipoleFactorization, X::AbstractVector,
              α::Real = 1, β::Real = 0; verbose::Bool = false)
    _mul!(Y, A, X, α, β, verbose = verbose)
end
function mul!(Y::AbstractMatrix, A::MultipoleFactorization, X::AbstractMatrix,
              α::Real = 1, β::Real = 0; verbose::Bool = false)
    _mul!(Y, A, X, α, β, verbose = verbose)
end

# IDEA: could pass data structure that reports how many compressions took place
function _mul!(y::AbstractVecOrMat, F::MultipoleFactorization, x::AbstractVecOrMat,
              α::Real = 1, β::Real = 0; verbose::Bool = false)
    _checksizes(y, F, x)
    comp_count = (0, 0) # total_compressed, total_not_compressed
    multipoles = allocate_multipoles(F, x) # move upward?
    compute_multipoles!(multipoles, F, x)
    @sync for (i, leaf) in enumerate(F.tree.allleaves)
        @spawn if !isempty(leaf.tgt_points) # race condition, with counters, but not necessary to be accurate
            leaf_comp_count = multiply_multipoles!(y, F, multipoles, leaf, x, α, β)
            comp_count = comp_count .+ leaf_comp_count
        end
    end
    total_compressed, total_not_compressed = comp_count
    verbose && println("Compressed: ",total_compressed," not compressed: ", total_not_compressed)
    return y
end

function allocate_multipoles(F::MultipoleFactorization, x::AbstractVector)
    T = transformation_eltype(F)
    zeros(T, (nmultipoles(F), length(F.tree.allnodes)))
end

function allocate_multipoles(F::MultipoleFactorization, X::AbstractMatrix)
    T = transformation_eltype(F)
    zeros(T, (nmultipoles(F), size(X, 2), length(F.tree.allnodes)))
end

# computes multipoles and stores them in place
function compute_multipoles!(multipoles::AbstractMatrix, F::MultipoleFactorization, x::AbstractVector)
    @. multipoles = 0
    @sync for (i, node) in enumerate(F.tree.allnodes) # computes all multipoles
        if !isempty(node.s2o)
            @spawn begin
                x_far = @view x[node.src_point_indices]
                multi = @view multipoles[:, i]
                mul!(multi, node.s2o, x_far)
            end
        end
    end
    return multipoles
end

function compute_multipoles!(multipoles::AbstractArray{<:Number, 3}, F::MultipoleFactorization, x::AbstractMatrix)
    @. multipoles = 0
    @sync for (i, node) in enumerate(F.tree.allnodes) # computes all multipoles
        if !isempty(node.s2o)
            @spawn begin
                x_far = @view x[node.src_point_indices, :]
                multi = @view multipoles[:, :, i]
                mul!(multi, node.s2o, x_far)
            end
        end
    end
    return multipoles
end

function multiply_multipoles!(y, F::MultipoleFactorization, multipoles,
                              leaf::BallNode, x, α::Real, β::Real)
    compressed, not_compressed = 0, 0
    xi = @views (x isa AbstractVector) ? x[leaf.near_point_indices] : x[leaf.near_point_indices, :]
    yi = @views (y isa AbstractVector) ? y[leaf.tgt_point_indices] : y[leaf.tgt_point_indices, :]
    mul!(yi, leaf.near_mat, xi, α, β) # near field interaction

    for far_node_idx in eachindex(leaf.far_nodes)
        far_node = leaf.far_nodes[far_node_idx]
        if isempty(far_node.src_points) continue end
        far_leaf_points = far_node.far_leaf_points
        far_src_points = length(far_node.src_points)
        if compression_is_efficient(F, far_node)
            compressed += 1
            xi = @views (x isa AbstractVector) ? multipoles[:, far_node.node_index] : multipoles[:, :, far_node.node_index]
        else
            not_compressed += 1
            xi = @views (x isa AbstractVector) ? x[far_node.src_point_indices] : x[far_node.src_point_indices, :]
        end
        o2i = leaf.o2i[far_node_idx]
        multiply_helper!(yi, o2i, xi, α)
    end
    return compressed, not_compressed
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

function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::LazyMultipoleMatrix{<:Complex}, xi::AbstractVecOrMat{<:Complex}, α::Real)
    multiply_helper!(yi, AbstractMatrix(o2i), xi, α)
end

function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::LazyMultipoleMatrix{<:Complex}, xi::AbstractVecOrMat{<:Real}, α::Real)
    multiply_helper!(yi, AbstractMatrix(o2i), xi, α)
end

function multiply_helper!(yi::AbstractVecOrMat{<:Real}, o2i::AbstractMatrix{<:Complex}, xi::AbstractVecOrMat{<:Complex}, α::Real)
    Re, Im = real_imag_views(o2i)
    re_xi, im_xi = real_imag_views(xi)
    mul!(yi, Re, re_xi, α, 1)
    mul!(yi, Im, im_xi, -α, 1)
end

# makes sure sizes of arguments for matrix multiplication agree
function _checksizes(y::AbstractVecOrMat, F::MultipoleFactorization, x::AbstractVecOrMat)
    if size(F, 2) ≠ size(x, 1)
        s = "second dimension of F, $(size(F, 2)), does not match length of x, $(length(x))"
        throw(DimensionMismatch(s))
    elseif size(y, 1) ≠ size(F, 1)
         s = "first dimension of F, $(size(F, 1)), does not match length of y, $(length(y))"
         throw(DimensionMismatch(s))
    elseif size(y, 2) ≠ size(x, 2)
        s = "second dimension of y, $(size(y, 2)), does not match second dimension of x, $(size(x, 2)))"
    end
end

############################### conjugate gradient solver ######################
function conj_grad(F::MultipoleFactorization, b::AbstractVector;
                   max_iter::Int = 128, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = false)
    conj_grad!(zero(b), F, b, max_iter = max_iter, tol = tol, precond = precond, verbose = verbose)
end

function conj_grad!(x::AbstractVector, F::MultipoleFactorization, b::AbstractVector;
                    max_iter::Int = 128, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = false)
    Ax = all(==(0), b) ? zero(x) : F * x
    r = b - Ax
    z = approx_inv(F, r)
    p = copy(z)
    Ap = similar(p)
    rsold = dot(r, z)

    for i in 1:max_iter
        mul!(Ap, F, p)

        alpha = rsold / dot(p, Ap)
        @. x = x + alpha * p
        @. r = r - alpha * Ap

        # if precond # what else changes?
        approx_inv!(z, F, r)
        # end
        rsnew = dot(r, z)
        verbose && println(i, " res ", rsnew)
        sqrt(rsnew) > tol || break
        @. p = z + (rsnew / rsold) * p;
        rsold = rsnew
    end
    return x
end

approx_inv(F::MultipoleFactorization, b::AbstractVector) = approx_inv!(zero(b), F, b)
function approx_inv!(total::AbstractVector, F::MultipoleFactorization, b::AbstractVector)
    @sync for cell in F.tree.allnodes
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
