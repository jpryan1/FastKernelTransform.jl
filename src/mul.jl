# matrix-vector multiplication and solves for MultipoleFactorization type
import LinearAlgebra: *, mul!, \
function *(fact::MultipoleFactorization, x::AbstractVector)
    b = zeros(Complex{Float64}, size(x)) # TODO: complex vector necessary?
    mul!(b, fact, x)
    real(b)
end
\(fact::MultipoleFactorization, b::AbstractVector) = conj_grad(fact, b)

function mul!(y::AbstractVector, fact::MultipoleFactorization, x::AbstractVector)
    _checksizes(y, fact, x)
    num_multipoles = binomial(fact.trunc_param+fact.tree.dimension, fact.trunc_param)
    counter = 0
    @sync for leaf in allleaves(fact.tree.root)
        if isempty(leaf.data.points) continue end
        @spawn begin
            xi = x[leaf.data.near_indices]
            yi = @view y[leaf.data.point_indices]
            mul!(yi, leaf.data.near_mat, xi, 1, 0) # near field interaction
            for far_node_idx in eachindex(leaf.data.far_nodes)
                far_node = leaf.data.far_nodes[far_node_idx]
                if isempty(far_node.data.points) continue end
                m = length(leaf.data.point_indices)
                n = length(far_node.data.point_indices)
                if num_multipoles * (m + n) < m * n # true if fast multiply is more efficient
                    counter += 1
                    if isempty(far_node.data.outgoing) # IDEA: have this pre-allocated in compute_transformation_mats
                        far_node.data.outgoing = far_node.data.s2o * x[far_node.data.point_indices]
                    end
                    xi = far_node.data.outgoing
                else
                    xi = x[far_node.data.point_indices]
                end
                mul!(yi, leaf.data.o2i[far_node_idx], xi, 1, 1) # yi should be real
            end
        end
    end
    # clean up IDEA: have this pre-allocated in compute_transformation_mats
    for cell in allcells(fact.tree.root)
        cell.data.outgoing = []
    end
    println("far field counter = $counter")
    return y
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
    for cell in allcells(fact.tree.root)
        if !isa(cell.data.diag_block, Factorization)
            continue
        end
        total[cell.data.point_indices] =  cell.data.diag_block \ b[cell.data.point_indices]
    end
    return total
end
