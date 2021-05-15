# For preconditioner
function compute_preconditioner!(fact::MultipoleFactorization, precond_param::Int,
                                 variance::Union{AbstractVector, Nothing} = fact.variance)
    node_queue = [fact.tree.root]
    @sync while !isempty(node_queue)
        node = pop!(node_queue)
        if length(node.tgt_point_indices) ≤ precond_param
            @spawn begin
                x = node.tgt_points
                K = fact.kernel.(x, permutedims(x)) # IDEA: can the kernel matrix be extracted from node.near_mat?
                K = diagonal_correction!(K, variance, node.tgt_point_indices)
                node.diag_block = cholesky!(K, Val(true), tol = 1e-7, check = false) # in-place
            end
        else
            push!(node_queue, node.left_child)
            push!(node_queue, node.right_child)
        end
    end
    return fact
 end
