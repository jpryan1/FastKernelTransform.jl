using NearestNeighbors

mutable struct NNNode
    center
    radius
    point_indices
    near_point_indices
    far_point_indices
    diag_block
    near_mat
    s2o
    o2i
end

struct NNTree
    kd_tree
    points
    idx_to_NNNode
    node_indices
    leaf_indices
end


function compute_self_indices!(tree, idx)
    push!(tree.node_indices, idx)
    if NearestNeighbors.isleaf(tree.kd_tree.tree_data.n_internal_nodes, idx)
        push!(tree.leaf_indices, idx)
        point_indices = tree.kd_tree.indices[NearestNeighbors.get_leaf_range(tree.kd_tree.tree_data, idx)]
        #TODO respect the trees reordering
    else
        compute_self_indices!(tree, NearestNeighbors.getleft(idx))
        compute_self_indices!(tree, NearestNeighbors.getright(idx))
        left_child = tree.idx_to_NNNode[NearestNeighbors.getleft(idx)]
        right_child = tree.idx_to_NNNode[NearestNeighbors.getright(idx)]
        point_indices = collect(union(Set(left_child.point_indices), Set(right_child.point_indices)))
    end

    points = tree.points[point_indices]
    center = sum(points)/length(points)
    radius = maximum(norm(pt - center) for pt in points)

    tree.idx_to_NNNode[idx] =  NNNode(center, radius, point_indices,
        nothing,nothing,nothing,nothing,nothing,nothing)
end


function compute_near_far_indices!(tree, idx)

    parent_near_indices = []
    if idx != 1
        parent_near_indices = tree.idx_to_NNNode[NearestNeighbors.getparent(idx)].near_point_indices
    else
        parent_near_indices = collect(1:length(tree.points))
    end

    tree_node = tree.idx_to_NNNode[idx]
    rprime = tree_node.radius
    min_dist_for_compress = rprime/0.75
    point_indices = tree.idx_to_NNNode[idx].point_indices
    near_point_indices = setdiff(
            Set(NearestNeighbors.inrange(
                tree.kd_tree, tree_node.center, min_dist_for_compress)
            ), Set(point_indices))
    far_point_indices = setdiff(Set(parent_near_indices) ,Set(near_point_indices))

    tree.idx_to_NNNode[idx].near_point_indices = collect(near_point_indices)
    tree.idx_to_NNNode[idx].far_point_indices = collect(far_point_indices)


    if !NearestNeighbors.isleaf(tree.kd_tree.tree_data.n_internal_nodes, idx)
        compute_near_far_indices!(tree, NearestNeighbors.getleft(idx))
        compute_near_far_indices!(tree, NearestNeighbors.getright(idx))
    else
        println("Testing pt indices is ", length(point_indices))
        println("inside is ", length(NearestNeighbors.inrange(
            tree.kd_tree, tree_node.center, tree_node.radius)))
        println("near pt indices is ", length(near_point_indices))
    end
end


function compute_near_far_indices!(tree)
    # recursively find at node's far points. If leaf, also near points
    # far will need to have intersection with parent removed.
    compute_self_indices!(tree, 1)
    compute_near_far_indices!(tree, 1)

end


function initialize_tree_nn(tgt_points, src_points, max_dofs_per_leaf,
                         neighbor_scale::Real = 1/2;
                         barnes_hut::Bool = false, verbose::Bool = false, lazy::Bool = false)
    dimension = isempty(tgt_points) ? src_points : length(tgt_points[1])

    # Assume same tgt_points and src_points for now
    kd_tree = NearestNeighbors.KDTree(hcat(tgt_points...), leafsize=max_dofs_per_leaf)


    tree = NNTree(kd_tree, tgt_points,
        Array{NNNode,1}(undef, kd_tree.tree_data.n_internal_nodes+kd_tree.tree_data.n_leafs),
        [],[])

    compute_near_far_indices!(tree)

    all_idcs = []
    for leaf_idx in tree.leaf_indices
        leaf_node = tree.idx_to_NNNode[leaf_idx]
        append!(all_idcs, leaf_node.point_indices)
    end
    # barnes_hut && compute_center_of_mass!(bt)
    # verbose && print_tree_statistics(bt)
    return tree
end
