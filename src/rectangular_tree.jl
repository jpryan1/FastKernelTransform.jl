# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
# IDEA: use NearestNeighbors.jl and StaticArrays to speed up tree construction
# (has support for BallTrees)
mutable struct BallNode{PT<:AbstractVector{<:AbstractVector{<:Real}},
                      PIT<:AbstractVector{<:Int},
                      NT<:AbstractVector,
                      # OT<:AbstractVector{<:Number},
                      #MT<:AbstractMatrix{<:Real},
                      DT,
                      # ST<:AbstractMatrix{<:Number},
                      # OIT<:AbstractVector{AbstractMatrix{<:Number}},
                      CT}

    node_index::Int # index in allnodes vector of tree
    dimension::Int

    tgt_points::PT # how to keep this type general? (i.e. subarray) + 1d?
    tgt_point_indices::PIT

    near_indices::PIT # these are indices of points
    src_points::PT
    src_point_indices::PIT

    neighbors::NT
    neighbor_indices::PIT # indices of neighbors in tree.allnodes

    far_nodes::NT
    far_leaf_points::Int
    near_mat::AbstractMatrix # TODO: think about how to handle lazy / dense matrices elegantly
    diag_block::DT
    # Below are source2outgoing and outgoing2incoming mats, created at factor time
    s2o::AbstractMatrix
    o2i::AbstractVector # {AbstractMatrix{<:Number}} # TODO: think about how to handle lazy / dense matrices elegantly

    left_child::Union{BallNode, Nothing} # can be BallNode or Nothing
    right_child::Union{BallNode, Nothing} # can be BallNode or Nothing
    parent::Union{BallNode, Nothing} # can be BallNode or Nothing

    center::CT
    com::CT
    splitter_normal # nothing if leaf
    sidelens
    max_rprime
end

# constructor convenience
# TODO: pass lazy parameter to allocate appropriate types for s2o, o2i ...

function BallNode(node_idx::Int, dimension::Int, ctr::AbstractVector{<:Real},
                  tgt_points::AbstractVecOfVec{<:Real}, tgt_point_indices::AbstractVector{<:Int},
                  src_points::AbstractVecOfVec{<:Real}, src_point_indices::AbstractVector{<:Int}, sidelens)
  near_indices = zeros(Int, 0)
  neighbors = Vector(undef, 0)
  neighbor_indices = zeros(Int, 0)

  far_nodes = Vector(undef, 0)
  far_leaf_points = 0
  # T = eltype(tgt_points[1]) # TODO get type someway else
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # TODO this forces the DT type in NodeData to be a Cholesky, should it?
  s2o = zeros(Complex{Float64}, 0, 0)
  o2i = fill(s2o, 0)
  if ctr isa SVector
      ctr = MVector(ctr) # needs to be mutable
  end
  com = zero(ctr)
  BallNode(node_idx, dimension, tgt_points, tgt_point_indices,
           near_indices, src_points, src_point_indices, neighbors, neighbor_indices,
           far_nodes, far_leaf_points, near_mat, diag_block, s2o, o2i,
           nothing, nothing, nothing, ctr, com, nothing, sidelens, -1)
end

# calculates the total number of far points for a given leaf # TODO: deprecate?
function get_tot_far_points(leaf::BallNode)
    isempty(leaf.far_nodes) ? 0 : sum(node->length(node.src_points), leaf.far_nodes)
end
isleaf(node::BallNode) = node.splitter_normal == nothing

# calculates points of source and its neighborhood
function source_neighborhood_points(leaf::BallNode)
    src_points = leaf.src_points
    src_neighbor = [neighbor.src_points for neighbor in leaf.neighbors]
    # src_neighbor = (neighbor.src_points for neighbor in leaf.neighbors)
    # src_neighbor = (neighbor.src_points for neighbor in leaf.neighbors)
    src_points = vcat(leaf.src_points, src_neighbor...) # this allocates!
    return src_points
end
source_points(leaf) = leaf.src_point
target_points(leaf) = leaf.tgt_points

function source_neighborhood_indices!(leaf::BallNode)
    src_indices = leaf.src_point_indices
    src_indices = vcat(src_indices, [neighbor.src_point_indices for neighbor in leaf.neighbors]...)
    leaf.near_indices = src_indices
    # src_indices = vcat(src_indices, (neighbor.src_point_indices for neighbor in leaf.neighbors)...)
    # src_indices = ApplyVector(vcat, src_indices, (neighbor.src_point_indices for neighbor in leaf.neighbors)...)
    return src_indices
end

################################ tree structure ################################
struct Tree{T<:Real, R<:BallNode, V<:AbstractVector{<:BallNode}}
    dimension::Int64
    root::R
    max_dofs_per_leaf::Int64
    allnodes::V
    allleaves::V
    neighbor_scale::T
end

function initialize_tree(tgt_points, src_points, max_dofs_per_leaf,
                         neighbor_scale::Real = 1/2; barnes_hut::Bool = false, verbose::Bool = false)
    dimension = isempty(tgt_points) ? src_points : length(tgt_points[1])
    center = sum(vcat(tgt_points, src_points)) / (length(tgt_points) + length(src_points))
    root_sidelen = maximum((maximum(abs, difference(pt, center)) for pt in vcat(tgt_points, src_points)))
    root = BallNode(1, dimension, center, tgt_points, collect(1:length(tgt_points)),
      src_points, collect(1:length(src_points)), [2.01*root_sidelen for i in 1:dimension])

    allnodes = [root]
    allleaves = fill(root, 0)
    bt = Tree(dimension, root, max_dofs_per_leaf, allnodes, allleaves, neighbor_scale)
    if (length(tgt_points) + length(src_points)) > 2max_dofs_per_leaf
        rec_split!(bt, root)
    end

    for node in bt.allnodes
        if isleaf(node)
            push!(bt.allleaves, node)
        end
    end
    compute_near_far_nodes!(bt)
    barnes_hut && compute_center_of_mass!(bt)
    verbose && print_tree_statistics(bt)
    return bt
end

function plane_intersects_sphere(plane_center, splitter_normal,
        sphere_center, sphere_leafrad)
  sphere_right_pole = sphere_center + sphere_leafrad * splitter_normal
  sphere_left_pole = sphere_center - sphere_leafrad * splitter_normal
  return (dot(sphere_right_pole - plane_center, splitter_normal)
          * dot(sphere_left_pole - plane_center, splitter_normal) < 0)
end

function is_ancestor_of(leaf, node)
  cur = leaf
  while cur.parent != nothing
    if cur.parent == node return true end
    cur = cur.parent
  end
  return false
end

# Compute neighbor lists (note: ONLY FOR LEAVES at this time)
function compute_near_far_nodes!(bt)
  for leaf in bt.allleaves
    isempty(leaf.tgt_points) && continue
    node_queue = [bt.root]
    while length(node_queue) > 0
      cur_node = pop!(node_queue)
      if cur_node == leaf continue end
      isempty(cur_node.src_points) && continue
      # Are they overlapping
      if !isleaf(cur_node) && is_ancestor_of(cur_node, leaf)
        push!(node_queue, cur_node.right_child)
        push!(node_queue, cur_node.left_child)
        continue
      end

      # Is the ratio satisfied?
      min_r_val = minimum((norm(difference(pt, cur_node.center)) for pt in leaf.tgt_points))
      if cur_node.max_rprime == -1
        cur_node.max_rprime = maximum((norm(difference(pt, cur_node.center)) for pt in cur_node.src_points)) # pre-compute
      end
      max_rprime_val = cur_node.max_rprime
      # min_r_val = norm(difference(leaf.center, cur_node.center)) - norm(leaf.sidelens)/2
      # max_rprime_val = norm(cur_node.sidelens)/2
      if max_rprime_val/min_r_val > bt.neighbor_scale # no compression here
        if isleaf(cur_node)
          push!(leaf.neighbors, cur_node)
        else
          push!(node_queue, cur_node.right_child)
          push!(node_queue, cur_node.left_child)
        end
      else # compression here
        push!(leaf.far_nodes, cur_node)
        cur_node.far_leaf_points += length(leaf.tgt_points)
      end
    end
  end
end


using CovarianceFunctions: difference
function find_farthest(far_pt, pts)
    max_dist = 0
    cur_farthest = far_pt
    for p in pts
        dist = norm(difference(p, far_pt))
        if dist > max_dist
            max_dist = dist
            cur_farthest = p
        end
    end
    return cur_farthest
end


function rec_split!(bt, node)
  # pt_L = find_farthest(node.center, vcat(node.tgt_points, node.src_points))
  # pt_R = find_farthest(pt_L,  vcat(node.tgt_points, node.src_points))
  # splitter_normal = pt_R-pt_L
  # splitter_normal /= norm(splitter_normal)
  #
  # node.splitter_normal = splitter_normal
  min_split_dif = length(node.tgt_points) + length(node.src_points)
  max_sidelen = maximum(node.sidelens)
  candidate_dims = []
  for d in 1:node.dimension
    if node.sidelens[d] == max_sidelen
      push!(candidate_dims, d)
    end
  end
  best_d = candidate_dims[1]
  for d in candidate_dims
    candidate_splitter = zeros(node.dimension)
    candidate_splitter[d] = 1
    right = 0
    left = 0
    for i in eachindex(node.tgt_points)
      cpt = difference(node.tgt_points[i], node.center)
      if dot(cpt, candidate_splitter) > 0 # TODO not dot
        right += 1
      else
        left += 1
      end
    end
    for i in eachindex(node.src_points)
      cpt = difference(node.src_points[i], node.center)
      if dot(cpt, candidate_splitter) > 0
        right += 1
      else
        left += 1
      end
    end
    if abs(right-left) < min_split_dif
      min_split_dif = abs(right-left)
      best_d = d
    end
  end

  splitter_normal = zeros(node.dimension)
  splitter_normal[best_d] = 1
  node.splitter_normal = splitter_normal
  T = eltype(node.tgt_points)
  left_tgt_points = Vector{T}(undef, 0)
  right_tgt_points = Vector{T}(undef, 0)
  left_tgt_indices = zeros(Int, 0)
  right_tgt_indices = zeros(Int, 0)
  left_src_points = Vector{T}(undef, 0)
  right_src_points = Vector{T}(undef, 0)
  left_src_indices = zeros(Int, 0)
  right_src_indices = zeros(Int, 0)

  for i in eachindex(node.tgt_points)
    pt = node.tgt_points[i]
    cpt = difference(pt, node.center)
    if dot(cpt, splitter_normal) > 0
      push!(right_tgt_indices, node.tgt_point_indices[i])
      push!(right_tgt_points, pt)
    else
      push!(left_tgt_indices, node.tgt_point_indices[i])
      push!(left_tgt_points, pt)
    end
  end
  for i in eachindex(node.src_points)
    pt = node.src_points[i]
    cpt = difference(pt, node.center)
    if dot(cpt, splitter_normal) > 0
      push!(right_src_indices, node.src_point_indices[i])
      push!(right_src_points, pt)
    else
      push!(left_src_indices, node.src_point_indices[i])
      push!(left_src_points, pt)
    end
  end

  left_points = vcat(left_tgt_points, left_src_points)
  right_points = vcat(right_tgt_points, right_src_points)

  # left_center = node.center isa SVector ? MVector(node.center)copy(node.center)
  left_center = copy(node.center)
  left_center[best_d] -= (node.sidelens[best_d]/4)
  right_center = copy(node.center)
  #   right_center = node.center isa SVector ? MVector(node.center) : copy(node.center)
  right_center[best_d] += (node.sidelens[best_d]/4)

  new_sidelens = copy(node.sidelens)
  new_sidelens[best_d] /= 2

  num_nodes = length(bt.allnodes)
  left_node_idx = num_nodes + 1
  right_node_idx = num_nodes + 2
  left_node = BallNode(left_node_idx, node.dimension, left_center, left_tgt_points,
              left_tgt_indices, left_src_points, left_src_indices, new_sidelens)
  right_node = BallNode(right_node_idx, node.dimension, right_center, right_tgt_points,
              right_tgt_indices, right_src_points, right_src_indices, new_sidelens)

    left_node.parent = node
    right_node.parent = node # IDEA: recurse before constructing node?
    push!(bt.allnodes, left_node) # WARNING: not thread-safe
    push!(bt.allnodes, right_node)
    node.left_child = left_node
    node.right_child = right_node
    if length(left_points) > 2bt.max_dofs_per_leaf
        rec_split!(bt, left_node)
    end
    if length(right_points) > 2bt.max_dofs_per_leaf
        rec_split!(bt, right_node)
    end
    # @sync begin
    #     if length(left_points) > 2bt.max_dofs_per_leaf
    #         @spawn rec_split!(bt, left_node)
    #     end
    #     if length(right_points) > 2bt.max_dofs_per_leaf
    #         @spawn rec_split!(bt, right_node)
    #     end
    # end
end

function compute_center_of_mass!(bt::Tree)
    for n in bt.allleaves
        isempty(n.tgt_points) && continue
        avg_pt = zero(n.tgt_points[1])
        for pt in n.tgt_points
            avg_pt += pt
        end
        avg_pt ./= length(n.tgt_points)
        n.com = avg_pt
    end
    return bt
end

function print_tree_statistics(bt::Tree)
    tot_far = 0
    tot_near = 0
    tot_leaf_points = 0
    for n in bt.allleaves
        tot_leaf_points += length(n.tgt_points)
        tot_far += length(n.far_nodes)
        tot_near += length(n.neighbors)
    end
    num_neighbors = sum([length(node.neighbors) for node in bt.allleaves])
    println("Avg neighborhood: ", num_neighbors/length(bt.allleaves))
    vec = [length(node.tgt_points) for node in bt.allleaves]
    println("Num leaves ", length(bt.allleaves))
    println("Num nodes ", length(bt.allnodes))
    println("Avg far ", tot_far/length(bt.allleaves))
    println("Mean points ", mean(vec))
    println("Median points ", median(vec))
    println("Minimum points ", minimum(vec))
    println("Maximum points ", maximum(vec))
    println("Avg leaf_points ", tot_leaf_points/length(bt.allleaves))
end

# Random.seed!(4);
# N=3000
# dimension = 2
# max_dofs_per_leaf = 250
# # points  = [randn(dimension) for i in 1:N]  #
# points = [rand() > 0.5 ? randn(dimension) : 3*ones(dimension)+randn(dimension) for i in 1:N]
# bt = initialize_tree(points, points, max_dofs_per_leaf, 0, 1.4)
# circ = Shape(Plots.partialcircle(0, 2π))
#
# scatter([pt[1] for pt in points], [pt[2] for pt in points],
#   markerstrokecolor = nothing, markershape = circ, color="hot pink")
# for node in bt.allnodes
#   if isleaf(node) continue end
#   split = [-node.splitter_normal[2], node.splitter_normal[1]]
#   node_rad_L = -10
#   node_rad_R = 10
#   endpt_L = 0
#   endpt_R = 0
#   # find where node ray intersects parents
#   cur_node = node.parent
#   while cur_node != nothing
#     cur_split = [-cur_node.splitter_normal[2], cur_node.splitter_normal[1]]
#     dx = cur_node.center[1]-node.center[1]
#     dy = cur_node.center[2]-node.center[2]
#     det = cur_split[1] * split[2] - cur_split[2] * split[1]
#     u = (dy * cur_split[1] - dx * cur_split[2]) / det
#     v = (dy * split[1] - dx * split[2]) / det
#     if u > 0 && u < node_rad_R
#       node_rad_R = u
#     end
#     if u < 0 && u > node_rad_L
#       node_rad_L = u
#     end
#     cur_node = cur_node.parent
#   end
#   endpt_L = node.center + node_rad_L * split
#   endpt_R = node.center + node_rad_R * split
#   plot!([endpt_L[1],endpt_R[1]], [endpt_L[2],endpt_R[2]], width=3 , color="dark green")
# end
#
# leaf = bt.allleaves[5]
# leafrad = sqrt(sum((leaf.sidelens ./ 2) .^ 2))
# # x(t) = cos(t)*leafrad + leaf.center[1]
# # y(t) = sin(t)*leafrad + leaf.center[2]
# # plot!(x, y, 0, 2pi, linewidth=4, color="black")
# x2(t) = (3/sqrt(3))*cos(t)*leafrad + leaf.center[1]
# y2(t) = (3/sqrt(3))*sin(t)*leafrad + leaf.center[2]
# plot!(x2, y2, 0, 2pi, linewidth=4, color="black")
# # plot!( ylim=(-2,5), xlim=(-2,5), legend=false, ticks=false)
# plot!(legend=false, ticks=false)
# # plot!(xlim=(-5,8), ylim=(-5,8),size = (700,700))
# plot!(xlim=(-2,5), ylim=(-2,5),size = (700,700))
# plot!(axis=nothing, foreground_color_subplot=colorant"white")
# gui()
# # plot!( ylim=(0,1), xlim=(0,1), legend=false, ticks=false)
# # savefig("domain.pdf")
# end
