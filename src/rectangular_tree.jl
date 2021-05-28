# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
# IDEA: use NearestNeighbors.jl and StaticArrays to speed up tree construction
# (has support for BallTrees)
using CovarianceFunctions: difference

mutable struct BallNode{
                      PIT<:AbstractVector{<:Int},
                      DT<:AbstractMatOrFac,
                      CT<:AbstractVector{<:Number},
                      SIDE, XPRIME<:Real}
    node_index::Int # index in allnodes vector of tree
    tgt_point_indices::PIT
    src_point_indices::PIT
    near_point_indices::PIT # these are indices of points
    near_point_indices_set::BitSet # these are indices of points
    far_point_indices::PIT
    near_mat::AbstractMatrix
    diag_block::DT

    # Below are source2outgoing and outgoing2incoming mats, created at factor time
    s2o::AbstractMatrix
    o2i::AbstractMatrix

    # lazy_s20::AbstractMatrix
    # lazy_o2i::O2I

    left_child::Union{BallNode, Nothing} # initialized after this node is created
    right_child::Union{BallNode, Nothing}
    parent::Union{BallNode, Nothing}

    center::CT
    center_of_mass::CT
    splitter_normal::Union{AbstractVector, Nothing} # nothing if leaf
    sidelens::SIDE
    max_rprime::XPRIME
end

# constructor convenience
function BallNode(node_idx::Int, ctr::AbstractVector{<:Real},
                   tgt_point_indices::AbstractVector{<:Int},
                   src_point_indices::AbstractVector{<:Int},
                  sidelens, lazy::Bool)
  near_point_indices = zeros(Int, 0)
  near_point_indices_set = BitSet(near_point_indices)

  far_point_indices = zeros(Int, 0)

  # T = eltype(tgt_points[1]) # TODO get type someway else
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # this forces the DT type in NodeData to be a Cholesky, should it?
  T = Float64 # eltype of transformations
  T = Complex{T}
  if lazy # TODO: pass lazy parameter to allocate appropriate types for s2o, o2i ...
      M = LazyMultipoleMatrix{T}
      s2o = zeros(0, 0)
      o2i = zeros(0, 0)
  else
      s2o = zeros(T, 0, 0)
      o2i = zeros(T, 0, 0)
  end

  if ctr isa SVector
      ctr = MVector(ctr) # needs to be mutable
  end
  center_of_mass = zero(ctr)
  max_rprime = NaN
  BallNode(node_idx, tgt_point_indices,
           src_point_indices, near_point_indices, near_point_indices_set,
           far_point_indices, near_mat, diag_block, s2o, o2i,
           nothing, nothing, nothing, ctr, center_of_mass, nothing, sidelens, max_rprime)
end

# calculates the total number of far points for a given leaf # TODO: deprecate?

isleaf(node::BallNode) = node.splitter_normal == nothing

# IDEA: experiment with viewing into the data representation of the tree,
# since it might be more cache friendly through reordering
get_source_points(tree, leaf) = @view tree.src_points[leaf.src_point_indices]
get_target_points(tree, leaf) = @view tree.tgt_points[leaf.tgt_point_indices]
get_near_points(tree, leaf) = @view tree.tgt_points[leaf.near_point_indices]
get_far_points(tree, leaf) = @view tree.tgt_points[leaf.far_point_indices]


################################ tree structure ################################
struct Tree{T<:Real, R<:BallNode, TGT<:AbstractVecOfVec, SRC<:AbstractVecOfVec,
            V<:AbstractVector{<:BallNode}}
    kd_tree::KDTree
    tgt_points::TGT
    src_points::SRC
    dimension::Int64
    root::R
    max_dofs_per_leaf::Int64
    allnodes::V
    allleaves::V
    neighbor_scale::T
end


function initialize_tree(tgt_points, src_points, max_dofs_per_leaf,
                         neighbor_scale::Real = 1/2;
                         barnes_hut::Bool = false, verbose::Bool = false, lazy::Bool = false)
    # Note that tgt_points are used for near and far points always
    # TODO Sebastian (smarter thing than hcat here)
    kd_tree = NearestNeighbors.KDTree(hcat(tgt_points...))
    #hcat not efficient, instead pass pre-init matrix or vec of stat arrays

    dimension = isempty(tgt_points) ? src_points : length(tgt_points[1])
    center = sum(vcat(tgt_points, src_points)) / (length(tgt_points) + length(src_points))
    root_sidelen = maximum((maximum(abs, difference(pt, center)) for pt in vcat(tgt_points, src_points)))
    node_index = 1
    root = BallNode(node_index, center, collect(1:length(tgt_points)), #TODO it is unnecessary to store points in nodes like the root where interactions aren't computed
      collect(1:length(src_points)), [2.01*root_sidelen for i in 1:dimension], lazy)

    allnodes = [root]
    allleaves = fill(root, 0)
    bt = Tree(kd_tree, tgt_points, src_points, dimension, root,
      max_dofs_per_leaf, allnodes, allleaves, neighbor_scale)
    if (length(tgt_points) + length(src_points)) > 2max_dofs_per_leaf
        rec_split!(bt, root, lazy)
    end

    for node in bt.allnodes # parallel over nodes?
        if isnan(node.max_rprime) && !isempty(node.src_point_indices)
            node_src_points = src_points[node.src_point_indices]
            node_tgt_points = tgt_points[node.tgt_point_indices]
          node.max_rprime = maximum((norm(difference(pt, node.center)) for pt in vcat(node_src_points,node_tgt_points))) # pre-compute
        end
        if isleaf(node) #not par
            push!(bt.allleaves, node)
        end
    end

    to = TimerOutput()
    bt.allnodes[1].near_point_indices_set = BitSet(1:length(bt.tgt_points))
    # get indices for all subtrees rooted at child of root.

    @timeit to "levels" levels = [nodes_above(root, node) for node in bt.allnodes]
    top_lev = maximum(levels)
    @timeit to "inrange" begin
    for lev in 1:top_lev
      level_nodes =  bt.allnodes[levels .== lev]
      @sync for node in level_nodes
         @spawn if !isempty(node.src_point_indices)
          level = nodes_above(root, node)
          rprime = node.max_rprime
          min_dist_for_compress = rprime/bt.neighbor_scale
          # @timeit to string("inrange", level)
          inrlist = NearestNeighbors.inrange(bt.kd_tree, node.center, min_dist_for_compress)
          # reorder list so src_point_indices and near_point_indices start the same way for sym
          # @timeit to string("setc", level)
          inr = BitSet(inrlist)

          # @timeit to string("intersect", level)
          node.near_point_indices_set = intersect(node.parent.near_point_indices_set, inr)

          # @timeit to string("far", level) begin
          #TODO(Sebastian) if we precondition, only do this reordering on precond
          # nodes determined by precond param
          proper_order_near = vcat(node.tgt_point_indices, collect(setdiff(node.near_point_indices_set, BitSet(node.tgt_point_indices))))
          node.near_point_indices = collect(proper_order_near)
          node.far_point_indices = collect(setdiff(node.parent.near_point_indices_set, node.near_point_indices_set))
          # end
        end
      end
    end
    end
    # examine difference in these costs across different levels.
    # 1) init bd tree for every node, no need for intersection
    # 2) Possible that new algorithmic idea could handle point set operations

    # display(to)
    # print_tree_debug(root, 1)
    barnes_hut && compute_center_of_mass!(bt)
    # verbose && print_tree_statistics(bt)
    return bt
end

function rec_split!(bt::Tree, node::BallNode, lazy::Bool)
  # pt_L = find_farthest(node.center, vcat(node.tgt_points, node.src_points))
  # pt_R = find_farthest(pt_L,  vcat(node.tgt_points, node.src_points))
  # splitter_normal = pt_R-pt_L
  # splitter_normal /= norm(splitter_normal)
  #
  # node.splitter_normal = splitter_normal
  min_split_dif = length(node.tgt_point_indices) + length(node.src_point_indices)
  max_sidelen = maximum(node.sidelens)
  candidate_dims = []
  for d in 1:bt.dimension
    if node.sidelens[d] == max_sidelen
      push!(candidate_dims, d)
    end
  end
  best_d = candidate_dims[1]
  tgt_points = bt.tgt_points[node.tgt_point_indices]
  src_points = bt.src_points[node.src_point_indices]

  for d in candidate_dims
    candidate_splitter = zeros(bt.dimension)
    candidate_splitter[d] = 1
    right = 0
    left = 0
    for i in eachindex(tgt_points)
      cpt = difference(tgt_points[i], node.center)
      if dot(cpt, candidate_splitter) > 0 # TODO not dot
        right += 1
      else
        left += 1
      end
    end
    for i in eachindex(src_points)
      cpt = difference(src_points[i], node.center)
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

  splitter_normal = zeros(bt.dimension)
  splitter_normal[best_d] = 1
  node.splitter_normal = splitter_normal
  left_tgt_indices = zeros(Int, 0)
  right_tgt_indices = zeros(Int, 0)
  left_src_indices = zeros(Int, 0)
  right_src_indices = zeros(Int, 0)

  for i in eachindex(tgt_points)
    pt = tgt_points[i]
    cpt = difference(pt, node.center)
    if dot(cpt, splitter_normal) > 0
      push!(right_tgt_indices, node.tgt_point_indices[i])
    else
      push!(left_tgt_indices, node.tgt_point_indices[i])
    end
  end
  for i in eachindex(src_points)
    pt = src_points[i]
    cpt = difference(pt, node.center)
    if dot(cpt, splitter_normal) > 0
      push!(right_src_indices, node.src_point_indices[i])
    else
      push!(left_src_indices, node.src_point_indices[i])
    end
  end

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
  left_node = BallNode(left_node_idx, left_center,
              left_tgt_indices, left_src_indices, new_sidelens, lazy)
  right_node = BallNode(right_node_idx, right_center,
              right_tgt_indices, right_src_indices, new_sidelens, lazy)

    left_node.parent = node
    push!(bt.allnodes, left_node) # WARNING: not thread-safe
    node.left_child = left_node

    right_node.parent = node
    push!(bt.allnodes, right_node)
    node.right_child = right_node

    # recurse down left and right node
    if length(left_tgt_indices)+length(left_src_indices) > 2bt.max_dofs_per_leaf
        rec_split!(bt, left_node, lazy) # IDEA: recurse before constructing node?
    end
    if length(right_tgt_indices)+length(right_src_indices) > 2bt.max_dofs_per_leaf
        rec_split!(bt, right_node, lazy)
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


function nodes_above(root, node)
  if node == root return 0 end
  cur = node
  counter = 1
  while cur.parent != root
    counter +=1
    cur = cur.parent
  end
  return counter
end


function compute_center_of_mass!(bt::Tree)
    for n in bt.allnodes
      tgt_points=bt.tgt_points[n.tgt_point_indices]
        isempty(tgt_points) && continue
        avg_pt = zero(tgt_points[1])
        for pt in tgt_points
            avg_pt += pt
        end
        avg_pt ./= length(tgt_points)
        n.center_of_mass = avg_pt
    end
    return bt
end

function print_tree_statistics(bt::Tree)
    tot_far = 0
    tot_near = 0
    tot_leaf_points = 0
    for n in bt.allleaves
        tot_leaf_points += length(n.tgt_point_indices)
        # tot_far += length(n.far_nodes)
        # tot_near += length(n.neighbors)
    end
    # num_neighbors = sum([length(node.neighbors) for node in bt.allleaves])
    # println("Avg neighborhood: ", num_neighbors/length(bt.allleaves))
    vec = [length(node.tgt_point_indices) for node in bt.allleaves]
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
