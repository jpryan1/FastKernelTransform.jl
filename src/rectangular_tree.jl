# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
# IDEA: use NearestNeighbors.jl and StaticArrays to speed up tree construction
mutable struct BallNode{PT<:AbstractVector{<:AbstractVector{<:Real}},
                      PIT<:AbstractVector{<:Int},
                      NT<:AbstractVector,
                      # OT<:AbstractVector{<:Number},
                      #MT<:AbstractMatrix{<:Real},
                      DT,
                      # ST<:AbstractMatrix{<:Number},
                      # OIT<:AbstractVector{AbstractMatrix{<:Number}},
                      CT}
    is_precond_node::Bool  # is Node whose diag block is inv'd for precond TODO may be unnecessary
    dimension::Int
    tgt_points::PT # how to keep this type general? (i.e. subarray) + 1d?
    tgt_point_indices::PIT
    near_indices::PIT
    src_points::PT
    src_point_indices::PIT
    neighbors::NT
    far_nodes::NT
    far_leaf_points::Int
    outgoing::AbstractVecOrMat{<:Number}  # This is an array of multipole coefficients, created at matvec time
    near_mat::AbstractMatrix # TODO: think about how to handle lazy / dense matrices elegantly
    diag_block::DT
    # Below are source2outgoing and outgoing2incoming mats, created at factor time
    s2o::AbstractMatrix
    o2i::AbstractVector # {AbstractMatrix{<:Number}} # TODO: think about how to handle lazy / dense matrices elegantly
    left_child # can be BallNode or Nothing
    right_child # can be BallNode or Nothing
    parent # can be BallNode or Nothing
    center::CT
    splitter_normal # nothing if leaf
    sidelens
end

# constructor convenience
function BallNode(isprecond::Bool, dimension::Int, ctr::AbstractVector{<:Real},
                  tgt_points::VecOfVec{<:Real}, tgt_point_indices::AbstractVector{<:Int},
                  src_points::VecOfVec{<:Real}, src_point_indices::AbstractVector{<:Int},
                  outgoing_length::Int, sidelens)
  near_indices = zeros(Int, 0)
  neighbors = Vector(undef, 0)
  far_nodes = Vector(undef, 0)
  far_leaf_points = 0
  # T = eltype(tgt_points[1]) # TODO get type someway else
  outgoing = zeros(Complex{Float64}, outgoing_length)
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # TODO this forces the DT type in NodeData to be a Cholesky, should it?
  s2o = zeros(Complex{Float64}, 0, 0)
  o2i = fill(s2o, 0)
  BallNode(isprecond, dimension, tgt_points, tgt_point_indices,
           near_indices, src_points, src_point_indices, neighbors, far_nodes, far_leaf_points, outgoing,
           near_mat, diag_block, s2o, o2i, nothing, nothing, nothing, ctr, nothing, sidelens)
end

# calculates the total number of far points for a given leaf # TODO: deprecate?
function get_tot_far_points(leaf::BallNode)
    isempty(leaf.far_nodes) ? 0 : sum(node->length(node.src_points), leaf.far_nodes)
end

struct Tree{T<:Real, R<:BallNode, V<:AbstractVector{<:BallNode}}
    dimension::Int64
    root::R
    max_dofs_per_leaf::Int64
    allnodes::V
    allleaves::V
    neighbor_scale::T
end

function isleaf(node::BallNode)
    return node.splitter_normal == nothing
end


function plane_intersects_sphere(plane_center, splitter_normal,
        sphere_center, sphere_leafrad)
  sphere_right_pole = sphere_center + sphere_leafrad * splitter_normal
  sphere_left_pole = sphere_center - sphere_leafrad * splitter_normal
  return (dot(sphere_right_pole - plane_center, splitter_normal)
          * dot(sphere_left_pole - plane_center, splitter_normal) < 0)
end


function radius(node)
  return norm(node.sidelens)/2
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
    if length(leaf.tgt_points)==0 continue end
    node_queue = [bt.root]
    while length(node_queue) > 0
      cur_node = pop!(node_queue)
      if cur_node == leaf continue end
      if length(cur_node.src_points)==0 continue end
      # Are they overlapping
      if !isleaf(cur_node) && is_ancestor_of(cur_node, leaf)
        push!(node_queue, cur_node.right_child)
        push!(node_queue, cur_node.left_child)
        continue
      end

      # Is the ratio satisfied?
      min_r_val = minimum((norm(difference(pt, cur_node.center)) for pt in leaf.tgt_points))
      max_rprime_val = maximum((norm(difference(pt, cur_node.center)) for pt in cur_node.src_points)) # pre-compute
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

  left_center = copy(node.center)
  left_center[best_d] -= (node.sidelens[best_d]/4)
  right_center = copy(node.center)
  right_center[best_d] += (node.sidelens[best_d]/4)

  new_sidelens = copy(node.sidelens)
  new_sidelens[best_d] /= 2

  left_node = BallNode(false, node.dimension, left_center, left_tgt_points,
              left_tgt_indices, left_src_points, left_src_indices, length(node.outgoing),new_sidelens)
  right_node = BallNode(false, node.dimension, right_center, right_tgt_points,
              right_tgt_indices, right_src_points, right_src_indices, length(node.outgoing), new_sidelens)

    left_node.parent = node
    right_node.parent = node # IDEA: recurse before constructing node?
    push!(bt.allnodes, left_node)
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


function initialize_tree(tgt_points, src_points, max_dofs_per_leaf, outgoing_length::Int,
                    neighbor_scale::Real = 0.75)

  dimension = isempty(tgt_points) ? src_points : length(tgt_points[1])
  center = sum(vcat(tgt_points, src_points)) / (length(tgt_points) + length(src_points))
  root_sidelen = maximum((maximum(abs, difference(pt, center)) for pt in vcat(tgt_points, src_points)))
  root = BallNode(false, dimension, center, tgt_points, collect(1:length(tgt_points)),
    src_points, collect(1:length(src_points)), outgoing_length, [2.01*root_sidelen for i in 1:dimension])

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
  num_neighbors = sum([length(node.neighbors) for node in bt.allleaves])
  println("Avg neighborhood: ", num_neighbors/length(bt.allleaves))
  tot_far = 0
  tot_near = 0
  tot_leaf_points = 0
  for n in bt.allleaves
    tot_leaf_points += length(n.tgt_points)
    tot_far += length(n.far_nodes)
    tot_near += length(n.neighbors)
  end

  vec = [length(node.tgt_points) for node in bt.allleaves]
  println("Num leaves ", length(bt.allleaves))
  println("Num nodes ", length(bt.allnodes))
  println("Avg far ", tot_far/length(bt.allleaves))
  println("Mean points ", mean(vec))
  println("Median points ", median(vec))
  println("Minimum points ", minimum(vec))
  println("Maximum points ", maximum(vec))
  println("Avg leaf_points ", tot_leaf_points/length(bt.allleaves))
  return bt
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
