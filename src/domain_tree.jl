
# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
mutable struct BallNode{PT<:AbstractVector{<:AbstractVector{<:Real}},
                      PIT<:AbstractVector{<:Int},
                      NT<:AbstractVector,
                      OT<:AbstractVector{<:Number},
                      MT<:AbstractMatrix{<:Real},
                      DT,
                      ST<:AbstractMatrix{<:Number},
                      OIT<:AbstractVector{<:AbstractMatrix{<:Number}}}
  is_precond_node::Bool  # is Node whose diag block is inv'd for precond
  # TODO is_precond_node may be unnecessary
  dimension::Int
  points::PT # how to keep this type general? (i.e. subarray) + 1d?
  point_indices::PIT
  near_indices::PIT
  neighbors::NT
  far_nodes::NT
  outgoing::OT  # This is an array of multipole coefficients, created at matvec time
  near_mat::MT
  diag_block::DT
  # Below are source2outgoing and outgoing2incoming mats, created at factor time
  s2o::ST
  o2i::OIT
  left_child
  right_child
  center
  splitter_normal # nothing if leaf
end


# constructor convenience
function BallNode(isprecond, dimension, ctr, points, point_indices = collect(1:length(points)))
  near_indices = zeros(Int, 0)
  neighbors = Vector(undef, 0)
  far_nodes = Vector(undef, 0)
  outgoing = zeros(Complex{Float64}, 0)
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # TODO this forces the DT type in NodeData to be a Cholesky, should it?
  s2o = zeros(Complex{Float64}, 0, 0)
  o2i = fill(s2o, 0)
  BallNode(isprecond, dimension, points, point_indices,
           near_indices, neighbors, far_nodes, outgoing,
           near_mat, diag_block, s2o, o2i, nothing, nothing, ctr, nothing)
end


mutable struct Tree
  dimension::Int64
  root::BallNode
  max_dofs_per_leaf::Int64
  allnodes::Array{BallNode}
  allleaves::Array{BallNode}
end


function isleaf(node)
  return node.splitter_normal == nothing
end


function plane_intersects_sphere(plane_center, splitter_normal,
        sphere_center, sphere_leafrad)
  sphere_right_pole = sphere_center + sphere_leafrad * splitter_normal
  sphere_left_pole = sphere_center - sphere_leafrad * splitter_normal
  return ( dot(sphere_right_pole - plane_center, splitter_normal)
          * dot(sphere_left_pole - plane_center, splitter_normal) < 0)
end

# Compute neighbor lists (note: ONLY FOR LEAVES at this time)
function compute_near_far_nodes!(bt)
  for leaf in bt.allleaves
    leafrad = maximum([norm(pt-leaf.center) for pt in leaf.points])
    cur_node = bt.root
    node_queue = [bt.root]
    while length(node_queue) > 0
      cur_node = pop!(node_queue)
      if isleaf(cur_node)
        if cur_node != leaf
          push!(leaf.neighbors, cur_node)
        end
      else
        intersected = plane_intersects_sphere(cur_node.center,
                        cur_node.splitter_normal, leaf.center, 1.1*leafrad)
        if intersected
          push!(node_queue, cur_node.left_child)
          push!(node_queue, cur_node.right_child)
        else
          if(dot(leaf.center-cur_node.center, cur_node.splitter_normal) > 0)
            push!(node_queue, cur_node.right_child)
            push!(leaf.far_nodes, cur_node.left_child)
          else
            push!(node_queue, cur_node.left_child)
            push!(leaf.far_nodes, cur_node.right_child)
          end
        end
      end
    end
  end
end


function find_farthest(far_pt, pts)
  max_dist = 0
  cur_farthest = far_pt
  for p in pts
    if norm(p-far_pt)>max_dist
      max_dist=norm(p-far_pt)
      cur_farthest = p
    end
  end
  return cur_farthest
end


function rec_split!(bt, node)
  pt_L = find_farthest(node.center, node.points)
  pt_R = find_farthest(pt_L, node.points)
  splitter_normal = pt_R-pt_L
  splitter_normal /= norm(splitter_normal)

  node.splitter_normal = splitter_normal

  T = eltype(node.points)
  left_points = Vector{T}(undef, 0)
  right_points = Vector{T}(undef, 0)
  left_indices = zeros(Int, 0)
  right_indices = zeros(Int, 0)

  for i in eachindex(node.points)
    pt = node.points[i]
    if dot(pt-node.center, splitter_normal) > 0
      push!(right_indices, node.point_indices[i])
      push!(right_points, pt)
    else
      push!(left_indices, node.point_indices[i])
      push!(left_points, pt)
    end
  end

  left_center = sum(left_points)/length(left_points)
  right_center = sum(right_points)/length(right_points)


  left_node = BallNode(false, node.dimension, left_center, left_points, left_indices)
  right_node = BallNode(false, node.dimension, right_center, right_points, right_indices)

  push!(bt.allnodes, left_node)
  push!(bt.allnodes, right_node)
  node.left_child = left_node
  node.right_child = right_node
  if(length(left_points) > bt.max_dofs_per_leaf)
    rec_split!(bt, left_node)
  end
  if(length(right_points)>bt.max_dofs_per_leaf)
    rec_split!(bt, right_node)
  end
end


function initialize_tree(points, max_dofs_per_leaf)
  dimension = length(points[1])
  center = sum(points)/length(points)

  root = BallNode(false, dimension, center, points)
  bt = Tree(dimension, root, max_dofs_per_leaf, [root], [])
  if(length(points) > max_dofs_per_leaf)
    rec_split!(bt, root)
  end

  for node in bt.allnodes
    if isleaf(node)
      push!(bt.allleaves, node)
    end
  end
  compute_near_far_nodes!(bt)

  return bt
end
