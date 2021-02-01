import RegionTrees: AbstractRefinery, needs_refinement, refine_data
import RegionTrees.Cell

# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
mutable struct NodeData{PT<:AbstractVector{<:AbstractVector{<:Real}},
                      PIT<:AbstractVector{<:Int},
                      NT<:AbstractVector{<:Cell},
                      OT<:AbstractVector{<:Number},
                      MT<:AbstractMatrix{<:Real},
                      DT,
                      ST<:AbstractMatrix{<:Number},
                      OIT<:AbstractVector{<:AbstractMatrix{<:Number}}}
  is_precond_node::Bool  # is Node whose diag block is inv'd for precond
  # TODO is_precond_node may be unnecessary
  dimension::Int
  level::Int
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
end

# constructor convenience
function NodeData(isprecond, dimension, level, points, point_indices = collect(1:length(points)))
  near_indices = zeros(Int, 0)
  neighbors = Vector{Cell}(undef, 0)
  far_nodes = Vector{Cell}(undef, 0)
  outgoing = zeros(Complex{Float64}, 0)
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # TODO this forces the DT type in NodeData to be a Cholesky, should it?
  s2o = zeros(Complex{Float64}, 0, 0)
  o2i = fill(s2o, 0)
  NodeData(isprecond, dimension, level, points, point_indices,
           near_indices, neighbors, far_nodes, outgoing,
           near_mat, diag_block, s2o, o2i)
end

mutable struct TreeLevel
  level::Int
  nodes::Vector{Cell}
end

mutable struct Tree
  dimension::Int64
  root::Cell
  levels::Vector{TreeLevel}
end

# Check if A (a leaf) is touching B via checking infinity norm
function are_adjacent(A::Cell, B::Cell, tol::Real = 1e-9)
  rad = RegionTrees.center(A) - RegionTrees.center(B)
  inf_norm = maximum(abs, rad)
  should_be = maximum(A.boundary.widths) / 2 + maximum(B.boundary.widths) / 2
  return abs(inf_norm - should_be) < tol
end

# Check if cella (a leaf) is far from cellb via checking infinity norm
function are_far(A::Cell, B::Cell, tol::Real = 1e-9)
  rad = RegionTrees.center(A) - RegionTrees.center(B)
  inf_norm = maximum(abs, rad)
  bubble = maximum(A.boundary.widths) * 3/2 + maximum(B.boundary.widths) / 2
  return inf_norm >= bubble - tol
end

# Compute neighbor lists (note: ONLY FOR LEAVES at this time)
function compute_neighbors!(qt)
  leaves = allleaves(qt.root)
  for leaf in leaves
    for level_node in qt.levels[leaf.data.level].nodes
      if are_adjacent(leaf, level_node)
        push!(leaf.data.neighbors, level_node)
      end
    end
    for level_idx in (leaf.data.level-1):-1:2
      for level_node in qt.levels[level_idx].nodes
        if isleaf(level_node) && are_adjacent(leaf, level_node)
          push!(leaf.data.neighbors, level_node)
        end
      end
    end
  end
end

# Compute neighbor lists (note: ONLY FOR LEAVES at this time)
# Note the queue implementation - far node list is partition of faraway nodes
# into as small a list as possible, ex: if node a and node a's child are far,
# the far list only contains node a
function compute_far_nodes!(qt)
  leaves = allleaves(qt.root)
  for leaf in leaves
    node_queue = [qt.root]
    while !isempty(node_queue)
      node = pop!(node_queue)
      if are_far(leaf, node)
        push!(leaf.data.far_nodes, node)
      elseif !isleaf(node) && node.data.level < leaf.data.level
          append!(node_queue, children(node))
      end
    end
  end
end


# Tree refinement
struct DomainTreeRefinery <: AbstractRefinery
  MAX_POINTS_IN_LEAF::Int64
end

function needs_refinement(r::DomainTreeRefinery, cell::Cell)
  return length(cell.data.points) > r.MAX_POINTS_IN_LEAF
end

function refine_data(r::DomainTreeRefinery, cell::Cell, indices)
  points = cell.data.points
  child_points = Vector{Vector{Float64}}(undef, 0)
  child_point_indices = zeros(Int, 0)
  boundary = child_boundary(cell, indices)
  c = RegionTrees.center(boundary)[:]
  # Check inf norm dist of all points to child center
  for i in eachindex(points)
    pt = points[i]
    rad = pt-c
    inf_norm = maximum(abs, rad)
    if inf_norm < maximum(boundary.widths) / 2
      child_pt_idx = cell.data.point_indices[i]
      push!(child_point_indices, child_pt_idx)
      push!(child_points, points[i])
    end
  end
  NodeData(false, cell.data.dimension, cell.data.level+1, child_points, child_point_indices)
end

# Helper function for plotting 3D
function plot_box(plt, v)
  pt_order = [1,2,4,3,1,5,6,2,6,8,4,8,7,3,7,5]
  plot3d!(plt, v[1,pt_order], v[2,pt_order], v[3,pt_order])
end


function initialize_tree(points, max_dofs_per_leaf)
  dimension = length(points[1])
  # Get limits of root node for tree.
  delta = 1e-2
  point_min = minimum(minimum, points) - delta
  point_max = maximum(maximum, points) + delta
  root_data = NodeData(false, dimension, 1, points)

  crnr = [point_min for i in 1:dimension]
  width = [point_max-point_min for i in 1:dimension]
  # Create root node
  boundary = RegionTrees.HyperRectangle(SVector{dimension}(crnr),
    SVector{dimension}(width))
  # Refine tree using RegionTrees adaptive sampling
  refinery = DomainTreeRefinery(max_dofs_per_leaf)
  root = RegionTrees.Cell(boundary, root_data)
  adaptivesampling!(root, refinery)

  # Populate list of levels, each level containing nodes at that level
  levels = []
  node_queue = [root]
  while !isempty(node_queue)
    node = pop!(node_queue)
    if node.data.level > length(levels)
      nodes = Vector{Cell}(undef, 0)
      push!(levels, TreeLevel(node.data.level, nodes))
    end
    push!(levels[node.data.level].nodes, node)
    if !isleaf(node)
        append!(node_queue, children(node))
    end
  end

  qt = Tree(dimension, root, levels)
  compute_neighbors!(qt)
  compute_far_nodes!(qt)
  # plt = plot3d(legend=nothing)
  return qt


  # total_num = 0
  # for level in qt.levels
  #   for node in level.nodes
  #     total_num += length(node.data.points)/dimension
  #   end
  #   println("num level ",total_num)
  #   total_num = 0
  # end
  # for leaf in allleaves(root)
  #   v = hcat(collect(vertices(leaf.boundary))...)
  #   plot_box(plt, v)
    # total_num = length(leaf.data.points)/dimension
    # for neighbor in leaf.data.neighbors
    #   total_num += length(neighbor.data.points)/dimension
    # end
    # for far in leaf.data.far_nodes
    #   total_num += length(far.data.points)/dimension
    # end
  # end
  # gui()
end
