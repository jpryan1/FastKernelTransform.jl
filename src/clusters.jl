# This struct stores domain tree nodes' data, including anything that makes
# matvec'ing faster due to precomputation and storage at factor time
mutable struct Cluster{PT<:AbstractVector{<:AbstractVector{<:Real}},
                      PIT<:AbstractVector{<:Int},
                      NT<:AbstractVector,
                      OT<:AbstractVector{<:Number},
                      DT,
                      CT}
  is_precond_node::Bool  # is Node whose diag block is inv'd for precond
  # TODO is_precond_node may be unnecessary
  dimension::Int
  tgt_points::PT # how to keep this type general? (i.e. subarray) + 1d?
  tgt_point_indices::PIT
  near_indices::PIT
  src_points::PT
  src_point_indices::PIT
  neighbors::NT
  far_nodes::NT
  outgoing::OT  # This is an array of multipole coefficients, created at matvec time
  near_mat::AbstractMatrix
  diag_block::DT
  # Below are source2outgoing and outgoing2incoming mats, created at factor time
  s2o::AbstractMatrix
  o2i::AbstractVector
  center::CT
  radius
end

# calculates the total number of far points for a given leaf
function get_tot_far_points(leaf)
    isempty(leaf.far_nodes) ? 0 : sum(node->length(node.src_points), leaf.far_nodes)
end

# constructor convenience
function Cluster(isprecond::Bool, dimension::Int, points, center::AbstractVector{<:Real},
      outgoing_length::Int, point_indices = collect(1:length(points)))
  near_indices = zeros(Int, 0)
  neighbors = Vector(undef, 0)
  far_nodes = Vector(undef, 0)
  outgoing = zeros(Complex{Float64}, outgoing_length)
  near_mat = zeros(0, 0)
  diag_block = cholesky(zeros(0, 0), Val(true), check = false) # TODO this forces the DT type in NodeData to be a Cholesky, should it?
  s2o = zeros(Complex{Float64}, 0, 0)
  o2i = fill(s2o, 0)
  radius = 0
  for pt in points
    radius = max(radius, norm(center - pt))
  end
  Cluster(isprecond, dimension, points, point_indices,
           near_indices, points, point_indices, neighbors, far_nodes, outgoing,
           near_mat, diag_block, s2o, o2i, center, radius)
end

mutable struct ClusterDecomp
  allleaves::Vector{Cluster}
  dimension::Int64
  NEIGHBORLY_PARAM
  IS_ZERO_DIST
end


function is_neighbor_to(A::Cluster, B::Cluster, decomp, tol::Real = 1e-9)
  recentered_B = [pt-A.center for pt in B.tgt_points]
  ratio = maximum([norm(pt) for pt in A.tgt_points])/minimum([norm(pt) for pt in recentered_B])
  return ratio > 0.6
end


function is_farnode_to( A::Cluster, B::Cluster, decomp, tol::Real = 1e-9)
  rad = A.center - B.center
  return (!is_neighbor_to(A,B,decomp)) && norm(rad)+A.radius+B.radius < decomp.IS_ZERO_DIST
end

function are_really_far(decomp, A::Cluster, B::Cluster, tol::Real = 1e-9)
  rad = A.center - B.center
  return norm(rad)+A.radius+B.radius > decomp.IS_ZERO_DIST
end

# Compute neighbor lists (note: ONLY FOR LEAVES at this time)
function compute_near_far!(decomp)
  for i in 1:length(decomp.allleaves)
    cluster_a = decomp.allleaves[i]
    for j in 1:length(decomp.allleaves)
      if i==j continue end
      cluster_b = decomp.allleaves[j]
      if(is_neighbor_to(cluster_a, cluster_b, decomp))
        push!(cluster_b.neighbors, cluster_a)
      elseif(is_farnode_to(cluster_a, cluster_b, decomp))
        push!(cluster_b.far_nodes, cluster_a)
      end
    end
  end
end



function initialize_cluster_decomp(points, dofs_per_leaf, outgoing_length)
  dimension = length(points[1])
  # Get limits of root node for tree.
  clusters = []
  cluster_centers = [points[rand(1:length(points))]]
  # First create set of centers
  max_num_clusters = length(points) / dofs_per_leaf  # this can be made smarter
  while length(cluster_centers) < max_num_clusters
    # Get farthest point from cluster centers
    dist_to_farthest_point = 0
    farthest_point = points[1]
    cluster_counts = zeros(Int, length(cluster_centers))
    for point in points
      closest_dist, idx = findmin([norm(difference(point, center)) for center in cluster_centers]) # difference?
      cluster_counts[idx] += 1
      if closest_dist > dist_to_farthest_point
        farthest_point = point
        dist_to_farthest_point = closest_dist
      end
    end
    push!(cluster_centers, farthest_point)
  end
  # Once centers are created, go through all points and assign to cluster
  # with closest center
  center_to_pt_indices = Dict()
  center_to_pts = Dict()
  for pt_idx in 1:length(points) # parallel and / or smarter?
    pt = points[pt_idx]
    closest_center = cluster_centers[1]
    min_dist = norm(difference(pt, closest_center))
    for center in cluster_centers
      d = norm(difference(center, pt))
      if d < min_dist
        min_dist = d
        closest_center = center
      end
    end
    if !haskey(center_to_pt_indices, closest_center)
      center_to_pt_indices[closest_center] = [pt_idx]
      center_to_pts[closest_center] = [pt]
    else
      push!(center_to_pt_indices[closest_center], pt_idx)
      push!(center_to_pts[closest_center],  pt)
    end
  end
  for center in cluster_centers
    push!(clusters, Cluster(false, dimension, center_to_pts[center], center, outgoing_length, center_to_pt_indices[center]))
  end
  decomp = ClusterDecomp(clusters, dimension, 1.5, 10)
  compute_near_far!(decomp)

  min_rad = minimum([cl.radius for cl in clusters])
  max_rad = maximum([cl.radius for cl in clusters])
  min_pts = minimum([length(cl.tgt_points) for cl in clusters])
  max_pts = maximum([length(cl.tgt_points) for cl in clusters])
  println("Cluster radii ", sort([cl.radius for cl in clusters]))
  println("Cluster num_points ", sort([length(cl.tgt_points) for cl in clusters]))
  println("Cluster neighbords ", sort([length(cl.neighbors) for cl in clusters]))
  println("Cluster far ", sort([length(cl.far_nodes) for cl in clusters]))
  println("Cluster src ", sort([length(cl.src_points) for cl in clusters]))
  really_far_counter = 0
  for cl_adx in 1:length(clusters)
    cl_a = clusters[cl_adx]
    for cl_bdx in (cl_adx+1):length(clusters)
      cl_b = clusters[cl_bdx]
      if are_really_far(decomp, cl_a, cl_b)
        really_far_counter += 1
      end
    end
  end
  println("Of the ", binomial(length(clusters), 2), " pairs, ", really_far_counter , " are really far apart")
  return decomp
end

# N=10000
# dimension = 2
# rad_thresh = 0.1
#
# decomp = initialize_cluster_decomp([rand(dimension) for i in 1:N], rad_thresh)
# xs = []
# ys = []
# colors = []
# sizes = []
# for i in 1:length(decomp.allleaves)
#   cluster=decomp.allleaves[i]
#   c_points = cluster.tgt_points
#   for pt in c_points
#     if pt == cluster.center
#       continue
#     else
#       push!(colors, i)
#       push!(sizes, 4)
#     end
#     push!(xs, pt[1])
#     push!(ys, pt[2])
#   end
# end
#
# # for i in 1:length(decomp.allleaves)
# #   cluster=decomp.allleaves[i]
# #   pt = cluster.center
# #     push!(colors, 0)
# #     push!(sizes, 10)
# #     push!(xs, pt[1])
# #     push!(ys, pt[2])
# # end
# for neighbor in decomp.allleaves[1].neighbors
#   pt = neighbor.center
#      push!(colors, 0)
#      push!(sizes, 10)
#      push!(xs, pt[1])
#      push!(ys, pt[2])
# end
#
# push!(colors, 1)
# push!(sizes, 13)
# push!(xs, decomp.allleaves[1].center[1])
# push!(ys, decomp.allleaves[1].center[2])
# scatter(xs, ys, color=colors, markersize=sizes)
#
# gui()
# end
