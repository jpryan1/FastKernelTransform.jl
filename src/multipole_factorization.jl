# multi_to_single: helper array, converting from (k, h, i) representation of
# multipole coefficients to single indices into an array (for efficient
# matrix vector products)
# TODO: add element type to Factorization
struct MultipoleFactorization{T, K, PAR <: FactorizationParameters,
                MST, NT, TT<:Tree, TP, SP, FT, GT, RT<:AbstractVector{Int}, VT} <: Factorization{T}
    kernel::K
    params::PAR

    multi_to_single::MST
    normalizer_table::NT
    tree::TT
    tgt_points::TP
    src_points::SP

    get_F::FT
    get_G::GT
    radial_fun_ranks::RT

    variance::VT # additive diagonal correction
    symmetric::Bool
    _k::T # sample output of kernel, only here to determine element type
end

# if only target points are passed, convert to src_points
function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real},
                    params::FactorizationParameters = FactorizationParameters())
    variance = nothing
    MultipoleFactorization(kernel, tgt_points, tgt_points, variance, params)
end

function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real},
        variance, params::FactorizationParameters = FactorizationParameters())
    MultipoleFactorization(kernel, tgt_points, tgt_points, variance, params)
end

function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                variance = nothing, params::FactorizationParameters = FactorizationParameters())
    dimension = length(tgt_points[1])
    get_F, get_G, radial_fun_ranks = init_F_G(kernel, dimension, params.trunc_param, Val(qrable(kernel)))
    MultipoleFactorization(kernel, tgt_points, src_points, get_F, get_G, radial_fun_ranks, variance, params)
end

# takes arbitrary isotropic kernel
# IDEA: convert points to vector of static arrays, dispatch 2d (and 1d) implementation on its type!
# IDEA: given radial_fun_ranks, can we get rid of trunc_param?
function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                get_F, get_G, radial_fun_ranks::AbstractVector,
                                variance = nothing, params = FactorizationParameters())
    (params.max_dofs_per_leaf ≤ params.precond_param || (params.precond_param == 0)) || throw(DomainError("max_dofs_per_leaf < precond_param"))
    dimension = length(tgt_points[1])
    multi_to_single = get_index_mapping_table(dimension, params.trunc_param, radial_fun_ranks)
    normalizer_table = squared_hyper_normalizer_table(dimension, params.trunc_param)
    outgoing_length = length(keys(multi_to_single))

    @time tree = initialize_tree(tgt_points, src_points, params.max_dofs_per_leaf,
                        params.neighbor_scale, barnes_hut = params.barnes_hut,
                        verbose = params.verbose, lazy = params.lazy)

    _k = kernel(tgt_points[1], src_points[1]) # sample evaluation used to determine element type
    symmetric = tgt_points === src_points

    fact = MultipoleFactorization(kernel, params, multi_to_single,
                    normalizer_table, tree, tgt_points, src_points,
                    get_F, get_G, radial_fun_ranks, variance, symmetric, _k)
    compute_transformation_mats!(fact)
    if symmetric && params.precond_param > 0
        compute_preconditioner!(fact, params.precond_param, variance)
    end
    return fact
end

############################ basic properties ##################################
Base.size(F::MultipoleFactorization) = (length(F.tgt_points), length(F.src_points))
Base.size(F::MultipoleFactorization, i::Int) = i > 2 ? 1 : size(F)[i]
islazy(F::MultipoleFactorization) = F.params.lazy # typeof(F.params.lazy) == Val{true}
isbarneshut(F::MultipoleFactorization) = F.params.barnes_hut
LinearAlgebra.issymmetric(F::MultipoleFactorization) = F.symmetric
Base.eltype(F::MultipoleFactorization{T}) where T = T
function Base.getindex(F::MultipoleFactorization, i::Int, j::Int)
    F.kernel(F.tgt_points[i], F.src_points[j])
end
# number of multipoles of factorization
nmultipoles(fact::MultipoleFactorization) = length(keys(fact.multi_to_single))

###################### transformation matrices #################################
function compute_transformation_mats!(fact::MultipoleFactorization)

    # For every node in tree, get radius of hypersphere as rprime, look for
    # points r such that rprime/r is not bad, and, if compression efficient,
    # make s2o and (single) o2i matrix for node.

    # @sync for leaf in fact.tree.allleaves
    #     if !isempty(leaf.tgt_points)
    #         @spawn transformation_mats!(fact, leaf)
    #     end
    # end
    for node in fact.tree.allnodes
        if isempty(node.tgt_point_indices) continue end
        tgt_points = fact.tgt_points[node.tgt_point_indices]
        if isleaf(node)
            src_points = fact.src_points[node.near_point_indices]
            node.near_mat = compute_interactions(fact, src_points, tgt_points) # near field interactions
            if issymmetric(fact) # if target and source are equal, need to apply diagonal correction
                node.near_mat = diagonal_correction!(node.near_mat, fact.variance, node.tgt_point_indices)
            end
        end
        if compression_is_efficient(fact, node)
            compute_compressed_interactions!(fact, node)
        else
            far_points = fact.tgt_points[node.far_point_indices]
            if length(far_points) > 0
                src_points = fact.src_points[node.src_point_indices]
                node.o2i = compute_interactions(fact, far_points, src_points)
            end
        end
    end
    return nothing
end

# better name? compute_transformation_matrices
# function transformation_mats!(F::MultipoleFactorization, leaf)
#     begin # IDEA: parallelize?
#         src_points = source_neighborhood_points(leaf) # WARNING: BOTTLENECK
#         tgt_points = target_points(leaf)
#         leaf.near_mat = compute_interactions(F, tgt_points, src_points) # near field interactions
#     end
#
#     source_neighborhood_indices!(leaf) # stores the indices of the source and neighborhood
#     if issymmetric(F) # if target and source are equal, need to apply diagonal correction
#         leaf.near_mat = diagonal_correction!(leaf.near_mat, F.variance, leaf.tgt_point_indices)
#     end
#
#     T = transformation_eltype(F)
#     M = islazy(F) ? LazyMultipoleMatrix{T} : Matrix{T}
#     leaf.o2i = Vector{M}(undef, length(leaf.far_nodes))
#     for far_node_idx in eachindex(leaf.far_nodes) # IDEA: parallelize?
#         compute_far_interactions!(F, leaf, far_node_idx)
#     end
#     return F
# end

function transformation_eltype(F::MultipoleFactorization)
    T = eltype(F)
    T = T <: Real ? Complex{T} : T
end

# computes interaction matrix, and stores it in near_mat, either lazily or densely
function compute_interactions(F::MultipoleFactorization, tgt_points, src_points, T::Type = transformation_eltype(F))
    G = gramian(F.kernel, tgt_points, src_points)
    return islazy(F) ? LazyMultipoleMatrix{T}(()->G, size(G)...) : Matrix(G)
end

# function compute_far_interactions!(F::MultipoleFactorization, leaf, far_node_idx)
#     far_node = leaf.far_nodes[far_node_idx]
#     if isempty(far_node.src_points) return end
#     if compression_is_efficient(F, far_node)
#         compute_compressed_interactions!(F, leaf, far_node_idx)
#     else
#         leaf.o2i[far_node_idx] = compute_interactions(F, leaf.tgt_points, far_node.src_points)
#     end
#     return nothing
# end

# computes whether or not compressing the far interaction is more efficient than a direct multiply
function compression_is_efficient(F::MultipoleFactorization, node)
    # return false
    far_points = length(node.far_point_indices)
    src_pts = length(node.src_point_indices)
    (nmultipoles(F) * (src_pts + far_points)) < (src_pts * far_points)
end

function compute_compressed_interactions!(F::MultipoleFactorization, node)
    center_point = get_center_point(F, node)
    center(x) = difference(x, center_point)
    begin
        # source to outgoing matrix
        if isempty(node.s2o)
            src_points = F.src_points[node.src_point_indices]
            recentered_src = center.(src_points) # WARNING: BOTTLENECK, move out?
            node.s2o = source2outgoing(F, recentered_src)
        end
        # outgoing to incoming matrix
        begin
            far_points = F.tgt_points[node.far_point_indices]
            recentered_tgt = center.(far_points) # WARNING: BOTTLENECK
            node.o2i = outgoing2incoming(F, recentered_tgt)
        end
    end
    return nothing
end

# returns center of box or center of mass of points in box for Barnes Hut
function get_center_point(F::MultipoleFactorization, node::BallNode)
    isbarneshut(F) ? node.center_of_mass : node.center
end

################################## helper #######################################
function get_index_mapping_table(dimension::Int, trunc_param::Int, radial_fun_ranks::AbstractVector{Int})
    counter = 0
    multi_to_single = Dict() # TODO: types or array
    for k in 0:trunc_param
        multiindices = get_multiindices(dimension, k)
        r = radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        for i in k:2:max_i
            for h in multiindices
                counter += 1
                multi_to_single[(k, h, i)] = counter
            end
        end
    end
    return multi_to_single
end
