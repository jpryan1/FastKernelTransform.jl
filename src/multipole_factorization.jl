# multi_to_single: helper array, converting from (k, h, i) representation of
# multipole coefficients to single indices into an array (for efficient
# matrix vector products)
# TODO: add element type to Factorization
struct MultipoleFactorization{T, K, PAR <: FactorizationParameters,
                MST, NT, TT<:Tree, FT, GT, RT<:AbstractVector{Int}, VT} <: Factorization{T}
    kernel::K
    params::PAR

    multi_to_single::MST
    normalizer_table::NT
    tree::TT

    get_F::FT
    get_G::GT
    radial_fun_ranks::RT

    variance::VT # additive diagonal correction
    symmetric::Bool
    to::TimerOutput
    _k::T # sample output of kernel, only here to determine element type
end

# if only target points are passed, convert to src_points
function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real},
                    params::FactorizationParameters = FactorizationParameters(), to::TimerOutput= TimerOutput())
    variance = nothing
    MultipoleFactorization(kernel, tgt_points, tgt_points, variance, params, to)
end

function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real},
        variance, params::FactorizationParameters = FactorizationParameters(), to::TimerOutput= TimerOutput())
    MultipoleFactorization(kernel, tgt_points, tgt_points, variance, params, to)
end

function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                variance = nothing, params::FactorizationParameters = FactorizationParameters(), to::TimerOutput= TimerOutput())
    dimension = length(tgt_points[1])
    get_F, get_G, radial_fun_ranks = init_F_G(kernel, dimension, params.trunc_param, Val(qrable(kernel)))
    MultipoleFactorization(kernel, tgt_points, src_points, get_F, get_G, radial_fun_ranks, variance, params, to)
end

# takes arbitrary isotropic kernel
# IDEA: convert points to vector of static arrays, dispatch 2d (and 1d) implementation on its type!
# IDEA: given radial_fun_ranks, can we get rid of trunc_param?
function MultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                get_F, get_G, radial_fun_ranks::AbstractVector,
                                variance = nothing, params = FactorizationParameters(), to::TimerOutput= TimerOutput())
    (params.max_dofs_per_leaf ≤ params.precond_param || (params.precond_param == 0)) || throw(DomainError("max_dofs_per_leaf < precond_param"))
    dimension = length(tgt_points[1])
    multi_to_single = get_index_mapping_table(dimension, params.trunc_param, radial_fun_ranks)
    normalizer_table = squared_hyper_normalizer_table(dimension, params.trunc_param)
    outgoing_length = length(keys(multi_to_single))
    tree = initialize_tree(tgt_points, src_points, params.max_dofs_per_leaf,
                        params.neighbor_scale, barnes_hut = params.barnes_hut,
                        verbose = params.verbose, lazy = params.lazy)

    _k = kernel(tgt_points[1], src_points[1]) # sample evaluation used to determine element type
    symmetric = tgt_points === src_points

    fact = MultipoleFactorization(kernel, params, multi_to_single,
                    normalizer_table, tree,
                    get_F, get_G, radial_fun_ranks, variance, symmetric, to, _k)
    compute_transformation_mats!(fact)
    if symmetric && params.precond_param > 0
        compute_preconditioner!(fact, params.precond_param, variance)
    end
    return fact
end

############################ basic properties ##################################
Base.size(F::MultipoleFactorization) = (length(F.tree.tgt_points), length(F.tree.src_points))
Base.size(F::MultipoleFactorization, i::Int) = i > 2 ? 1 : size(F)[i]
islazy(F::MultipoleFactorization) = F.params.lazy # typeof(F.params.lazy) == Val{true}
isbarneshut(F::MultipoleFactorization) = F.params.barnes_hut
LinearAlgebra.issymmetric(F::MultipoleFactorization) = F.symmetric
Base.eltype(F::MultipoleFactorization{T}) where T = T
function Base.getindex(F::MultipoleFactorization, i::Int, j::Int)
    F.kernel(F.tree.tgt_points[i], F.tree.src_points[j])
end
# number of multipoles of factorization
nmultipoles(fact::MultipoleFactorization) = length(keys(fact.multi_to_single))

###################### transformation matrices #################################
function compute_transformation_mats!(fact::MultipoleFactorization)

    # For every node in tree, get radius of hypersphere as rprime, look for
    # points r such that rprime/r is not bad, and, if compression efficient,
    # make s2o and o2i matrix for node.

    @sync for node in fact.tree.allnodes
        if isempty(node.src_point_indices) continue end
        @spawn begin
        src_points = get_source_points(fact.tree, node)
        if isleaf(node) && !isempty(node.near_point_indices)
            near_points = get_near_points(fact.tree, node) # node.tgt_points[node.near_point_indices]
            node.near_mat = compute_interactions(fact, near_points, src_points) # near field interactions
            if issymmetric(fact) # if target and source are equal, need to apply diagonal correction
                node.near_mat = diagonal_correction!(node.near_mat, fact.variance, node.tgt_point_indices)
            end
        end
        if compression_is_efficient(fact, node)
            compute_compressed_interactions!(fact, node)
        else
            if !isempty(node.far_point_indices)
                far_points = get_far_points(fact.tree, node)
                node.o2i = compute_interactions(fact, far_points, src_points)
            end
        end
        end
    end
    return nothing
end

function transformation_eltype(F::MultipoleFactorization)
    T = eltype(F)
    T = T <: Real ? Complex{T} : T
end

# computes interaction matrix, and stores it in near_mat, either lazily or densely
function compute_interactions(F::MultipoleFactorization, tgt_points, src_points, T::Type = transformation_eltype(F))
    G = gramian(F.kernel, tgt_points, src_points)
    islazy(F) ? G : Matrix(G)
end

# computes whether or not compressing the far interaction is more efficient than a direct multiply
function compression_is_efficient(F::MultipoleFactorization, node)
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
            src_points = get_source_points(F.tree, node)
            recentered_src = center.(src_points) # WARNING: BOTTLENECK, move out?
            node.s2o = source2outgoing(F, recentered_src)
        end
        # outgoing to incoming matrix
        begin
            far_points = get_far_points(F.tree, node)
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
