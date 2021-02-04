mutable struct FmmMatrix{K, V<:AbstractVector{<:AbstractVector{<:Real}}, VT} # IDEA: use CovarianceFunctions.Gramian
    kernel::K
    tgt_points::V
    src_points::V
    max_dofs_per_leaf::Int64
    precond_param::Int64
    trunc_param::Int64
    to::TimerOutput
    variance::VT
end


function FmmMatrix(kernel, tgt_points::VecOfVec{<:Real}, src_points::VecOfVec{<:Real},
                   max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int,
                   to::TimerOutput = TimerOutput())
    variance = nothing
    return FmmMatrix(kernel, tgt_points, src_points, max_dofs_per_leaf, precond_param,
                     trunc_param, to, variance)
end

function FmmMatrix(kernel, points::VecOfVec{<:Real},
                   max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int,
                   to::TimerOutput = TimerOutput(), variance = nothing)
    return FmmMatrix(kernel, points, points, max_dofs_per_leaf, precond_param,
                     trunc_param, to, variance)
end

# fast kernel transform
function fkt(mat::FmmMatrix)
    return MultipoleFactorization(mat.kernel, mat.tgt_points, mat.src_points,
        mat.max_dofs_per_leaf, mat.precond_param, mat.trunc_param, mat.to, mat.variance)
end

# factorize only calls fkt if it is worth it
function LinearAlgebra.factorize(mat::FmmMatrix)
    if max(length(mat.tgt_points),length(mat.src_points)) < mat.max_dofs_per_leaf
        x = mat.src_points
        return factorize(k.(mat.tgt_points, permutedims(mat.src_points)))
    else
        return fkt(mat)
    end
end

# multi_to_single: helper array, converting from (k, h, i) representation of
# multipole coefficients to single indices into an array (for efficient
# matrix vector products)
struct MultipoleFactorization{K, TO<:TimerOutput, MST, NT, TT<:Tree, FT, GT, RT<:AbstractVector{Int}, VT}
    kernel::K
    trunc_param::Int64
    to::TO

    multi_to_single::MST
    normalizer_table::NT
    tree::TT
    n_tgt_points::Int
    n_src_points::Int

    get_F::FT
    get_G::GT
    radial_fun_ranks::RT

    variance::VT # additive diagonal correction
    symmetric::Bool
end
LinearAlgebra.issymmetric(F::MultipoleFactorization) = F.symmetric

# if only target points are passed, convert to src_points
function MultipoleFactorization(kernel, tgt_points::VecOfVec{<:Real},
                                max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int,
                                to::TimerOutput = TimerOutput(), variance = nothing)
    MultipoleFactorization(kernel, tgt_points, tgt_points, max_dofs_per_leaf,
                           precond_param, trunc_param, to, variance)
end

function MultipoleFactorization(kernel, tgt_points::VecOfVec{<:Real}, src_points::VecOfVec{<:Real},
                                max_dofs_per_leaf::Int, precond_param::Int, trunc_param::Int,
                                to::TimerOutput = TimerOutput(), variance = nothing)
    dimension = length(tgt_points[1])
    @timeit to "computing F and G" get_F, get_G, radial_fun_ranks = init_F_G(kernel, dimension, trunc_param, Val(qrable(kernel)))
    MultipoleFactorization(kernel, tgt_points, src_points, max_dofs_per_leaf, precond_param,
                           trunc_param, get_F, get_G, radial_fun_ranks, to, variance)
end

# takes arbitrary isotropic kernel
# IDEA: convert points to vector of static arrays, dispatch 2d (and 1d) implementation on its type!
# IDEA: given radial_fun_ranks, can we get rid of trunc_param?
function MultipoleFactorization(kernel, tgt_points::VecOfVec{<:Real}, src_points::VecOfVec{<:Real},
                                max_dofs_per_leaf::Int, precond_param::Int,
                                trunc_param::Int, get_F, get_G, radial_fun_ranks::AbstractVector,
                                to::TimerOutput = TimerOutput(), variance = nothing)
    max_dofs_per_leaf ≤ precond_param || throw(DomainError("max_dofs_per_leaf < precond_param"))
    n_tgt_points = length(tgt_points)
    n_src_points = length(src_points)
    dimension = length(tgt_points[1])
    multi_to_single = get_index_mapping_table(dimension, trunc_param, radial_fun_ranks)
    @timeit to "Populate normalizer table" normalizer_table = squared_hyper_normalizer_table(dimension, trunc_param)
    outgoing_length = length(keys(multi_to_single))
    @timeit to "Initialize tree" tree = initialize_tree(tgt_points, src_points, max_dofs_per_leaf, outgoing_length)

    symmetric = tgt_points === src_points
    fact = MultipoleFactorization(kernel, trunc_param, to,
                                multi_to_single, normalizer_table, tree, n_tgt_points, n_src_points,
                                get_F, get_G, radial_fun_ranks, variance, symmetric)
    @timeit fact.to "Populate transformation table" compute_transformation_mats!(fact)
    if tgt_points === src_points && precond_param > 0
        @timeit fact.to "Get diag inv for precond" compute_preconditioner!(fact, precond_param, variance)
    end
    return fact
end

Base.size(F::MultipoleFactorization) = (F.n_tgt_points, F.n_src_points)
Base.size(F::MultipoleFactorization, i::Int) = i > 2 ? 1 : (i==1 ? F.n_tgt_points : F.n_src_points)

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

# For preconditioner
function compute_preconditioner!(fact::MultipoleFactorization, precond_param::Int,
                                 variance::Union{AbstractVector, Nothing} = fact.variance)
    node_queue = [fact.tree.root]
    @sync while !isempty(node_queue) # IDEA: parallelize
        node = pop!(node_queue)
        if length(node.tgt_point_indices) ≤ precond_param
            @spawn begin
                tgt_points = node.tgt_points
                K = fact.kernel.(tgt_points, permutedims(tgt_points)) # IDEA: can the kernel matrix be extracted from node.near_mat?
                diagonal_correction!(K, variance, node.tgt_point_indices)
                node.diag_block = cholesky!(K, Val(true), tol = 1e-7, check = false) # in-place
            end
        else
            push!(node_queue, node.left_child)
            push!(node_queue, node.right_child)
        end
    end
    return fact
 end

using LinearAlgebra: checksquare

diagonal_correction!(K::AbstractMatrix, variance::Nothing, indices) = nothing
function diagonal_correction!(K::AbstractMatrix, variance::AbstractVector, indices)
    (checksquare(K) == length(indices)) || throw(DimensionMismatch("checksquare(K) = $(checksquare(K)) ≠ $(length(indices)) = length(indices)"))
    var_ind = @view variance[indices]
    for (i, di) in enumerate(diagind(K))
         K[di] += var_ind[i]
     end
     return K
end

function compute_transformation_mats!(fact::MultipoleFactorization)
    @timeit fact.to "parallel transformation_mats" begin
        @sync for leaf in fact.tree.allleaves
            if !isempty(leaf.tgt_points)
                @spawn transformation_mats_kernel!(fact, leaf, false) # have to switch off timers if parallel
                # transformation_mats_kernel!(fact, leaf, true) # have to switch off timers if parallel
            end
        end
    end
end

# computational kernel of transformation mats that is run in parallel
function transformation_mats_kernel!(fact::MultipoleFactorization, leaf, timeit::Bool = true)
    num_multipoles = length(keys(fact.multi_to_single))

    tgt_points = leaf.tgt_points
    src_points = leaf.src_points
    src_points = vcat(src_points, [neighbor.src_points for neighbor in leaf.neighbors]...)
    src_indices = leaf.src_point_indices
    src_indices = vcat(src_indices, [neighbor.src_point_indices for neighbor in leaf.neighbors]...)
    leaf.near_indices = src_indices

    if timeit
        @timeit fact.to "get near mat" leaf.near_mat = fact.kernel.(tgt_points, permutedims(src_points))
    else
        leaf.near_mat = fact.kernel.(tgt_points, permutedims(src_points)) # IDEA: make lazy if they are too large?
    end

    if issymmetric(fact) # if target and source are equal,
        ind = 1:size(leaf.near_mat, 1) # square submatrix needs diagonal correction (self interactions)
        A = @view leaf.near_mat[ind, ind]
        diagonal_correction!(A, fact.variance, leaf.tgt_point_indices)
    end

    leaf.o2i = Vector{Matrix{Float64}}(undef, length(leaf.far_nodes))
    tot_far_points = get_tot_far_points(leaf)
    for far_node_idx in eachindex(leaf.far_nodes) # IDEA: parallelize?
        far_node = leaf.far_nodes[far_node_idx]
        if isempty(far_node.src_points) continue end
        src_points = far_node.src_points
        m = length(leaf.tgt_point_indices)
        if (num_multipoles * (m + tot_far_points)) < (m * tot_far_points)
            center(x) = x - far_node.center
            recentered_tgt = center.(tgt_points)
            recentered_src = center.(src_points)
            if isempty(far_node.s2o)
                if timeit
                    @timeit fact.to "source2outgoing" far_node.s2o = source2outgoing(fact, recentered_src)
                else
                    far_node.s2o = source2outgoing(fact, recentered_src, timeit)
                end
            end
            if timeit
                @timeit fact.to "outgoing2incoming" leaf.o2i[far_node_idx] = outgoing2incoming(fact, recentered_tgt)
            else
                leaf.o2i[far_node_idx] = outgoing2incoming(fact, recentered_tgt, timeit)
            end
        else
            if timeit
                @timeit fact.to "dense outgoing2incoming" leaf.o2i[far_node_idx] = fact.kernel.(leaf.tgt_points, permutedims(far_node.src_points))
            else
                leaf.o2i[far_node_idx] = fact.kernel.(leaf.tgt_points, permutedims(far_node.src_points))
            end
        end
    end
end

# IDEA: adaptive trunc_param, based on distance?
function outgoing2incoming(fact::MultipoleFactorization, recentered_tgt::AbstractVector{<:AbstractVector{<:Real}}, timeit::Bool = true)
    n, d = length(recentered_tgt), length(recentered_tgt[1])
    o2i_mat = zeros(Complex{Float64}, length(recentered_tgt), length(keys(fact.multi_to_single)))
    if timeit
        @timeit fact.to "norms" norms = norm.(recentered_tgt)
        @timeit fact.to "hyps" ra_hyps = cart2hyp.(recentered_tgt)
        @timeit fact.to "ffun" ffun = fact.get_F(norms)
    else
        norms = norm.(recentered_tgt)
        ra_hyps = cart2hyp.(recentered_tgt)
        ffun = fact.get_F(norms)
    end
    max_length_multi = max_num_multiindices(d, fact.trunc_param)
    hyp_harms = zeros(Complex{Float64}, n, max_length_multi) # pre-allocating
    denoms = similar(norms)
    for k in 0:fact.trunc_param # IDEA: parallel?
        if timeit
            @timeit fact.to "multiindices" multiindices = get_multiindices(d, k)
            @timeit fact.to "hyperharms" begin
                hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
                if d > 2
                    hyp_harms_k .= hyperspherical.(ra_hyps, k, permutedims(multiindices), Val(false))
                elseif d == 2
                    hyp_harms_k .= hypospherical.(ra_hyps, k, permutedims(multiindices))
                end # does not need to be normalized (done in s2o)
            end
            @timeit fact.to "denoms" @. denoms = norms^(k+1)
        else
            multiindices = get_multiindices(d, k)
            hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
            if d > 2
                hyp_harms_k .= hyperspherical.(ra_hyps, k, permutedims(multiindices), Val(false))
            elseif d == 2
                hyp_harms_k .= hypospherical.(ra_hyps, k, permutedims(multiindices)) # needs to be normalized
            end
            @. denoms = norms^(k+1)
        end
        r = fact.radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        for i in k:2:max_i
            ind = [fact.multi_to_single[(k, multiindices[h_idx], i)] for h_idx in 1:length(multiindices)]
            if timeit
                @timeit fact.to "f_coefs" F_coefs = ffun(k, i)
                @timeit fact.to "store" @. o2i_mat[:, ind] = (hyp_harms_k / denoms) * F_coefs
            else
                F_coefs = ffun(k, i)
                @. o2i_mat[:, ind] = (hyp_harms_k / denoms) * F_coefs
            end
        end
    end
    return o2i_mat
end

function source2outgoing(fact::MultipoleFactorization, recentered_src::AbstractVector{<:AbstractVector{<:Real}}, timeit::Bool = true)
    s2o_mat = zeros(Complex{Float64}, length(keys(fact.multi_to_single)), length(recentered_src))
    n, d = length(recentered_src), length(recentered_src[1])
    rj_hyps = cart2hyp.(recentered_src)
    if timeit
        @timeit fact.to "norms" norms = norm.(recentered_src)
    else
        norms = norm.(recentered_src)
    end
    gfun = fact.get_G(norms)
    max_length_multi = max_num_multiindices(d, fact.trunc_param)
    hyp_harms = zeros(Complex{Float64}, n, max_length_multi) # pre-allocating
    pows = similar(norms)
    for k in 0:fact.trunc_param
        N_k_alpha = gegenbauer_normalizer(d, k)
        if timeit
            @timeit fact.to "multiindices" multiindices = get_multiindices(d, k)
            # hyp_harms_k = @view hyp_harms[1:length(multiindices), :]
            # @timeit fact.to "hypharmcalc" hyperharms!(hyp_harms_k, fact, k, rj_hyps, true, multiindices)
            @timeit fact.to "hypharmcalc" begin
                hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
                if d > 2
                    hyp_harms_k .= hyperspherical.(rj_hyps, k, permutedims(multiindices), Val(false)) # needs to be normalized
                    hyp_harms_k ./= fact.normalizer_table[k+1, 1:length(multiindices)]' # normalizing
                    # hyp_harms_k .= hyperspherical.(rj_hyps, k, permutedims(multiindices), Val(true)) # needs to be normalized
                elseif d == 2
                    hyp_harms_k .= hypospherical.(rj_hyps, k, permutedims(multiindices)) # needs to be normalized
                end
            end
            @timeit fact.to "pows" @. pows = norms^k
        else
            multiindices = get_multiindices(d, k)
            hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
            if d > 2
                hyp_harms_k .= hyperspherical.(rj_hyps, k, permutedims(multiindices), Val(false)) # needs to be normalized
                hyp_harms_k ./= fact.normalizer_table[k+1, 1:length(multiindices)]'
                # hyp_harms_k .= hyperspherical.(rj_hyps, k, permutedims(multiindices), Val(true)) # needs to be normalized
            elseif d == 2
                hyp_harms_k .= hypospherical.(rj_hyps, k, permutedims(multiindices)) # needs to be normalized
            end
            @. pows = norms^k
        end
        r = fact.radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        for i in k:2:max_i
            ind = [fact.multi_to_single[(k, multiindices[h_idx], i)] for h_idx in 1:length(multiindices)]
            if timeit
                @timeit fact.to "g_coefs" G_coefs = gfun(k, i)
                @timeit fact.to "store" @. s2o_mat[ind, :] = N_k_alpha * $transpose(conj(hyp_harms_k) * G_coefs * pows)
            else
                G_coefs = gfun(k, i)
                @. s2o_mat[ind, :] = N_k_alpha * $transpose(conj(hyp_harms_k) * G_coefs * pows)
            end
        end
    end
    return s2o_mat
end

# function L(k,h,rj)
#     spher = SphericalFromCartesian()(rj)
#     harmonic = computeYlm((pi/2)-spher.ϕ,spher.θ, lmax=k,  SHType = SphericalHarmonics.RealHarmonics())[end-k+h]
#     return harmonic*(norm(rj)^(k))
# end
#
# function M(k,h,ra, alpha)
#     spher = SphericalFromCartesian()(ra)
#     harmonic = (computeYlm((pi/2)-spher.ϕ,spher.θ, lmax=k,  SHType = SphericalHarmonics.RealHarmonics())[end-k+h])
#     # N_alpha = 4pi #alpha dependent generally
#     N_k_alpha = (4pi)/(2k+1)
#     return N_k_alpha*harmonic/(norm(ra)^(k+1))
# end

# function split_spher_harm_coef(poly, param, correction)
#   poly_split = Array{Pair{Basic,Basic}}(undef, 0)

#   poly = expand(poly/correction)
#   # println("\n\nAfter correction ", poly,"\n\n")
#   poly = expand(subs(poly, (correction/correction)=>1))
#   # println(poly,"\n\n")
#   lowest_pow_r = 0
#   highest_pow_r = 0
#   for i in -(400):(400)
#     if coeff(poly, r, Basic(i)) != 0
#       lowest_pow_r = i
#       break
#     end
#   end
#   for i in (400):-1:-(400)
#     if coeff(poly, r, Basic(i)) != 0
#       highest_pow_r = i
#       break
#     end
#   end
#   lowest_pow_r -= 1 # for sqrt
#   poly = expand(poly / (r^lowest_pow_r))
#   # now lowest pow is 0
#   # println(poly)
#   mat = Array{Complex{Rational{BigInt}}}(undef, param+1, max(highest_pow_r-lowest_pow_r, param+1))

#     mat[1, 1] = subs(poly, rprime=>0, r=>0)
#     for i in 2:(max(highest_pow_r-lowest_pow_r, param+1))
#       # println("poly is ", poly)
#         tmp = coeff(subs(poly, r=>0), rprime, Basic(i-1))
#         # println("tmp is ", tmp)
#         # println("Type is ", typeof(tmp))
#         # println("after sub is ", subs(poly, r=>0))
#         num = convert(Complex{BigInt}, numerator(tmp))
#         den = convert(Complex{BigInt}, denominator(tmp))
#         mat[1,i] = num//den
#     end
#     for i in 2:(param+1)
#         tmp = coeff(subs(poly, rprime=>0), r, Basic(i-1))
#         num = convert(Complex{BigInt}, numerator(tmp))
#         den = convert(Complex{BigInt}, denominator(tmp))
#         mat[i,1] = num//den
#     end
#     for i in 1:(max(highest_pow_r-lowest_pow_r, param+1))
#         for j in 1:(param+1)
#           tmp = coeff(coeff(poly, rprime, Basic(i-1)), r, Basic(j-1))
#           num = convert(Complex{BigInt}, numerator(tmp))
#           den = convert(Complex{BigInt}, denominator(tmp))
#             mat[j,i] = num//den
#         end
#     end
#     Qmat, Rmat, Pmat = rationalrrqr(mat)
#     Rmat = Rmat * transpose(Pmat)
#   for i in 1:size(Rmat, 1)
#     if(norm(Qmat[:,i])*norm(Rmat[i,:])==0)
#       println("Rank ",i-1," it seems")
#       break
#     end
#     r_poly = 0
#     rprime_poly = 0
#     for j in 1:length(Qmat[:,i])
#       rprime_poly += (rprime^(j-1))*Rmat[i,j]
#       r_poly += (r^(j-1))*Qmat[j,i]
#     end
#     r_poly *= correction*(r^lowest_pow_r)
#     push!(poly_split, Pair(expand(rprime_poly), expand(r_poly)))
#   end
#   return SpherHarmCoef(poly_split)
# end
