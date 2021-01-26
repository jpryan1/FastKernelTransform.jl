mutable struct FmmMatrix{K, V<:AbstractVector{<:AbstractVector{<:Real}}} # IDEA: use CovarianceFunctions.Gramian
    kernel::K
    points::V
    max_dofs_per_leaf::Int64
    precond_param::Int64
    trunc_param::Int64
    to::TimerOutput
end

# fast kernel transform
function fkt(mat::FmmMatrix)
    return MultipoleFactorization(mat.kernel, mat.points, mat.max_dofs_per_leaf,
                            mat.precond_param, mat.trunc_param, mat.to)
end

# factorize only calls fkt if it is worth it
function LinearAlgebra.factorize(mat::FmmMatrix)
    if length(mat.points) < mat.max_dofs_per_leaf
        x = math.points
        return factorize(k.(x, permutedims(x)))
    else
        return fkt(mat)
    end
end

mutable struct MultipoleFactorization{K, TO<:TimerOutput, MST, TCT, NT, TT<:Tree} # DT<:Vector, FT, RFT<:AbstractVector{<:Int}
  kernel::K
  precond_param::Int64
  trunc_param::Int64
  to::TO
  # derivs::DT # Derivatives of kernel
  # Below are two helper arrays, converting from k,h,i representation of
  # multipole coefficients to single indices into an array (for efficient
  # matrix vector products)
  multi_to_single::MST
  # Lookup table for Julia functions for F and G functions in expansion
  # F_coef_table::FT
  # G_coef_table::FT
  # radial_fun_ranks::RFT
  # Lookup table for transformation coefficients
  transform_coef_table::TCT
  normalizer_table::NT
  tree::TT
end

# TODO: max_dofs_per_leaf < precond_param
# takes arbitrary isotropic kernel function k as input and creates symbolic expression
function MultipoleFactorization(kernel, points, max_dofs_per_leaf, precond_param,
                                trunc_param, to)
    # @vars r
    # kernel = k(r) # symbolic kernel
    multi_to_single = Dict() # TODO this doesn't need to be a dict anymore
    # F_coef_table = Matrix{Union{Nothing, Function}}(nothing, trunc_param+1, trunc_param+1)
    # G_coef_table = Matrix{Union{Nothing, Function}}(nothing, trunc_param+1, trunc_param+1)
    # radial_fun_ranks = zeros(Int64, trunc_param+1)

    d = length(points[1])
    transform_coef_table = transformation_coefficients(d, trunc_param)
    normalizer_table = Matrix{Float64}(undef, trunc_param+1, length(get_multiindices(d,trunc_param)))
    tree = initialize_tree(points, max_dofs_per_leaf)

    fact = MultipoleFactorization(kernel, precond_param, trunc_param, to,
        # get_derivs(kernel, trunc_param),
        multi_to_single,
        # F_coef_table, G_coef_table, radial_fun_ranks,
        transform_coef_table, normalizer_table, tree)

    # @timeit fact.to "Populate F, G tables" fill_f_g_tables!(fact)
    fill_index_mapping_tables!(fact)
    @timeit fact.to "Populate normalizer table" fill_normalizer_table!(fact)
    @timeit fact.to "Populate transformation table" compute_transformation_mats!(fact)
    return fact
end

function fill_normalizer_table!(fact::MultipoleFactorization)
    # upper arg ranges from 1//2 up to (2k+d-2)//2, by halves
    d = fact.tree.dimension
    k = fact.trunc_param
    for k_idx in 0:fact.trunc_param
        multiindices = get_multiindices(d, k_idx)
        for h_idx in 1:length(multiindices)
            h = multiindices[h_idx]
            h[end] = abs(h[end])
            normalizer!(fact, k_idx, h, h_idx)
        end
    end
end

function fill_index_mapping_tables!(fact::MultipoleFactorization)
    counter = 0
    for k in 0:fact.trunc_param
        multiindices = get_multiindices(fact.tree.dimension, k)
        # for i in k:2:(k+2*(fact.radial_fun_ranks[k+1]-1))
        for i in k:2:fact.trunc_param
            for h in multiindices
                counter += 1
                fact.multi_to_single[(k, h, i)] = counter
            end
        end
    end
end

function compute_transformation_mats!(fact::MultipoleFactorization)
    dim = fact.tree.dimension
    alpha = dim//2 - 1
    # For preconditioner
    node_queue = [fact.tree.root]
    while !isempty(node_queue)
        node = pop!(node_queue)
        if length(node.data.point_indices) < fact.precond_param
            tgt_points = node.data.points
            @timeit fact.to "get diag inv for precond" begin
                # node.data.diag_block = factorize(fact.kernel.(tgt_points, permutedims(tgt_points)))
                node.data.diag_block = cholesky(fact.kernel.(tgt_points, permutedims(tgt_points)))
            end
        else
            append!(node_queue, children(node))
        end
    end
    @timeit fact.to "parallel transformation_mats" begin
        @sync for leaf in allleaves(fact.tree.root)
            if !isempty(leaf.data.points)
                @spawn transformation_mats_kernel!(fact, leaf, false) # have to switch off timers if parallel
                # transformation_mats_kernel!(fact, leaf, true) # have to switch off timers if parallel
            end
        end
    end
end

# computational kernel of transformation mats that is run in parallel
function transformation_mats_kernel!(fact::MultipoleFactorization, leaf, timeit::Bool = true)
    tgt_points = leaf.data.points
    src_points = copy(tgt_points)
    src_points = vcat(src_points, [neighbor.data.points for neighbor in leaf.data.neighbors]...)

    src_indices = copy(leaf.data.point_indices)
    src_indices = vcat(src_indices, [neighbor.data.point_indices for neighbor in leaf.data.neighbors]...)
    leaf.data.near_indices = src_indices

    if timeit
        @timeit fact.to "get near mat" leaf.data.near_mat = fact.kernel.(tgt_points, permutedims(src_points))
    else
        leaf.data.near_mat = fact.kernel.(tgt_points, permutedims(src_points))
    end
    leaf.data.o2i = Vector{Matrix{Float64}}(undef, length(leaf.data.far_nodes))
    for far_node_idx in eachindex(leaf.data.far_nodes) # IDEA: parallelize?
        far_node = leaf.data.far_nodes[far_node_idx]
        if isempty(far_node.data.points) continue end
        # @timeit fact.to "far" begin
            src_points = far_node.data.points
            center(x) = x - RegionTrees.center(far_node)
            recentered_tgt = center.(tgt_points)
            recentered_src = center.(src_points)
            if isempty(far_node.data.s2o)
                if timeit
                    @timeit fact.to "source2outgoing" far_node.data.s2o = source2outgoing(fact, recentered_src)
                else
                    far_node.data.s2o = source2outgoing(fact, recentered_src, timeit)
                end
            end
            if timeit
                @timeit fact.to "outgoing2incoming" leaf.data.o2i[far_node_idx] = outgoing2incoming(fact, recentered_tgt)
            else
                leaf.data.o2i[far_node_idx] = outgoing2incoming(fact, recentered_tgt, timeit)
            end
        # end
    end
end

function normalizer!(fact, k, h, h_idx) # TODO no need to call this twice, just get rid of sqrt, move to one harmonic call?
    m_vec = vcat(k,h) #TODO fix this hack
    d = length(m_vec)+1
    N2 = 2π
    for j in 1:(d-2)
        alpha_j = (d-j-1)//2
        numer = (sqrt(π)
                *gamma(alpha_j+m_vec[j+1]+(1//2))
                *(alpha_j+m_vec[j+1])
                *factorial(2alpha_j+m_vec[j]+m_vec[j+1]-1))
        denom = (gamma(alpha_j+m_vec[j+1]+1)
                *factorial(m_vec[j]-m_vec[j+1])
                *(alpha_j+m_vec[j])
                *factorial(2alpha_j+2*m_vec[j+1]-1))
        N2 *= (numer/denom)
    end
    fact.normalizer_table[k+1,h_idx] = (N2)
    return (N2)
end

# IDEA: change pt_hyps from vec of vec to matrix for cache locality
function hyperharms(fact, k, pt_hyps, use_normalizer)  # 2-arr from h_idx, a to float
    # TODO write unit test to check that addition theorem is respected
    d = length(pt_hyps[1])
    multiindices = get_multiindices(d, k)
    harms = Matrix{Complex{Float64}}(undef, length(multiindices), length(pt_hyps))
    sins = map((x)->sin.(x[2:end]), pt_hyps)
    coss = map((x)->cos.(x[2:end]), pt_hyps)

    # i is alpha index
    # n is polynomial order
    # j is coss index
    @inline function gegenbauer_helper!(r::AbstractVector, i, n, j)
        _d = fact.tree.dimension
        _k = fact.trunc_param
        alphas = (1:(2*_k+_d-2)) / 2 # rational slows things down if we are otherwise doing FLOPs
        α = alphas[i]
        @. r = gegenbauer(α, n, getindex(coss, j)) # IDEA: @simd?
    end
    @inline function gegenbauer_helper(i, n, j)
        r = zeros(length(coss))
        gegenbauer_helper!(r, i, n, j)
    end

    for h_idx in 1:length(multiindices)
        h = multiindices[h_idx]
        neg_ind = h[d-2] < 0
        h[d-2] = abs(h[d-2])
        N = 1
        if use_normalizer
            N = fact.normalizer_table[k+1, h_idx]
        end

        gegenbauer = gegenbauer_helper(d-2+(2*h[1]), k-h[1], 1)
        @. harms[h_idx, :] = (1/N) * exp(1im*h[d-2] * getindex(pt_hyps, d)) * getindex(sins, 1)^h[1] * gegenbauer
        for j in 2:(d-2)
            gegenbauer_helper!(gegenbauer, d-j-1+2*h[j], h[j-1]-h[j], j)
            @. harms[h_idx, :] *= getindex(sins, j)^h[j] * gegenbauer
        end

        if neg_ind
            @. harms[h_idx, :] = conj(harms[h_idx, :])
        end
        if neg_ind
            h[d-2] *= -1
        end
    end
    return harms
end

function source2outgoing(fact::MultipoleFactorization, recentered_src::AbstractVector{<:AbstractVector{<:Real}}, timeit::Bool = true)
    s2o_mat = zeros(Complex{Float64}, length(keys(fact.multi_to_single)), length(recentered_src))
    d = length(recentered_src[1])
    rj_hyps = map(cart2hyp, recentered_src)

    if timeit
        @timeit fact.to "norms" norms = norm.(recentered_src)
    else
        norms = norm.(recentered_src)
    end
    G_coefs = similar(norms)
    for k in 0:fact.trunc_param
        N_k_alpha = 1/((d+2k-2)*doublefact(d-4))
        N_k_alpha = convert(Float64, N_k_alpha)
        N_k_alpha *= iseven(d) ? (2π)^(d/2) : 2*(2π)^((d-1)/2)

        if timeit
            @timeit fact.to "hypharmcalc" hyp_harms = hyperharms(fact, k, rj_hyps, true)
            @timeit fact.to "multiindices" multiindices = get_multiindices(d, k)
            @timeit fact.to "pows" pows = norms .^ (k)
        else
            hyp_harms = hyperharms(fact, k, rj_hyps, true)
            multiindices = get_multiindices(d, k)
            pows = norms .^ (k)
        end

        # for i in k:2:(k+2*(fact.radial_fun_ranks[k+1]-1))
        for i in k:2:fact.trunc_param
            if timeit
                # @timeit fact.to "g_coefs" G_coefs = map(fact.G_coef_table[k+1, i+1], norms)
                gfun = G(k, i)
                @timeit fact.to "g_coefs" @. G_coefs = gfun(norms)
            else
                gfun = G(k, i)
                @. G_coefs = gfun(norms)
            end
            for h_idx in 1:length(multiindices)
                if timeit
                    @timeit fact.to "store" @. s2o_mat[fact.multi_to_single[(k,multiindices[h_idx],i)], :] = N_k_alpha * conj(hyp_harms[h_idx, :]) * G_coefs * pows
                else
                    @. s2o_mat[fact.multi_to_single[(k,multiindices[h_idx],i)], :] = N_k_alpha * conj(hyp_harms[h_idx, :]) * G_coefs * pows
                end
            end
        end
    end
    return s2o_mat
end

# IDEA: adaptive trunc_param, based on distance?
function outgoing2incoming(fact::MultipoleFactorization, recentered_tgt::AbstractVector{<:AbstractVector{<:Real}}, timeit::Bool = true)
    d = length(recentered_tgt[1])
    o2i_mat = zeros(Complex{Float64}, length(recentered_tgt), length(keys(fact.multi_to_single)))
    if timeit
        @timeit fact.to "norms" norms = norm.(recentered_tgt)
        @timeit fact.to "hyps" ra_hyps = map(cart2hyp, recentered_tgt)
        @timeit fact.to "ffun" ffun = get_F(fact, norms)
    else
        norms = norm.(recentered_tgt)
        ra_hyps = map(cart2hyp, recentered_tgt)
        ffun = get_F(fact, norms)
    end
    F_coefs = similar(norms)
    for k in 0:fact.trunc_param
        if timeit
            @timeit fact.to "hypharmcalc" hyp_harms = hyperharms(fact, k, ra_hyps, false) # 2-arr from h_idx,a to float
            @timeit fact.to "multiindices" multiindices = get_multiindices(d, k)
            @timeit fact.to "denoms" denoms = norms .^ (k+1)
        else
            hyp_harms = hyperharms(fact, k, ra_hyps, false)
            multiindices = get_multiindices(d, k)
            denoms = norms.^(k+1)
        end
        # for i in k:2:(k+2*(fact.radial_fun_ranks[k+1]-1))
        for i in k:2:fact.trunc_param
            if timeit
                # @timeit fact.to "f_coefs" F_coefs = map(fact.F_coef_table[k+1, i+1], norms) # bottleneck
                @timeit fact.to "f_coefs" F_coefs = ffun(k, i)
            else
                # F_coefs = map(fact.F_coef_table[k+1, i+1], norms)
                F_coefs = ffun(k, i)
            end
            for h_idx in 1:length(multiindices)
                if timeit
                    @timeit fact.to "store" @. o2i_mat[:, fact.multi_to_single[(k,multiindices[h_idx],i)]] = (hyp_harms[h_idx, :] / denoms) * F_coefs
                else
                    @. o2i_mat[:, fact.multi_to_single[(k,multiindices[h_idx],i)]] = (hyp_harms[h_idx, :] / denoms) * F_coefs
                end
            end

        end
    end
    return o2i_mat
end

# function L(k,h,rj)
#     spher = SphericalFromCartesian()(rj)
#     harmonic = computeYlm((pi/2)-spher.ϕ,spher.θ, lmax=k,  SHType = SphericalHarmonics.RealHarmonics())[end-k+h]
#     return harmonic*(norm(rj)^(k))
# end
#
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
