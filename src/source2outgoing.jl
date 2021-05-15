function source2outgoing(fact::MultipoleFactorization, recentered_src::AbstractVecOfVec{<:Real})
    if islazy(fact)
        generator = ()->_source2outgoing(fact, recentered_src)
        n, m = nmultipoles(fact), length(recentered_src)
        return LazyMultipoleMatrix{Complex{T}}(generator, n, m)
    else
        return _source2outgoing(fact, recentered_src)
    end
end
# computes source to outgoing matrices densely
function _source2outgoing(fact::MultipoleFactorization, recentered_src::AbstractVector{<:AbstractVector{<:Real}}, timeit::Bool = true)
    s2o_mat = zeros(Complex{Float64}, nmultipoles(fact), length(recentered_src))
    n, d = length(recentered_src), length(recentered_src[1])
    rj_hyps = cart2hyp.(recentered_src)
    norms = norm.(recentered_src)
    gfun = fact.get_G(norms)
    max_length_multi = max_num_multiindices(d, fact.trunc_param)
    hyp_harms = zeros(Complex{Float64}, n, max_length_multi) # pre-allocating
    pows = similar(norms)
    for k in 0:fact.trunc_param
        N_k_alpha = gegenbauer_normalizer(d, k)

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

        r = fact.radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        for i in k:2:max_i # abstract away
            ind = [fact.multi_to_single[(k, multiindices[h_idx], i)] for h_idx in 1:length(multiindices)]
            G_coefs = gfun(k, i)
            @. s2o_mat[ind, :] = N_k_alpha * $transpose(conj(hyp_harms_k) * G_coefs * pows)
        end
    end
    return s2o_mat
end
