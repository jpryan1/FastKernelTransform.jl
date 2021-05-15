function outgoing2incoming(fact::MultipoleFactorization, recentered_tgt::AbstractVecOfVec{<:Real})
    if islazy(fact)
        generator = ()->_outgoing2incoming(fact, recentered_tgt)
        n, m = length(recentered_tgt), nmultipoles(fact)
        return LazyMultipoleMatrix{Complex{T}}(generator, n, m)
    else
        return _outgoing2incoming(fact, recentered_tgt)
    end
end

# IDEA: adaptive trunc_param, based on distance?
function _outgoing2incoming(fact::MultipoleFactorization, recentered_tgt::AbstractVector{<:AbstractVector{<:Real}})
    n, d = length(recentered_tgt), length(recentered_tgt[1])
    o2i_mat = zeros(Complex{Float64}, length(recentered_tgt), length(keys(fact.multi_to_single)))

    norms = norm.(recentered_tgt)
    ra_hyps = cart2hyp.(recentered_tgt)
    ffun = fact.get_F(norms)

    max_length_multi = max_num_multiindices(d, fact.trunc_param)
    hyp_harms = zeros(Complex{Float64}, n, max_length_multi) # pre-allocating
    denoms = similar(norms)
    for k in 0:fact.trunc_param # IDEA: parallel?
        multiindices = get_multiindices(d, k)
        hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
        if d > 2 # TODO: abstract away
            hyp_harms_k .= hyperspherical.(ra_hyps, k, permutedims(multiindices), Val(false))
        elseif d == 2
            hyp_harms_k .= hypospherical.(ra_hyps, k, permutedims(multiindices)) # needs to be normalized
        end
        @. denoms = norms^(k+1)
        r = fact.radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        for i in k:2:max_i
            ind = [fact.multi_to_single[(k, multiindices[h_idx], i)] for h_idx in 1:length(multiindices)]
            F_coefs = ffun(k, i)
            @. o2i_mat[:, ind] = (hyp_harms_k / denoms) * F_coefs
        end
    end
    return o2i_mat
end
