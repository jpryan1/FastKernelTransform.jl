
struct MiniMultipoleFactorization{T, K, PAR <: FactorizationParameters,
                MST, NT, FT, GT, RT<:AbstractVector{Int}} <: Factorization{T}
    kernel::K
    params::PAR

    multi_to_single::MST
    normalizer_table::NT

    get_F::FT
    get_G::GT
    radial_fun_ranks::RT
    _k::T # sample output of kernel, only here to determine element type
end

function pick_best_degree(kernel, tgt_pts::AbstractVecOfVec{<:Real},
                                src_pts::AbstractVecOfVec{<:Real},
                                params::FactorizationParameters)
    dimension = length(tgt_pts[1])
    alpha=(dimension-2)/2
    center = src_pts[1]
    for i in 2:length(src_pts)
        center += src_pts[i]
    end
    center /= length(src_pts)
    for i in 1:length(src_pts)
        src_pts[i] -= center
    end
    for i in 1:length(tgt_pts)
        tgt_pts[i] -= center
    end

    max_rprime = maximum(norm.(src_pts))
    min_r = minimum(norm.(tgt_pts))
    max_r = maximum(norm.(tgt_pts))

    TEST_SCALE = 10
    test_r_primes = Array{Array{Float64, 1}, 1}(undef,0)
    test_rs = Array{Array{Float64, 1}, 1}(undef,0)
    for i in collect((max_rprime/TEST_SCALE):(max_rprime/TEST_SCALE):max_rprime)
        tmp = [i,0]
        for extra_d in 1:(dimension-2)
            push!(tmp,0)
        end
        push!(test_r_primes, tmp)
    end

    for i in collect(min_r:((max_r-min_r)/TEST_SCALE):max_r)
        for ang in pi/2:(2pi/TEST_SCALE):2pi
            tmp = [i*cos(ang), i*sin(ang)]
            for extra_d in 1:(dimension-2)
                push!(tmp, 0)
            end
            push!(test_rs,tmp)
        end
    end

    src_norms = norm.(test_r_primes)
    tgt_norms = norm.(test_rs)

    # Iterate over degrees until desired tol met.
    max_degree = 20

    for degree_test in 0:max_degree
        new_params = FactorizationParameters(trunc_param = degree_test,
                                            neighbor_scale = params.neighbor_scale,
                                            lazy = params.lazy) #HACK lazy for qrable

        F = MiniMultipoleFactorization(kernel, test_r_primes, test_rs, new_params)
        gfun = F.get_G(src_norms)
        ffun = F.get_F(tgt_norms)

        max_deg_err = 0
        err_sum=0
        err_count=0
        for (test_rp_idx, test_rprime) in enumerate(test_r_primes)
            for (test_r_idx, test_r) in enumerate(test_rs)
                guess = 0

                for k in 0:degree_test
                    src_norm = norm(test_rprime)
                    tgt_norm = norm(test_r)

                    pow = src_norm^k
                    r = F.radial_fun_ranks[k+1]
                    max_i = k+2*(r-1)
                    denom = tgt_norm ^ (k+1)

                    for i in k:2:max_i
                        guess += (
                        gegenbauer(alpha, k, dot(test_r,test_rprime)/(norm(test_r)*norm(test_rprime)))
                                *gfun(k,i)[test_rp_idx]
                                *pow
                                *ffun(k,i)[test_r_idx]
                                /denom)
                    end
                end
                err = abs(kernel(norm(test_r-test_rprime))-guess)
                max_deg_err = max(max_deg_err, err)
                err_sum += err
                err_count +=1
                if degree_test==10 && err >0.005
                    println("Err ", err ," for pts ", test_r, " ", test_rprime)
                    println("Compare ", norm(test_r), " to ", min_r, " and " ,norm(test_rprime), " to ", max_rprime)
                end

            end
        end
        println("deg=",degree_test," max err ",max_deg_err, " avg err ", err_sum/err_count)

    end

    return 5

end


function MiniMultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                params::FactorizationParameters = FactorizationParameters())
    dimension = length(tgt_points[1])
    qrable = params.lazy
    if params.trunc_param == 0
        qrable = false
    end
    # get_F, get_G, radial_fun_ranks = init_F_G(kernel, dimension, params.trunc_param, Val(qrable(kernel)))
    get_F, get_G, radial_fun_ranks = init_F_G(kernel, dimension, params.trunc_param, Val(qrable)) # HACK lazy for qrable
    MiniMultipoleFactorization(kernel, tgt_points, src_points, get_F, get_G, radial_fun_ranks, params)
end

# takes arbitrary isotropic kernel
# IDEA: convert points to vector of static arrays, dispatch 2d (and 1d) implementation on its type!
# IDEA: given radial_fun_ranks, can we get rid of trunc_param?
function MiniMultipoleFactorization(kernel, tgt_points::AbstractVecOfVec{<:Real}, src_points::AbstractVecOfVec{<:Real},
                                get_F, get_G, radial_fun_ranks::AbstractVector,
                                params = FactorizationParameters())
    (params.max_dofs_per_leaf ≤ params.precond_param || (params.precond_param == 0)) || throw(DomainError("max_dofs_per_leaf < precond_param"))
    dimension = length(tgt_points[1])
    multi_to_single = get_index_mapping_table(dimension, params.trunc_param, radial_fun_ranks)
    normalizer_table = squared_hyper_normalizer_table(dimension, params.trunc_param)
    outgoing_length = length(keys(multi_to_single))

    _k = kernel(tgt_points[1], src_points[1]) # sample evaluation used to determine element type

    fact = MiniMultipoleFactorization(kernel, params, multi_to_single,
                    normalizer_table, get_F, get_G, radial_fun_ranks, _k)
    return fact
end



# nmultipoles(F), F.get_F, F.get_G(norms), F.params.trunc_param,  F.radial_fun_ranks[k+1], F.multi_to_single,F.normalizer_table



function fkt_wrapper(lkern, tgt_pts::Array{Array{Float64,1},1},
    src_pts::Array{Array{Float64,1},1}, params)
    F = MiniMultipoleFactorization(lkern, tgt_pts, src_pts, params)

    center = src_pts[1]
    for i in 2:length(src_pts)
        center += src_pts[i]
    end
    center /= length(src_pts)
    for i in 1:length(src_pts)
        src_pts[i] -= center
    end
    for i in 1:length(tgt_pts)
        tgt_pts[i] -= center
    end
    println("Center is ", center)
    max_rprime = maximum(norm.(src_pts))
    min_r = minimum(norm.(tgt_pts))
    println("ratio ", max_rprime/min_r)

    d = length(src_pts[1])

    nmultipoles =  length(keys(F.multi_to_single))
    s2o_mat = zeros(Complex{Float64}, nmultipoles, length(src_pts))
    o2i_mat = zeros(Complex{Float64}, length(tgt_pts), length(keys(F.multi_to_single)))

    src_hyps = cart2hyp.(src_pts)
    src_norms = norm.(src_pts)
    tgt_norms = norm.(tgt_pts)
    tgt_hyps = cart2hyp.(tgt_pts)

    gfun = F.get_G(src_norms)
    ffun = F.get_F(tgt_norms)

    pows = similar(src_norms)

    mat_ind = 1

    top_sing = 0
    for k in 0:F.params.trunc_param
        N_k_alpha = gegenbauer_normalizer(d, k)

        multiindices = get_multiindices(d, k)
        if d > 2
            src_hyp_harms_k = hyperspherical.(src_hyps, k, permutedims(multiindices), Val(false)) # needs to be normalized
            src_hyp_harms_k ./= F.normalizer_table[k+1, 1:length(multiindices)]'
            tgt_hyp_harms_k = hyperspherical.(tgt_hyps, k, permutedims(multiindices), Val(false))
        elseif d == 2
            src_hyp_harms_k = hypospherical.(src_hyps, k, permutedims(multiindices)) # needs to be normalized
            tgt_hyp_harms_k = hypospherical.(tgt_hyps, k, permutedims(multiindices)) # needs to be normalized
        end

        @. pows = src_norms^k
        r = F.radial_fun_ranks[k+1]
        max_i = k+2*(r-1)
        denoms = tgt_norms .^ (k+1)

        sz = convert(Int64, F.radial_fun_ranks[k+1])
        println("Initial size ", sz)
        println("Ranks ", F.radial_fun_ranks)
        left_mat = zeros(length(tgt_pts), sz)
        right_mat = zeros(sz, length(src_pts))
        rad_mat_ind = 1
        for i in k:2:max_i
            G_coefs = N_k_alpha * gfun(k, i) .* pows
            F_coefs = ffun(k, i) ./ denoms
            for j in 1:length(tgt_pts)
                left_mat[j, rad_mat_ind] = F_coefs[j]
            end
            for j in 1:length(src_pts)
                right_mat[rad_mat_ind,j] = G_coefs[j]
            end
            rad_mat_ind += 1
        end

        q1, r1 = qr(left_mat)
        q1 = q1[:, 1:size(r1,1)]
        q2, r2 = qr(transpose(right_mat))
        q2 = q2[:, 1:size(r1,1)]

        u1, svals, vt1 = svd(r1*transpose(r2))
        umat = q1*u1
        vtmat = q2*vt1
        # @time big_mat = left_mat*right_mat
        # small_mat = right_mat*left_mat

        # @time umat, svals, vtmat = svd(big_mat)
        if k == 0
            top_sing = svals[1]
        end

        tol = params.neighbor_scale # HACK TODO better
        rnk = length(svals)
        for i in 1:length(svals)
            if svals[i]/top_sing < tol
                rnk = i-1
                break
            end
        end
        println("k=",k," rank is ", rnk," out of ", length(svals))
        # println(rnk, " ", svals[1])

        if rnk==0
            continue
        end
        svals = svals[1:rnk]
        umat = umat[:, 1:rnk]
        vtmat = vtmat[:, 1:rnk]
        new_left_mat = umat*diagm(sqrt.(svals))
        new_right_mat = diagm(sqrt.(svals))*transpose(vtmat)
        # println("err = ",norm(left_mat*right_mat - new_left_mat*new_right_mat))
        # new_left_mat = left_mat
        # new_right_mat = right_mat

        for h_idx in 1:length(multiindices)

            for rad_mat_ind in 1:size(new_left_mat,2) # abstract away
                G_coefs = new_right_mat[ rad_mat_ind,:]
                @. s2o_mat[mat_ind, :] = src_hyp_harms_k[:,h_idx] * G_coefs

                F_coefs = new_left_mat[ :,rad_mat_ind]
                @. o2i_mat[:, mat_ind] = (tgt_hyp_harms_k[:,h_idx]) * F_coefs

                mat_ind += 1
            end
        end
    end


    return o2i_mat[:,1:(mat_ind-1)], s2o_mat[1:(mat_ind-1),:]
end


# transition plan
# 3 combine radials into matrix to index into
# 4 do truncated svd on matrix, maintain accuracy
