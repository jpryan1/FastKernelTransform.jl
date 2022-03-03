module rank_compare

using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using FastKernelTransform
using FastKernelTransform: FactorizationParameters

src_size = 500
tgt_size = 2000
num_points = 1000

d = 3

src_pts = [rand(d)]
tgt_pts = [rand(d)]
for i in 1:src_size
    dir = randn(d)
    dir /= norm(dir)
    rad = rand()
    push!(src_pts, rad*dir)
end

for i in 1:tgt_size
    dir = randn(d)
    dir /= norm(dir)
    rad = rand()
    rad += 1.5
    push!(tgt_pts, rad*dir)
end

#TODO get rid of this hack
src_pts = src_pts[2:end]
tgt_pts = tgt_pts[2:end]


# tgt_pts = [rand(d) .+ 2 for _ in 1:tgt_size]
kernel_name="electrostatic"
mat_kern(r) = 1/r
mat_kern(x,y) = 1/norm(x-y)
# mat_kern(r) = exp(-r)
# mat_kern(x,y) = exp(-norm(x-y))

# mat_kern(r) = 1/(1+r^2)
# mat_kern(x,y) = 1/(1+norm(x-y)^2)
truth_mat = mat_kern.(tgt_pts, permutedims(src_pts))

svecs, svals = svd(truth_mat);
test_tols = [0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001]
fkt_ranks = []
fkt_errs = []
# for rtol in test_tols
for qrable in [true false]
    for degree in 0:7
        rtol = 1e-32
        println("fkt ",degree)

        params = FactorizationParameters(trunc_param = degree,
                                         neighbor_scale=rtol,
                                          lazy = qrable) #HACK lazy for qrable
        best_degree = pick_best_degree(mat_kern, tgt_pts, src_pts, params)
        # params = FactorizationParameters(trunc_param = best_degree,
        #                                      neighbor_scale=rtol,
        #                                      lazy = qrable) #HACK lazy for qrable
        #
        #
        # U_mat, V_mat = fkt_wrapper(mat_kern, tgt_pts, src_pts, params)
        # guess = real(U_mat*(V_mat))
        # err = norm(guess-truth_mat, 2)/norm(truth_mat,2)
        # fkt_rank = size(V_mat, 1)
        # println(fkt_rank)
        # push!(fkt_ranks, fkt_rank)
        # push!(fkt_errs, err)
    end
end

# nystrom_ranks = []
# nystrom_errs = []
#
# #FKT is linear in size of new matrix
# # ID is size of new matrix times num cols in old matrix, aka num proxy points
#
# for tol in test_tols
#     # tol = 5*10.0^tol_pow
#     println("id ", tol)
#     sk, rd, T = id(truth_mat, rtol=tol)
#     id_rank = length(sk)
#     id_guess = copy(truth_mat)
#     id_guess[:,rd] = id_guess[:,sk]*T
#     id_err = norm(id_guess-truth_mat,2)/norm(truth_mat,2)
#     push!(nystrom_ranks, id_rank)
#     push!(nystrom_errs, id_err)
# end
#
# svd_output = open(string("pyplots/data/relerr_vs_rank_svd_", kernel_name, ".txt"), "w")
# fktoutput = open(string("pyplots/data/relerr_vs_rank_fkt_", kernel_name, ".txt"), "w")
# nystromoutput = open(string("pyplots/data/relerr_vs_rank_nystrom_", kernel_name, ".txt"), "w")
#
# for i in 1:min(length(svals)-1, maximum(fkt_ranks))
#     write(svd_output, string(i,  ",", svals[i+1]/svals[1],"\n"))
# end
# close(svd_output)
#
# for i in 1:length(fkt_ranks)
#     write(fktoutput, string(fkt_ranks[i],  ",", fkt_errs[i],"\n"))
# end
# close(fktoutput)
#
#
# for i in 1:length(nystrom_ranks)
#     write(nystromoutput, string(nystrom_ranks[i],  ",", nystrom_errs[i],"\n"))
# end
# close(nystromoutput)

#Experiment conclusion: Recompression needed in non-qrable cases of FKT



end # module rank_compare
