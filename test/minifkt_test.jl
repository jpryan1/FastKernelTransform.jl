module minifkt_test

using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: FactorizationParameters

using LowRankApprox

N      = 1000
d      = 3
src_pts = [rand(d) for _ in 1:N]
tgt_pts = [rand(d) .+ 2 for _ in 1:N]

#
mat_kern(r) = 1/r
mat_kern(x,y) = 1/norm(x-y)
qrable = true
# mat_kern(r) = 1/(1+r^2)
# mat_kern(x,y) = 1/(1+norm(x-y)^2)
# mat_kern(r) = exp(-r)
# mat_kern(x,y) = exp(-norm(x-y))
#
truth_mat = mat_kern.(tgt_pts, permutedims(src_pts))
# # TODO Error strangely high, next thing to try is replace upper bound for non-low rank mats
# # turns our trick into a bit of hack status but so bet it
#
degree = 10
rtol = 1e-6
params = FactorizationParameters(trunc_param = degree,
                                    neighbor_scale=rtol,
                                    lazy = qrable) #HACK lazy for qrable
U_mat, V_mat = fkt_wrapper(mat_kern, tgt_pts, src_pts, params)
guess = real(U_mat*(V_mat))
err = norm(guess-truth_mat, 2)/norm(truth_mat,2)
println("Err: ", err)
#
#
# sk, rd, T = id(truth_mat, rtol=1e-3)
# id_guess = copy(truth_mat)
# id_guess[:,rd] = id_guess[:,sk]*T
# id_err = norm(id_guess-truth_mat,2)/norm(truth_mat,2)
# println("ID Err: ", id_err)





end
