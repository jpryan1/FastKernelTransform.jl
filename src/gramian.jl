using CovarianceFunctions
function fkt(G::Gramian; max_dofs_per_leaf::Int = default_max_dofs_per_leaf,
    precond_param::Int = default_precond_paramt, trunc_param::Int = default_trunc_param,
    lazy_size::Int = lazy_size_heuristic(G.x, G.y), neighbor_scale::Real = 1/2,
    verbose::Bool = false, barnes_hut::Bool = (trunc_param == 0))
    MultipoleFactorization(G.k, G.x, G.y, max_dofs_per_leaf, precond_param, trunc_param,
                           lazy_size = lazy_size, neighbor_scale = neighbor_scale,
                           barnes_hut = barnes_hut, verbose = verbose)
end

# factorize only calls fkt if it is worth it
# TODO: need to think this over
# function LinearAlgebra.factorize(G::Gramian{<:IsotropicKernel})
#     if length(mat.points) < mat.max_dofs_per_leaf
#         x = math.points
#         return factorize(k.(x, permutedims(x)))
#     else
#         return fkt(mat)
#     end
# end
