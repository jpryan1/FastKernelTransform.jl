using CovarianceFunctions
function fkt(G::Gramian; max_dofs_per_leaf::Int = default_max_dofs_per_leaf,
    precond_param::Int = default_precond_paramt, trunc_param::Int = default_trunc_param,
    lazy::Bool = default_lazy, neighbor_scale::Real = default_neighbor_scale,
    verbose::Bool = default_verbose, barnes_hut::Bool = (trunc_param == 0))
    params = FactorizationParameters(max_dofs_per_leaf, precond_param, trunc_param,
                                        neighbor_scale, lazy, verbose, barnes_hut)
    return fkt(G, params)
end


function fkt(G::Gramian, params::FactorizationParameters)
    MultipoleFactorization(G.k, G.x, G.y, nothing, params) # nothing refers to variance
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
