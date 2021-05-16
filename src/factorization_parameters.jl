const default_max_dofs_per_leaf = 256
const default_precond_paramt = 2default_max_dofs_per_leaf
const default_trunc_param = 5
const default_neighbor_scale = 1/2
const default_lazy = false
const default_verbose = false

# keeping all factorization parameters in one place
struct FactorizationParameters
    max_dofs_per_leaf::Int
    precond_param::Int
    trunc_param::Int
    neighbor_scale::Float64

    lazy::Bool
    verbose::Bool
    barnes_hut::Bool
end

function FactorizationParameters(; max_dofs_per_leaf::Int = default_max_dofs_per_leaf,
                                precond_param::Int = default_precond_paramt,
                                trunc_param::Int = default_trunc_param,
                                neighbor_scale::Real = default_neighbor_scale, lazy::Bool = default_lazy,
                                barnes_hut::Bool = (trunc_param == 0), verbose::Bool = default_verbose)
    FactorizationParameters(max_dofs_per_leaf, precond_param, trunc_param,
                            neighbor_scale, lazy, verbose, barnes_hut)
end

# potentially for lazy parameter
# L<:Union{Val{true}, Val{false}}
