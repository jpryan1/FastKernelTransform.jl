module FastKernelTransform

using LinearAlgebra
using SparseArrays
using StaticArrays
using LazyArrays

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}

using Base.Threads: @threads, @spawn, @sync
using SpecialFunctions
using SpecialPolynomials: Gegenbauer

using SymEngine
using RegionTrees
using CoordinateTransformations
using SphericalHarmonics
using TaylorSeries

using TimerOutputs

using Statistics
using CovarianceFunctions

export MultipoleFactorization
export fkt # fast kernel transform

#######################################
include("util.jl")
include("gegenbauer.jl")
include("hyperspherical.jl")
include("data_generators.jl")
include("derivatives.jl")
#######################################
include("rectangular_tree.jl")
# include("square_tree.jl")
# include("triangular_tree.jl")
#######################################
# IDEA: define default parameters here, add to constructor definitions
const default_max_dofs_per_leaf = 256
const default_precond_paramt = 2default_max_dofs_per_leaf
const default_trunc_param = 5
include("fmm_matrix.jl")
include("multipole_factorization.jl")
include("outgoing2incoming.jl")
include("source2outgoing.jl")
include("preconditioner.jl")
include("expansion.jl")
include("symbolic.jl")

include("mul.jl")
include("gramian.jl") # defines "fkt" on gramian matrix

end # module AutoFMM
