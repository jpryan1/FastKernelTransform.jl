module FastKernelTransform

using LinearAlgebra
using SparseArrays
using StaticArrays
using LazyArrays

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
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
using NearestNeighbors

export MultipoleFactorization
export fkt, fkt_wrapper, pick_best_degree # fast kernel transform

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
include("factorization_parameters.jl")
include("fmm_matrix.jl")
include("multipole_factorization.jl")
include("outgoing2incoming.jl")
include("source2outgoing.jl")
include("preconditioner.jl")
include("expansion.jl")
include("symbolic.jl")
include("minifkt.jl")

include("mul.jl")
include("gramian.jl") # defines "fkt" on gramian matrix

end # module AutoFMM
