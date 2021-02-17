module FastKernelTransform

using LinearAlgebra
using SparseArrays
using StaticArrays

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

include("util.jl")
include("gegenbauer.jl")
include("hyperspherical.jl")
include("data_generators.jl")
include("derivatives.jl")
# include("rectangular_tree.jl")
# include("square_tree.jl")
include("triangular_tree.jl")
include("multipole_factorization.jl")
include("expansion.jl")
include("symbolic.jl")
include("mul.jl")

end # module AutoFMM
