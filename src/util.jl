################################# Linear Algebra ###############################
# type to delay instantiating multipole matrices until multiply
struct LazyMultipoleMatrix{T, G} <: AbstractMatrix{T}
    generator::G
    n::Int
    m::Int
end
function LazyMultipoleMatrix{T}(generator::G, n::Int, m::Int) where {T, G}
    LazyMultipoleMatrix{T, G}(generator, n, m)
end
Base.size(L::LazyMultipoleMatrix) = (L.n, L.m)
LinearAlgebra.Matrix(A::LazyMultipoleMatrix) = Matrix(A.generator()) # instantiates matrix when needed
LinearAlgebra.AbstractMatrix(A::LazyMultipoleMatrix) = A.generator() # instantiates matrix when needed
function LinearAlgebra.mul!(y::AbstractVector, A::LazyMultipoleMatrix, x::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(y, AbstractMatrix(A), x, α, β) # multiply
end
function LinearAlgebra.mul!(y::AbstractMatrix, A::LazyMultipoleMatrix, x::AbstractMatrix, α::Real = 1, β::Real = 0)
    mul!(y, AbstractMatrix(A), x, α, β) # multiply
end

using LinearAlgebra: checksquare
diagonal_correction!(K::AbstractMatrix, variance::Nothing, indices) = K
function diagonal_correction!(K::AbstractMatrix, variance::AbstractVector, indices)
    n = length(indices)
    n == length(diagind(K)) || throw(DimensionMismatch("checksquare(K) = $(checksquare(K)) ≠ $(length(indices)) = length(indices)"))
    var_ind = @view variance[indices]
    for (i, di) in enumerate(diagind(K))
         K[di] += var_ind[i]
     end
     return K
end

# WARNING: return type is critical here!
function diagonal_correction!(K::Gramian, variance::AbstractVector, indices)
    D = Diagonal(variance[indices])
    LazyDiagonalCorrection(K, D)
end

# type to add diagonal correction to lazy matrix
struct LazyDiagonalCorrection{T, AT<:AbstractMatrix{T}, DT<:Diagonal} <: AbstractMatrix{T}
    A::AT
    D::DT
end
Base.size(L::LazyDiagonalCorrection) = size(L.A)
function Base.getindex(L::LazyDiagonalCorrection, i::Int, j::Int)
    L.A[i, j] + (i == j ? L.D[i, j] : 0)
end
function LinearAlgebra.Matrix(L::LazyDiagonalCorrection)
    A = Matrix(L.A)
    for (i, ii) in enumerate(diagind(A))
        A[ii] += L.D[i, i]
    end
    return A
end
function LinearAlgebra.mul!(y::AbstractVector, A::LazyDiagonalCorrection, x::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(y, A.A, x, α, β) # multiply
    n = size(A.D, 1)
    i = 1:n
    xi, yi = @views x[i], y[i]
    return mul!(yi, A.D, xi, α, 1)
end

############################# rational QR factorization ########################
function proj(a, u)
    return u*(dot(u, a)//dot(u, u))
end

function rationalrrqr!(mat)
    n, m = size(mat)
    Q = Array{Complex{Rational{BigInt}}}(undef, n, n)
    perm = Matrix(1I, m, m)

    for col in 1:n
        matcol = @view mat[:, col:end] # view into columns of mat starting at col
        maxcol = argmax([norm(x) for x in eachcol(matcol)]) + (col - 1) # correcting offset

        mat[:, col], mat[:, maxcol] = mat[:, maxcol], mat[:, col]
        perm[:, col], perm[:, maxcol] = perm[:, maxcol], perm[:, col]

        u = mat[:, col]
        u_copy = copy(u)
        for j in 1:(col-1)
            u -= proj(u_copy, Q[:, j])
        end
        if norm(u) != 0
            Q[:, col] = u
        else
            u = rand(1:10000, n, 1)
            u_copy = copy(u)
            for j in 1:(col-1)
                u -= proj(u_copy, Q[:, j])
            end
            Q[:, col] = u
        end
    end
    R = Q \ mat
    return Q, R, perm
end

# create view into imaginary and real part of complex array
function real_imag_views(z::AbstractVector{<:Complex})
    z = reinterpret(Float64, z)
    re = @view z[1:2:end-1]
    im = @view z[2:2:end]
    return re, im
end

function real_imag_views(Z::AbstractArray{<:Complex})
    z = vec(Z)
    re, im = real_imag_views(z)
    Re = reshape(re, size(Z))
    Im = reshape(im, size(Z))
    return Re, Im
end


############################ miscellaneous helpers #############################
using Combinatorics: doublefactorial
doublefact(n::Int) = (n < 0) ? BigInt(1) : doublefactorial(n)
