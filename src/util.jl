using Combinatorics: doublefactorial
doublefact(n::Int) = (n < 0) ? BigInt(1) : doublefactorial(n)

# miscellaneous helpers
function proj(a, u)
    return u*(dot(u, a)//dot(u, u))
end

function get_multiindices(d, k)
    if d == 2 return [@SVector [i] for i in 0 : (k==0 ? 0 : 1)] end
    current_index = @MVector zeros(Int, d-2) # mutable static array
    T = typeof(current_index)
    multiindices = zeros(T, 0)
    push!(multiindices, copy(current_index))
    while true
        current_index[(d-2)] += 1
        for i in (d-2):-1:2
            if current_index[i] > current_index[i-1]
                current_index[i-1] += 1
                current_index[i] = 0
            end
        end
        if current_index[1] > k
            break
        end
        push!(multiindices, copy(current_index))
        if current_index[d-2] != 0
            current_index[d-2] *= -1
            push!(multiindices, copy(current_index))
            current_index[d-2] *= -1
        end
    end
    return multiindices # d - 2 indices, ordered, last can be neg
end

# cartesian to hyper-spherical coordinates
function cart2hyp(pt)
    d = length(pt)
    pt_hyp = Array{Float64, 1}(undef, d)
    pt_hyp[1] = norm(pt)
    pt_hyp[d] = 2*atan(pt[2],pt[1]+norm(pt[1:2]))
    for i in 2:(d-1)
        pt_hyp[d-i+1] = atan(norm(pt[1:i]), pt[i+1])
    end
    return pt_hyp
end


# rational qr factorization
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

# iterative Gegenbauer implementation
function gegenbauer(α::Real, n::Int, x::Real)
    α, x = promote(α, x)
    gegenbauer(α, n, x)
end
# for full performance, important to pass α as a float too
@inline function gegenbauer(α::T, n::Int, x::T) where {T <: Real}
    C1 = one(T)
    n == 0 && return C1
    C2 = 2α*x
    for k in 2:n
        @fastmath C1, C2 = C2, (2*x*(k+α-1) * C2 - (k+2α-2) * C1) / k
    end
    return C2
end

# iterative Chebyshev of the first or second kind
@inline function chebyshev(n::Int, x::Real, kind::Val{T} = Val(1)) where {T}
    C1 = one(x)
    n == 0 && return C1
    C2 = T*x
    for k in 2:n
        @fastmath C1, C2 = C2, (2x*C2 - C1)
    end
    return C2
end
