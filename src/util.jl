using Combinatorics: doublefactorial
doublefact(n::Int) = (n < 0) ? BigInt(1) : doublefactorial(n)

# miscellaneous helpers
function proj(a, u)
    return u*(dot(u, a)//dot(u, u))
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
