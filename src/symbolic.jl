# kernel-related functions
function get_kernel_fun(kernel::Basic)
    @vars r
    kern_lam = lambdify(kernel, [r]) # lambdify relies on invokelatest
    kernel_fun(x::AbstractVector, y::AbstractVector) = kern_lam(norm(x-y))
end

function get_derivs(kernel::Basic, trunc_param::Int64)
    @vars r
    derivs = [kernel]
    for i in 1:(2*trunc_param) # What limit is appropriate here?
        newdv = expand(diff(derivs[end], r))
        push!(derivs, newdv)
    end
    return derivs
end

# symbolic precomputation, independent of data
function fill_f_g_tables!(fact::MultipoleFactorization)
    @vars r, rprime
    dim = fact.tree.dimension
    alpha = dim//2 -1
    for k in 0:fact.trunc_param
        poly = 0
        for i in k:2:fact.trunc_param
            if i != 0
                tot = 0
                for m in 1:i
                    tot += fact.derivs[m+1]*r^(m-i+k+1)*transform_coef!(fact, i, k, m, alpha)
                end
                # println("k ",k," i ",i," poly ",expand(tot*rprime^(i-k)))
                fact.F_coef_table[k+1, i+1] = lambdify(tot, [r])
                fact.G_coef_table[k+1, i+1] = (r) -> r^(i-k)
                poly += tot*rprime^(i-k)
            else
                fact.F_coef_table[k+1, i+1] = lambdify(fact.derivs[1]*r, [r])
                fact.G_coef_table[k+1, i+1] = (r) -> r^(i-k)
                poly += fact.derivs[1]*r*rprime^(i-k)
            end
        end

        correction = fact.derivs[1]
        poly = expand(poly/correction)
        poly = expand(subs(poly, (correction/correction)=>1))
        lowest_pow_r = 0
        highest_pow_r = 0
        for i in -(100):(100)  # TODO smarter way to do these limits
            if coeff(poly, r, Basic(i)) != 0
                lowest_pow_r = i
                break
            end
        end
        for i in (100):-1:-(100)
            if coeff(poly, r, Basic(i)) != 0
                highest_pow_r = i
                break
            end
        end
        poly = expand(poly / (r^lowest_pow_r))
        # now lowest pow is 0
        mat = Matrix{Complex{Rational{BigInt}}}(undef, fact.trunc_param+1, max(highest_pow_r-lowest_pow_r, fact.trunc_param+1))

        mat[1, 1] = subs(poly, rprime=>0, r=>0)
        for i in 2:(max(highest_pow_r-lowest_pow_r, fact.trunc_param+1))
            tmp = coeff(subs(poly, r=>0), rprime, Basic(i-1))
            num = convert(Complex{BigInt}, numerator(tmp))
            den = convert(Complex{BigInt}, denominator(tmp))
            mat[1, i] = num//den
        end
        for i in 2:(fact.trunc_param+1)
            tmp = coeff(subs(poly, rprime=>0), r, Basic(i-1))
            num = convert(Complex{BigInt}, numerator(tmp))
            den = convert(Complex{BigInt}, denominator(tmp))
            mat[i, 1] = num//den
        end
        for i in 1:(max(highest_pow_r-lowest_pow_r, fact.trunc_param+1))
            for j in 1:(fact.trunc_param+1)
              tmp = coeff(coeff(poly, rprime, Basic(i-1)), r, Basic(j-1))
              num = convert(Complex{BigInt}, numerator(tmp))
              den = convert(Complex{BigInt}, denominator(tmp))
                mat[j, i] = num//den
            end
        end
        Qmat, Rmat, Pmat = rationalrrqr!(mat)
        Rmat = Rmat * transpose(Pmat)
        for i in 1:size(Rmat, 1)
            if @views all(==(0), Qmat[:, i]) || all(==(0), Rmat[i, :])
                println("Rank ",i-1," it seems")
                fact.radial_fun_ranks[k+1] = i-1
                for j in (i):div(fact.trunc_param-k+2,2)
                    fact.F_coef_table[k+1, (k+1)+ 2(j-1)] = (r) -> zero(r)
                    fact.G_coef_table[k+1, (k+1)+ 2(j-1)] = (r) -> zero(r)
                end
                break
            end
            r_poly = 0
            rprime_poly = 0
            for j in 1:size(Qmat, 1)
              rprime_poly += (rprime^(j-1))*convert(Float64,Rmat[i,j]) # TODO Float conversion smartest?
              r_poly += (r^(j-1))*convert(Float64,Qmat[j,i])
            end
            r_poly *= correction*(r^lowest_pow_r)
            fact.F_coef_table[k+1, (k+1)+ 2(i-1)] = lambdify(expand(r_poly), [r])
            fact.G_coef_table[k+1, (k+1) + 2(i-1)] = lambdify(expand(rprime_poly), [rprime])
        end
        new_poly = 0
        old_poly = 0
    end
end

function transform_coef!(fact, j, k, m, alpha) #TODO why do we have two of these?
    println("Called the wrong one")
    if (m < 1 || m > j || k > j || k < mod(j, 2) || isodd(k+j))
        println("Improper pair jkm: ", j, k, m)
        return -1
    end
    if fact.transform_coef_table[j+1, k+1, m+1] != -1
        return fact.transform_coef_table[j+1, k+1, m+1]
    end
    total = BigInt(0)
    for i in max(k,2m-j):2:j
        bigfact=BigInt(1)
        for p in 1:(i+j)
            numer = BigInt(1)
            denom = BigInt(1)
            if p <= (i+j-m-1)
                numer = p
            end
            if p <= i
                numer *= p
            end
            if p <= (m-1)
                denom *= p
            end
            if p <= (div(i+j,2)-m)
                denom *= p
            end
            if p <= div(i+j,2)
                denom *= p
            end
            if p <= div(i-k,2)
                denom *= p
            end
            bigfact *= (numer//denom)
        end
        powertwo=BigInt(2)
        if m-j-i > 0
            powertwo = BigInt(2^(m-j-i))
        else
            powertwo = 1//(BigInt(2^(-m+j+i)))
        end
        rising = 1
        prod_idx = alpha
        while prod_idx <= alpha + div(i+k,2)
            rising *= prod_idx
            prod_idx += 1
        end
        total += (
            ((-1)^div(3i+j+2m, 2))
            *powertwo
            * (1//rising)
            *(alpha + k)
            *binomial(div(i+j,2), div(j-i, 2))
            *bigfact
            )
    end
    fact.transform_coef_table[j+1, k+1, m+1] = total
    return total
end

# this could be pre-computed for common dimensions
function transformation_coefficients(d::Int, trunc_param::Int)
    alpha = d//2 - 1
    table = zeros(trunc_param+1, trunc_param+1, trunc_param+1)
    # table = [spzeros(trunc_param+1, trunc_param+1) for _ in 1:trunc_param+1]
    for index in CartesianIndices(table)
        j, k, m = index.I .- 1
        if (m < 1 || m > j || k > j || k < mod(j, 2) || isodd(k+j))
            continue
        end
        table[index] = transformation_coefficient(j, k, m, alpha)
        # table[j][k, m] = transformation_coefficient(j, k, m, alpha)
    end
    return table
end

function transformation_coefficient(j::Int, k::Int, m::Int, alpha::Real)
    total = BigInt(0)
    for i in max(k,2m-j):2:j
        bigfact = BigInt(1)
        for p in 1:(i+j)
            numer = BigInt(1)
            denom = BigInt(1)
            if p <= (i+j-m-1)
                numer = p
            end
            if p <= i
                numer *= p
            end
            if p <= (m-1)
                denom *= p
            end
            if p <= (div(i+j,2)-m)
                denom *= p
            end
            if p <= div(i+j,2)
                denom *= p
            end
            if p <= div(i-k,2)
                denom *= p
            end
            bigfact *= (numer//denom)
        end
        powertwo = BigInt(2)
        if m-j-i > 0
            powertwo = BigInt(2^(m-j-i))
        else
            powertwo = 1//(BigInt(2^(-m+j+i)))
        end
        rising = BigInt(1)
        prod_idx = alpha
        while prod_idx <= alpha + div(i+k,2)
            rising *= (prod_idx == 0 ? 1 : prod_idx)
            prod_idx += 1
        end
        total += (
            ((-1)^div(3i+j+2m, 2))
            *powertwo
            * (1//rising)
            *(alpha==0 ? 2 : (alpha+k))
            *binomial(div(i+j,2), div(j-i, 2))
            *bigfact
            )
    end
    if (alpha==0 && k==0) total/=2 end
    return total
end
