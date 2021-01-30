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

function init_F_G(kernel, dimension::Int, trunc_param::Int, doQR::Val{true})
    println("QR init ", kernel)
    f = transformation_coefficients(dimension, trunc_param, Rational{BigInt})
    F_coeff, G_coeff, radial_fun_ranks, correction, lowest_pow_r = compute_f_g_tables(kernel, f, dimension, trunc_param)
    function get_F(r::AbstractVector)
        val = similar(r)
        corr = correction.(r)
        function F(k::Int, i::Int) # returns F_ki applied to r
            @. val = 0
            FDict = F_coeff[k+1, i+1]
            for pow in keys(FDict)
                @. val += r ^ (pow +lowest_pow_r[k+1]) * FDict[pow]
            end
            return val .*= corr
        end
    end
    function get_G(rprime::AbstractVector)
        val = similar(rprime)
        function G(k::Int, i::Int) # returns G_ki applied to r
            @. val = 0
            GDict = G_coeff[k+1, i+1]
            for pow in keys(GDict)
                @. val += (rprime ^ (pow)) * GDict[pow]
            end
            return val
        end
    end
    return get_F, get_G, radial_fun_ranks # IDEA can we encapsulate radial_fun_ranks in get_F, get_G?
end

function compute_f_g_tables(kernel, transform_coef_table::AbstractArray, dimension::Int, trunc_param::Int)
    @vars r, rprime
    p = trunc_param
    # F_coeff = zeros(p+1, p+1, p+1) # TODO: make powers of r dimension the first for cache locality
    # G_coeff = zeros(p+1, p+1, p+1)
    # F_coeff = fill(Dict(), p+1, p+1)
    F_coeff = Array{Any, 2}(undef, p+1, p+1)
    G_coeff = Array{Any, 2}(undef, p+1, p+1)
    for i in 1:(p+1)
        for j in 1:(p+1)
            F_coeff[i,j] = Dict()
        end
    end

    # F_coeff[2,1][0]=0
    # G_coeff = fill(Dict(), p+1, p+1)
    for i in 1:(p+1)
        for j in 1:(p+1)
            G_coeff[i,j] = Dict()
        end
    end
    radial_fun_ranks = zeros(Int, p+1)
    lowest_pow_r = zeros(Int, p+1)

    sym_kernel = kernel(r) # TODO: make this smarter
    derivs = get_derivs(sym_kernel, p)
    sym_correction = derivs[1]
    correction = lambdify(sym_correction, [r])
    # correction = (r)->exp(-r)
    # sym_correction = correction(r)
    for k in 0:p
        poly = 0
        for i in k:2:p
            if i != 0
                tot = 0
                for m in 1:i
                    tot += derivs[m+1] * r^(m-i+k+1) * transform_coef_table[i+1, k+1, m+1]
                end
                poly += tot * rprime^(i-k)
            else
                poly += derivs[1] * r * rprime^(i-k)
            end
        end

        poly = expand(poly / sym_correction)
        poly = expand(subs(poly, (sym_correction / sym_correction)=>1))
        lowest_pow_r[k+1] = 0
        highest_pow_r = 0
        for i in -(100):(100)  # TODO smarter way to do these limits
            if coeff(poly, r, Basic(i)) != 0
                lowest_pow_r[k+1] = i
                break
            end
        end
        for i in (100):-1:-(100)
            if coeff(poly, r, Basic(i)) != 0
                highest_pow_r = i
                break
            end
        end
        poly = expand(poly / (r^lowest_pow_r[k+1]))
        # now lowest pow is 0
        mat = Matrix{Complex{Rational{BigInt}}}(undef, trunc_param+1, max(highest_pow_r - lowest_pow_r[k+1], trunc_param+1))
        mat[1, 1] = subs(poly, rprime=>0, r=>0)
        for i in 2:(max(highest_pow_r - lowest_pow_r[k+1], trunc_param+1))
            tmp = coeff(subs(poly, r=>0), rprime, Basic(i-1))
            num = convert(Complex{BigInt}, numerator(tmp))
            den = convert(Complex{BigInt}, denominator(tmp))
            mat[1, i] = num//den
        end
        for i in 2:(trunc_param+1)
            tmp = coeff(subs(poly, rprime=>0), r, Basic(i-1))
            num = convert(Complex{BigInt}, numerator(tmp))
            den = convert(Complex{BigInt}, denominator(tmp))
            mat[i, 1] = num//den
        end
        for i in 1:(max(highest_pow_r - lowest_pow_r[k+1], trunc_param+1))
            for j in 1:(trunc_param+1)
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
                radial_fun_ranks[k+1] = i-1
                break
            end
            r_poly = 0
            rprime_poly = 0
            for j in 1:size(Qmat, 1)
                r_pow = j-1 # + lowest_pow_r[k+1]
                rprime_pow = j-1
                F_coeff[k+1, (k+1)+ 2(i-1)][r_pow] = convert(Float64, Qmat[j, i])
                G_coeff[k+1, (k+1)+ 2(i-1)][rprime_pow] = convert(Float64, Rmat[i, j])
            end
        end
        new_poly = 0
        old_poly = 0
    end
    return F_coeff, G_coeff, radial_fun_ranks, correction, lowest_pow_r
end

# this could be pre-computed for common dimensions
function transformation_coefficients(dimension::Int, trunc_param::Int, T = Float64)
    alpha = dimension//2 - 1
    table = zeros(T, trunc_param+1, trunc_param+1, trunc_param+1)
    # table = [spzeros(trunc_param+1, trunc_param+1) for _ in 1:trunc_param+1]
    @sync for index in CartesianIndices(table)
        j, k, m = index.I .- 1
        if (m < 1 || m > j || k > j || k < mod(j, 2) || isodd(k+j)) continue end
        @spawn table[index] = transformation_coefficient(j, k, m, alpha)
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
    if (alpha == 0 && k == 0) total /= 2 end
    return total
end
