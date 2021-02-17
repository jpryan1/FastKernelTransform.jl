# random data generators
uniform_data(n, d) = [rand(d) for _ in 1:n]
gaussian_data(n, d) = [randn(d) for _ in 1:n]

# data size n per cluster center
# dimension d
# number of centers c
# σ is std of data around centers

function min_dist_between_pts(centers)
    min_dist = Inf
    for i in 1:length(centers)
        for j in (i+1):length(centers)
            ci, cj = centers[i], centers[j]
            nij = norm(difference(ci, cj))
            if nij > 0 && nij < min_dist
                min_dist = nij
            end
        end
    end
    return min_dist
end


function gaussian_mixture_data(n::Int, c::Int, d::Int, σ::Real)
    iszero(mod(n, c)) || throw("n ($n) is not divisible by c ($c)")
    nc = Int(floor(n/c))
    centers = uniform_data(c, d)
    while(min_dist_between_pts(centers) < 2σ) centers = uniform_data(c, d) end
    data = zeros(d, nc*c)
    for (i, μ) in enumerate(centers)
        ind_i = nc*(i-1)+1:nc*i
        data[:, ind_i] = @. μ + σ * $randn(d, nc)
    end
    return [copy(c) for c in eachcol(data)]
end

# embeds data in d dimensional space
function embedded_data(data::AbstractVector{<:AbstractVector{<:Real}}, d::Int)
    di = length(data[1])
    E = randn(d, di) # random linear embedding
    return [E*x for x in data]
end

function two_bump_data(n::Int, d::Int, σ::Real = .1)
    iseven(n) || throw("n ($n) is not even")
    nc = n ÷ 2
    centers = [fill(1.0, d), fill(-1.0, d)]
    data = zeros(d, nc*2)
    for (i, μ) in enumerate(centers)
        ind_i = nc*(i-1)+1:nc*i
        data[:, ind_i] = @. μ + σ * $randn(d, nc)
    end
    return [copy(c) for c in eachcol(data)]
end
