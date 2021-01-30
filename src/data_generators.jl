# random data generators
uniform_data(n, d) = [rand(d) for _ in 1:n]
gaussian_data(n, d) = [randn(d) for _ in 1:n]

# data size n per cluster center
# dimension d
# number of centers c
# σ is std of data around centers
function gaussian_mixture_data(n::Int, c::Int, d::Int, σ::Real)
    nc = Int(floor(n/c))
    centers = gaussian_data(c, d)
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
