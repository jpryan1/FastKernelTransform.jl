include("gp.jl")

# load data
using HDF5
ndays = 7
file = h5open("oceanographic_data_$(ndays)_days.h5", "r")
X = read(file, "X")
y = read(file, "y")
variance = read(file, "variance")
ndays = read(file, "ndays")
lon = read(file, "lon")
lat = read(file, "lat")
iswater = read(file, "iswater")
iswater = Bool.(iswater)
close(file)

# preprocess data
using Statistics
# normalize data
X_normal = copy(X)
X_normal[1, :] ./= 180 # maximum(abs, X_normal[1, :]) # putting lat, lon in [-1, 1]
X_normal[2, :] ./= 90 # maximum(abs, X_normal[2, :])
secondsperday = 60*60*24
X_normal[3, :] .-= minimum(X_normal[3, :]) # begin time at 0
X_normal[3, :] ./= secondsperday # convert to days

mean_y, std_y = mean(y), std(y)
y_normal = @. (y - mean_y) / std_y
σ²_normal = @. variance / std_y^2

# subsampling
subsampling = 8 * ndays
# observation_indices = range(1, length(y_normal), length = 16_000)
observation_indices = 1:subsampling:length(y_normal)
Xi = X_normal[:, observation_indices]
yi = y_normal[observation_indices]
σ²i = σ²_normal[observation_indices]
# σ²i = fill(1e-2, length(yi))
@. σ²i = min(σ²i, 1e-4)

# lengthscales in x, y, and time
# lx, ly, lt = 1e-2, 1e-2, 3
# lx, ly, lt = 2e-2, 2e-2, 3
lx, ly, lt = 3e-2, 3e-2, 3 # seems to be best
lengthscales = [lx, ly, lt]
U = inv(Diagonal(lengthscales))

# pre-process data with scaling
@. Xi *= inv(lengthscales)
xi = [c for c in eachcol(Xi)] # convert to vector of vectors

# k = RQ(1.0) # is EQ with different lengthscales but still smooth
k = MaternP(1) # is only once differentiable, can model sharp transitions better
# k = CovarianceFunctions.ScaledInputKernel(k, U)

println("calculating conditional")
@time C = ConditionalMean(k, xi, yi, σ²i, fast = true)

# plot result
plot_subsampling = ceil(Int, 8 / sqrt(2))
lon_i = @view lon[1:plot_subsampling:end]
lat_i = @view lat[1:plot_subsampling:end]
t = ndays/2
xs_i = [[x/180, y/90, t] .* inv.(lengthscales) for x in lon_i, y in lat_i]

nlat = length(lat_i)
nlon = length(lon_i)

# don't need values above certain latutiude
latitude_limit = 60
above_lat = lat_i .> latitude_limit
below_lat = lat_i .< -latitude_limit
between_lat = @. latitude_limit ≥ abs(lat_i)

xs_i = xs_i[:, between_lat]
xs_i = vec(xs_i)

# don't need values above certain latutiude
ys_i = zeros(nlon, nlat)
@. ys_i[:, above_lat] = NaN
@. ys_i[:, below_lat] = NaN

println("calculating marginal")
@time ys_temp = marginal(C, xs_i) # IDEA: could blend out points that are not in water

# ys_temp = reshape(ys_temp, nlon, :)
ys_i[:, between_lat] .= reshape(ys_temp, nlon, :)

# scale up to original data range
@. ys_i = std_y * ys_i + mean_y

# paint in landmass
iswater_i = @view iswater[1:plot_subsampling:end, 1:plot_subsampling:end]
@. ys_i[.!iswater_i] = -1000

dosave = false
if dosave
    file = h5open("oceanographic_results_$(ndays)_days.h5", "w")
    file["alpha"] = C.α
    file["lengthscales"] = lengthscales
    file["prediction"] = ys_i
    file["subsampling"] = subsampling
    file["plot_subsampling"] = plot_subsampling
    file["lon_i"] = lon_i
    file["lat_i"] = lat_i
    file["t"] = t
    close(file)
    end

using Plots
plotly()
cmax = ceil(maximum(y)) + 1 # 36
cmin = floor(minimum(y)) - 1 # -5

# Plots.scalefontsizes(2.0)
heatmap(lon_i, lat_i, ys_i', clims = (cmin, cmax),
        xlabel = "longitude", ylabel = "latitude", # title = "Sea Surface Temperature",
        colorbar_title = "degrees C", legend = false)
savefig("SST_Matern_7day_craziness.pdf")
