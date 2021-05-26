using HDF5
using Statistics
using Plots

# to read results
f = h5open("FKT_synthetic_experiments_N_scaling.h5")
sizes = read(f, "sizes")
dimensions = read(f, "dimensions")
gen_names = read(f, "generators")
factor_times = read(f, "factor_times")
fast_times = read(f, "fast_times")
lazy_times = read(f, "lazy_times")
close(f)

# 1. scaling with data size
fac = mean(factor_times, dims = 1) # mean over independent experiments
fas = mean(fast_times, dims = 1)
laz = mean(lazy_times, dims = 1)

comp = [sizes sizes .* log.(sizes) sizes.^2]
comp ./= comp[1, :]' * 1e2
comp_names = ["n", "n log(n)", "n^2"]

fac = fac[1, :, 1, :]
fas = fas[1, :, 1, :]
laz = laz[1, 1:end-1, 1, :]

plot(scale = :log10, legend = false)
# plot(xscale = :log10, legend = false)
plot!(sizes, fac)
plot!(sizes, fas)
plot!(sizes[1:end-1], laz)
plot!(sizes, comp, linestyle = :dash)
gui()

# 2. scaling with dimension
# to read results
f = h5open("FKT_synthetic_experiments_D_scaling.h5")
sizes = read(f, "sizes")
dimensions = read(f, "dimensions")
gen_names = read(f, "generators")
factor_times = read(f, "factor_times")
fast_times = read(f, "fast_times")
lazy_times = read(f, "lazy_times")
close(f)

# 1. scaling with data size
fac = mean(factor_times, dims = 1) # mean over independent experiments
fas = mean(fast_times, dims = 1)
laz = mean(lazy_times, dims = 1)

# comp = [sizes sizes .* log.(sizes) sizes.^2]
# comp ./= comp[1, :]' * 1e2
# comp_names = ["n", "n log(n)", "n^2"]

fac = fac[1, 1, :, :]
fas = fas[1, 1, :, :]
laz = laz[1, 1, :, :]

plot(yscale = :log10, legend = false)
# plot(xscale = :log10, legend = false)
plot!(dimensions, fac)
plot!(dimensions, fas)
# plot!(dimensions, laz)
# plot!(dimensions, comp, linestyle = :dash)
gui()
