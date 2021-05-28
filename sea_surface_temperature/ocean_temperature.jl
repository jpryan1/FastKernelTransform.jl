# using CSV
# path = "Ocean Temperature Data/"
# filename = "IMMA1_R3.0.1_2020-07"
# f = open(path * filename)
# file = CSV.File(f, limit = 16)

using NetCDF
using Base.Threads

function get_sst_lvl4(file)
    # println(file)
    lat = ncread(file, "lat")
    lon = ncread(file, "lon")
    sst = ncread(file, "analysed_sst") # T in Kelvin
    sst_scale = ncgetatt(file, "analysed_sst", "scale_factor")
    sst_add_offset = ncgetatt(file, "analysed_sst", "add_offset") # this is 273.15 = |0K in C|
    mask = ncread(file, "mask")
    iswater = mask .== 1
    # island = mask .== 2
    sst = float.(sst) # float32??
    @. sst[!iswater] = NaN
    @. sst = sst_scale * sst # + sst_add_offset
    sst = @view sst[:, :, 1]

    analysis_uncertainty = ncread(file, "analysis_uncertainty")
    sig_scale = ncgetatt(file, "analysis_uncertainty", "scale_factor")
    sig_sst = @. sig_scale * float.(analysis_uncertainty)
    @. sig_sst[!iswater] = NaN
    sig_sst = @view sig_sst[:, :, 1]
    return lon, lat, sst, sig_sst
end

function get_all_sst_lvl_4(max_t::Int = 8)
    path = "Ocean Temperature Data/"
    folder = "dataset-satellite-sea-surface-temperature-second-half-2020/"
    files = readdir(path*folder)
    lon, lat, sst_1 = get_sst(path*folder*files[1])
    max_t = min(max_t, length(files))
    # lon = zeros(nlon, max_t)
    # lat = zeros(nlat, max_t)
    # @. lon[:, 1] = lon_1
    # @. lat[:, 1] = lat_1
    nlon, nlat = length(lon), length(lat)
    subsampling = 4
    if subsampling ≠ 1
        lon_i = 1:subsampling:nlon
        lat_i = 1:subsampling:nlat
        lon = lon[lon_i]
        lat = lat[lat_i]
        nlon, nlat = length(lon), length(lat)
        sst_1 = @view sst_1[lon_i, lat_i]
    end
    sst = zeros(nlon, nlat, max_t)
    @. sst[:, :, 1] = sst_1
    for i in 2:max_t
        file = path*folder*files[i]
        # lon[:, i], lat[:, i], sst[:, :, i] = get_sst(file)
        _, _, sst_i = get_sst(file)
        @. sst[:, :, i] = sst_i[lon_i, lat_i]
    end
    # f = isapprox(@view(lon[:, 1]))
    # if all(f, eachcol(lon))
    #     lon = lon[:, 1]
    # end
    # f = isapprox(@view(lat[:, 1]))
    # if all(f, eachcol(lat))
    #     lat = lat[:, 1]
    # end
    lon, lat, sst
end

function get_sst_lvl3(file)
    # println(file)
    lat = ncread(file, "lat")
    lon = ncread(file, "lon")
    temperature = ncread(file, "sea_surface_temperature") # T in Kelvin
    sst_scale = ncgetatt(file, "sea_surface_temperature", "scale_factor")
    sst_add_offset = ncgetatt(file, "sea_surface_temperature", "add_offset") # this is 273.15 = |0K in C|
    sst_fill = ncgetatt(file, "sea_surface_temperature", "_FillValue") # this is 273.15 = |0K in C|
    # mask = ncread(file, "mask")
    # iswater = mask .== 1
    # island = mask .== 2
    sst = float.(temperature) # float32??
    # @. sst[!iswater] = NaN
    @. sst = sst_scale * sst # + sst_add_offset
    sst = @view sst[:, :, 1]

    for i in eachindex(temperature)
        if temperature[i] == sst_fill
            sst[i] = NaN
        end
    end

    sst_dtime = ncread(file, "sst_dtime") # seconds
    time = ncread(file, "time") # seconds
    time = @. @views sst_dtime[:, :, 1] + time[1]
    # println(sst_dtime[1:16, 1:])

    uncertainty = ncread(file, "uncorrelated_uncertainty")
    sig_scale = ncgetatt(file, "uncorrelated_uncertainty", "scale_factor")
    sig_fill = ncgetatt(file, "uncorrelated_uncertainty", "_FillValue")
    sig_sst = @. sig_scale * float(uncertainty)

    for i in eachindex(uncertainty)
        if uncertainty[i] == sig_fill
            sig_sst[i] = NaN
        end
    end

    # only use high-quality data
    quality = ncread(file, "quality_level")
    for i in eachindex(quality)
        if quality[i] ≤ 4
            sst[i] = NaN
            sig_sst[i] = NaN
        end
    end
    println(sum(!isnan, sst), " number of data points")
    sig_sst = @view sig_sst[:, :, 1]
    return lon, lat, sst, sig_sst, time
end

function get_iswater_lvl4(file)
    mask = ncread(file, "mask")
    iswater = @views mask[:, :, 1] .== 1
end

# make a data set (single day)
function make_data(lon, lat, sst::AbstractMatrix, sig_sst::AbstractMatrix, time::AbstractMatrix)
    indices = findall(!isnan, sst)
    indices = CartesianIndices(sst)[indices]
    n = length(indices)
    x = zeros(3, n)
    y = zeros(n)
    variance = zeros(n)
    for (i, ind) in enumerate(indices)
        lon_ind, lat_ind = ind.I
        x[1, i] = lon[lon_ind]
        x[2, i] = lat[lat_ind]
        x[3, i] = time[ind]
        y[i] = sst[ind]
        variance[i] = sig_sst[ind]^2 # variance is square of standard deviation
    end
    # sort data by time
    sortind = sortperm(@view(x[3, :]))
    x = x[:, sortind]
    y = y[sortind]
    variance = variance[sortind]
    return x, y, variance
end

function make_data(lon, lat, sst::AbstractArray, sig_sst::AbstractArray, time::AbstractArray)
    x, y, variance = @views make_data(lon, lat, sst[:, :, 1], sig_sst[:, :, 1], time[:, :, 1])
    for d in 2:size(sst, 3)
        xd, yd, vd = @views make_data(lon, lat, sst[:, :, d], sig_sst[:, :, d], time[:, :, d])
        x = hcat(x, xd)
        y = vcat(y, yd)
        variance = vcat(variance, vd)
    end
    x, y, variance
end

path = "Ocean Temperature Data/"
level_3 = "Level 3/dataset-satellite-sea-surface-temperature-2019/"
dir = readdir(path * level_3)

# first, get water map
level_4 = "Level 4/dataset-satellite-sea-surface-temperature-first-half-2019/"
filename = readdir(path*level_4)[1]
iswater = get_iswater_lvl4(path * level_4 * filename)

# INFO: odd indices are from day, even from night
filename = dir[1] # "20200701120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.0-v02.0-fv01.0.nc"
# ncinfo(path * level_3 * filename)
lon, lat, sst, sig_sst, time = get_sst_lvl3(path*level_3*filename)
@. sst[!iswater] = NaN
@. sig_sst[!iswater] = NaN

println(sum(!isnan, sst))
println(sum(!isnan, sig_sst))

X, y, variance = @views make_data(lon, lat, sst, sig_sst, time)

# load multiple days
ndays = 31
for d in 2:ndays
    global X, y, variance
    local lon, lat, sst, sig_sst, time, filename
    println("getting data for day $d")
    filename = dir[2d-1] # "20200701120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.0-v02.0-fv01.0.nc"
    # ncinfo(path * level_3 * filename)
    @time lon, lat, sst, sig_sst, time = get_sst_lvl3(path*level_3*filename)
    @. sst[!iswater] = NaN
    @. sig_sst[!iswater] = NaN
    @time Xd, yd, vd = @views make_data(lon, lat, sst, sig_sst, time)
    X = hcat(X, Xd)
    y = vcat(y, yd)
    variance = vcat(variance, vd)
end

# rows in X are:
# - longitude (in degrees east) ± 180
# - lattitude (in degrees north) ± 90
# - time (in seconds since 1981-01-01 00:00:00)
# X, y, variance = make_data(lon, lat, sst, sig_sst, time)

dosave = true
using HDF5
if dosave
    file = h5open("oceanographic_data_$(ndays)_days.h5", "w")
    file["X"] = X
    file["y"] = y
    file["variance"] = variance
    file["ndays"] = ndays
    file["lon"] = lon
    file["lat"] = lat
    file["iswater"] = Int.(iswater) # WARNING: can't save it in bool
    close(file)
end

# using Plots
# plotly()

# lon, lat, sst, sig_sst = get_sst(path*filename)
# # -sst_add_offset # land data (innocuous hack to get plots dark)
# @. sst[!iswater] = -1000
# subsampling = 1
# lon_i = @view lon[1:subsampling:end]
# lat_i = @view lat[1:subsampling:end]
# sst_i = @view sst[1:subsampling:end, 1:subsampling:end]
# Plots.scalefontsizes(2.0)
#
# # @. sig_sst[isnan(sig_sst)] = 0
# heatmap(lon_i, lat_i, sst_i', legend = false, clims = (-3, 35),
#         xlabel = "longitude", ylabel = "latitude", # title = "Sea Surface Temperature",
#         colorbar_title = "degrees C") #, size = ())
# gui()
#
# savefig("SST_single_day_measurements.pdf")

# lon, lat, sst = get_all_sst_lvl_4()
# anim = @animate for t in 1:size(sst, 3)
#     sst_t = @view sst[:, :, t]
#     heatmap(lon, lat, sst_t', clims = (0, 36), title = "Sea Surface Temperature")
# end every 1
# gif(anim, "anim_fps15.gif", fps = 16)
