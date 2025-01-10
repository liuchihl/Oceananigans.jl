using Random
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hour

# Stretched grid
Lz = 32          # (m) domain depth
Lx = Ly = 64     # domain width    
Nx = Ny = 32
Nz = 24          # number of points in the vertical direction

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Vertically-stretched and uniform options
z_stretched(k) = Lz * (ζ₀(k) * Σ(k) - 1)
z_uniform = (-Lz, 0)

grid = RectilinearGrid(; size = (Nx, Ny, Nz), halo=(3, 3, 3),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Lx/2),
                       z = z_stretched)

@info "Build a grid:"
@show grid

# 10 Lagrangian particles
Nparticles = 10
x₀ = Lx / 10 * (2rand(Nparticles) .- 1)
y₀ = Ly / 10 * (2rand(Nparticles) .- 1)
z₀ = - Lz / 10 * rand(Nparticles)
particles = LagrangianParticles(x=x₀, y=y₀, z=z₀, restitution=0)

@info "Initialized Lagrangian particles"
@show particles

# Convection
b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))

model = NonhydrostaticModel(; grid, particles,
                            advection = UpwindBiased(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (; b=b_bcs))

@info "Constructed a model"
@show model

bᵢ(x, y, z) = 1e-5 * z + 1e-9 * rand()
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=10.0, stop_iteration=100)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

b = model.tracers.b
particles = model.particles
simulation.output_writers[:particles] = 
                NetCDFOutputWriter(model, model.particles, filename="my-tests/particles.nc", schedule=IterationInterval(10))
simulation.output_writers[:buoyancy] = 
                NetCDFOutputWriter(model, (b=b,), filename="my-tests/b.nc", schedule=IterationInterval(10))


run!(simulation)



using CairoMakie
using NCDatasets
using Printf

fname = "my-tests/particles.nc"
ds = Dataset(fname,"r")

x = ds["x"][:,:]
y = ds["y"][:,:]
z = ds["z"][:,:]

fname = "my-tests/b.nc"
ds = Dataset(fname,"r")

# grids
zC = ds["zC"]; Nz=length(zC)
zF = ds["zF"]; #Nz=length(zF)
xC = ds["xC"]; Nx=length(xC)
xF = ds["xF"];

yC = ds["yC"]; Ny=length(yC)
t = ds["time"];

u = ds["u"][:,:,:,:];
# w = ds["w"][:,:,:,:];
udiv = ds["udiv"][:,:,:,:];

u_center = (u[:,:,:,:].+vcat(u[2:end,:,:,:], u[1:1,:,:,:]))./2
# w_center = (w[:,:,1:end-1,:].+w[:,:,2:end,:])./2
u_center[u_center.==0].=NaN
# w_center[w_center.==0].=NaN
# w[w.==0].=NaN
u[u.==0].=NaN


# plot
n = Observable(1)
uₙ = @lift(u_center[:,1,:,$n])
# wₙ = @lift(w_center[:,1,:,$n])
udivₙ = @lift(udiv[:,1,:,$n])

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800),fontsize=20)
axis_kwargs = (xlabel = "x (m)",
                  ylabel = "z (m)",
                  limits = ((0, ds["xF"][end]), (0, ds["zF"][end])),
                  )
title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                  
                  
ax_u = Axis(fig[2, 1]; title = "u", axis_kwargs...)
# ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
ax_udiv = Axis(fig[3, 1]; title = L"∇⋅\vec{u}", axis_kwargs...)



using ColorSchemes
U₀ = 0.01
hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
# hm_w = heatmap!(ax_w, xC[:], zC[:], wₙ,
#     colorrange = (-U₀, U₀), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
#     nan_color = :gray)
# Colorbar(fig[3,2], hm_w; label = "m/s")

hm_udiv = heatmap!(ax_udiv, xC[:], zC[:], udivₙ,
    colorrange = (-1e-8,1e-8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[3,2], hm_udiv; label = "1/s")


frames =  (1:length(t))

filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=23) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
