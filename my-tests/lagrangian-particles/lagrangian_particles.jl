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
# x₀ = zeros(Nparticles) .+ 1e-6*rand(Nparticles)
# y₀ = zeros(Nparticles) .+ 1e-6*rand(Nparticles)
# z₀ = zeros(Nparticles) .+ 1e-6*rand(Nparticles)
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

simulation = Simulation(model, Δt=10.0, stop_iteration=6000)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

b = model.tracers.b
# particles = model.particles
    simulation.output_writers[:particles] = 
                    NetCDFOutputWriter(model, model.particles, filename="my-tests/lagrangian-particles/particles.nc", schedule=IterationInterval(10))
simulation.output_writers[:buoyancy] = 
                NetCDFOutputWriter(model, (b=b,), filename="my-tests/lagrangian-particles/b.nc", schedule=IterationInterval(10))
checkpointer = Checkpointer(model,
                schedule = IterationInterval(6000),
                dir="my-tests/lagrangian-particles/",
                prefix = "lagrangian_particles",
                cleanup = false)

simulation.output_writers[:checkpointer] = checkpointer
progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(1))

run!(simulation,pickup=true)


using CairoMakie
using NCDatasets
using Printf

# Load particle data
fname = "my-tests/lagrangian-particles/particles.nc"
ds = Dataset(fname,"r")

x = ds["x"][:,:]
y = ds["y"][:,:]
z = ds["z"][:,:]
close(ds)

# Load buoyancy data
fname = "my-tests/lagrangian-particles/b.nc"
ds = Dataset(fname,"r")

# grids
xC = ds["xC"]
yC = ds["yC"]
t = ds["time"]

# Get buoyancy field
b = ds["b"][:,:,:,:]

# Create animation
n = Observable(1)
bₙ = @lift(b[:,:,end,$n]) # Take surface layer
xₙ = @lift(x[:,$n])
yₙ = @lift(y[:,$n])

fig = Figure(resolution = (800, 400), figure_padding=(10, 40, 10, 10), fontsize=20)
axis_kwargs = (xlabel = "x (m)",
              ylabel = "y (m)",
              limits = ((minimum(xC), maximum(xC)), (minimum(yC), maximum(yC))),
              aspect = 1)

title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax = Axis(fig[2, 1]; title = "Buoyancy and Particles", axis_kwargs...)

# Plot buoyancy field
hm = heatmap!(ax, xC[:], yC[:], bₙ,
              colormap = :thermal,
              nan_color = :gray)
Colorbar(fig[2,2], hm; label = "b (m/s²)")

# Plot particles
particles = scatter!(ax, xₙ, yₙ, color=:white, markersize=10)

frames = 1:length(t)
filename = "my-tests/lagrangian-particles/particles_animation"

record(fig, string(filename,".mp4"), frames, framerate=23) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds)




# Create 3D animation
n = Observable(1)
bₙ = @lift(b[:,:,:,$n])
xₙ = @lift(x[:,$n])
yₙ = @lift(y[:,$n])
zₙ = @lift(z[:,$n])

fig3D = Figure(resolution = (800, 600))
ax3D = Axis3(fig3D[1,1]; 
             xlabel = "x (m)",
             ylabel = "y (m)", 
             zlabel = "z (m)",
             )

title3D = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
fig3D[1, :] = Label(fig3D, title3D, fontsize=20, tellwidth=false)

# Create volume visualization
zC = ds["zC"][:]  # Get z coordinates
volumes = @lift begin
    # Normalize buoyancy for better visualization
    b_norm = ($bₙ .- minimum($bₙ)) ./ (maximum($bₙ) - minimum($bₙ))
    volume = b_norm
end

# Plot volume rendering
vol = volume!(ax3D, xC, yC, zC, volumes,
              colormap = :thermal,
              transparency = true,
              alpha = 0.6)
Colorbar(fig3D[1,2], vol, label="Normalized buoyancy")

# Plot particles as spheres
particles3D = scatter!(ax3D, xₙ, yₙ, zₙ, 
                      color = :black,
                      markersize = 15)

# Adjust camera
cam3D = cameracontrols(ax3D)
cam3d!(ax3D, elevation=0.7, azimuth=0.3, roll=0.0)
setperspective!(cam3D, 0.7)
rotate_cam!(ax3D, 0.7, 0.3, 0.0)

# Record animation
filename3D = "my-tests/lagrangian-particles/particles3D"
record(fig3D, string(filename3D,".mp4"), frames, framerate=23) do i
    @info "Plotting 3D frame $i of $(frames[end])..."
    n[] = i
    
    # Slowly rotate camera during animation
    # rotate_cam!(ax3D, 0.7, 0.3 + i*0.01, 0.0)
    # cam3d!(ax3D, elevation=0.7, azimuth=0.3 + i*0.01, roll=0.0)

end


