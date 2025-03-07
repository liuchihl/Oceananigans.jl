using Random
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hour
# using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner
using StructArrays: StructArray
# Stretched grid
Lz = 4          # (m) domain depth
Lx = 16     # domain width    
Ly = 1
Nx = 64
Ny=1
Nz = 40          # number of points in the vertical direction

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Vertically-stretched and uniform options
z_stretched(k) = Lz * (ζ₀(k) * Σ(k) - 1)
z_uniform = (-Lz, 0)

grid = RectilinearGrid(; size = (Nx, Nz), halo=(4, 4),
                       x = (-Lx/2, Lx/2),
                    #    y = (-Ly/2, Lx/2),
                       z = z_stretched,topology = (Bounded, Flat, Bounded))

@info "Build a grid:"
@show grid

# Create an immersed boundary grid with a rectangular obstacle
# Define a rectangular shape from the surface to the bottom
topo = ones(Nx,Ny)*-Lz
# topo[1:Nx,1] .= -4
topo[Nx÷2-1:Nx÷2+1,1] .= -1
topo[Nx÷4-1:Nx÷4+1,1] .= -1
topo[3Nx÷4-1:3Nx÷4+1,1] .= -1
# Create immersed boundary grid
grid= ImmersedBoundaryGrid(grid, GridFittedBottom(topo))


restitution = 1  # Restitution coefficient for particle collisions

part=true
# 10 Lagrangian particles
Random.seed!(123)  # Set a fixed seed for reproducibility
Nparticles = 30
x₀ = Lx / 10 * (2rand(Nparticles) .- 1)
y₀ = Ly / 10 * (2rand(Nparticles) .- 1)
z₀ = - Lz / 10 * rand(Nparticles)
b = 1e-5*ones(Nparticles)
u = zeros(Nparticles)
w = zeros(Nparticles)

struct CustomParticle
    x::Float64  # x-coordinate
    y::Float64  # y-coordinate
    z::Float64  # z-coordinate
    b::Float64  # buoyancy
    u::Float64  
    w::Float64  
end

lagrangian_particles = StructArray{CustomParticle}((x₀, y₀, z₀, b, u, w));

# Define tracked fields as a NamedTuple
tracers = (; b=CenterField(grid), c=CenterField(grid))
# tracers = (; b=CenterField(grid))
tracked_fields = (; b=tracers.b)

particles = LagrangianParticles(lagrangian_particles; tracked_fields=tracked_fields, restitution=restitution)
cᵢ(x, z) = 1 * z + 1e-9 * rand()

# Convection
b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))
model = NonhydrostaticModel(; grid, particles,
            # pressure_solver = ConjugateGradientPoissonSolver(grid; 
            # maxiter=100, preconditioner=AsymptoticPoissonPreconditioner(),
            # reltol=tol),
            advection = WENO(),
            timestepper = :RungeKutta3,
            tracers = part==true ? tracers : :b,
            buoyancy = BuoyancyTracer(),
            closure = AnisotropicMinimumDissipation(),
            boundary_conditions = (; b=b_bcs))

@info "Constructed a model"
@show model

bᵢ(x, z) = 1e-5 * z + 1e-9 * rand()
tf = 5000
# set!(model, b=bᵢ)
set!(model, "my-tests/lagrangian-particles/lagrangian_particles_iteration2500.jld2",
)
set!(model, c=cᵢ)

simulation = Simulation(model, Δt=10, stop_iteration=tf)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

b = model.tracers.b
c = model.tracers.c
simulation.output_writers[:particles] = 
                    NetCDFOutputWriter(model, model.particles, filename=string("my-tests/lagrangian-particles/particles_immerse_",tf,"_restitution=",restitution,".nc"), schedule=IterationInterval(10),
                    overwrite_existing=true)
simulation.output_writers[:buoyancy] = 
                NetCDFOutputWriter(model, (b=b,c=c), filename=string("my-tests/lagrangian-particles/b_immerse_",tf,"_restitution=",restitution,".nc"), schedule=IterationInterval(10),
                overwrite_existing=true)
simulation.output_writers[:checkpointer] = Checkpointer(model,
                schedule = IterationInterval(tf),
                dir="my-tests/lagrangian-particles/",
                prefix = "lagrangian_particles",
                cleanup = false)

# simulation.output_writers[:checkpointer] = checkpointer
progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(1))

run!(simulation,pickup=false)


using CairoMakie
using NCDatasets
using Printf
tf=5000
fname = string("my-tests/lagrangian-particles/particles_immerse_",tf,"_restitution=",restitution,".nc")
ds_par = Dataset(fname,"r")

x = ds_par["x"][:,:]
y = ds_par["y"][:,:]
z = ds_par["z"][:,:]
b_par = ds_par["b"][:,:]
u_par = ds_par["u"][:,:]
w_par = ds_par["w"][:,:]
close(ds_par)

# Load buoyancy data
fname = string("my-tests/lagrangian-particles/b_immerse_",tf,"_restitution=",restitution,".nc")
# fname = string("my-tests/lagrangian-particles/b_immerse_restitution=",restitution,".nc")
ds = Dataset(fname,"r")

# grids
xC = ds["xC"]
yC = ds["yC"]
t = ds["time"]

# Get buoyancy field
b = ds["b"][:,:,:,:]
b[b.==0] .= NaN
n = Observable(1)
bxzₙ = @lift(b[:,Ny,:,$n]) # Take center slice
xₙ = @lift(x[:,$n])
yₙ = @lift(y[:,$n])
zₙ = @lift(z[:, $n])

if tf!==5000
    fig = Figure(resolution = (900, 500), figure_padding=(10, 40, 10, 10), fontsize=20)
    axis_kwargs = (xlabel = "x (m)",
                aspect = 1)

    title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
    fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

    # First column: x-z view of buoyancy
    ax1 = Axis(fig[2, 1]; title = "Buoyancy (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(ds["zC"][:]), 0)),
            axis_kwargs...)
    # Calculate global min/max for buoyancy and concentration
    b_global_min = minimum(filter(!isnan, b))
    b_global_max = maximum(filter(!isnan, b))
    # Plot buoyancy field in x-z plane
    hm1 = heatmap!(ax1, xC[:], ds["zC"][:], bxzₙ,
                colormap = :thermal,
                colorrange = (b_global_min, b_global_max),
                nan_color = :gray)
    # Add colorbars
    Colorbar(fig[2, 3], hm1, label = "b (m/s²)")

    # Plot particles in both views
    particles_xz1 = scatter!(ax1, xₙ, zₙ, color=:black, markersize=10)

else
    c = ds["c"][:,:,:,:]
    c[c.==0] .= NaN
    # Create animation
   
    cxzₙ = @lift(c[:,Ny,:,$n]) # Take center slice
    fig = Figure(resolution = (900, 500), figure_padding=(10, 40, 10, 10), fontsize=20)
    axis_kwargs = (xlabel = "x (m)",
                aspect = 1)

    title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
    fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

    # First column: x-z view of buoyancy
    ax1 = Axis(fig[2, 1]; title = "Buoyancy (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(ds["zC"][:]), 0)),
            axis_kwargs...)

    # Second column: x-z view of concentration
    ax2 = Axis(fig[2, 2]; title = "Concentration (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(ds["zC"][:]), 0)),
            axis_kwargs...)

    # Calculate global min/max for buoyancy and concentration
    b_global_min = minimum(filter(!isnan, b))
    b_global_max = maximum(filter(!isnan, b))
    c_global_min = minimum(filter(!isnan, c))
    c_global_max = maximum(filter(!isnan, c))

    # Plot buoyancy field in x-z plane
    hm1 = heatmap!(ax1, xC[:], ds["zC"][:], bxzₙ,
                colormap = :thermal,
                colorrange = (b_global_min, b_global_max),
                nan_color = :gray)

    # Plot concentration field in x-z plane
    hm2 = heatmap!(ax2, xC[:], ds["zC"][:], cxzₙ,
                colormap = :viridis,
                colorrange = (c_global_min, c_global_max),
                nan_color = :gray)

    # Add colorbars
    Colorbar(fig[2, 3], hm1, label = "b (m/s²)")
    Colorbar(fig[2, 4], hm2, label = "c (concentration)")

    # Plot particles in both views
    particles_xz1 = scatter!(ax1, xₙ, zₙ, color=:black, markersize=10)
    particles_xz2 = scatter!(ax2, xₙ, zₙ, color=:black, markersize=10)

end
frames = 1:length(t)
filename = string("my-tests/lagrangian-particles/particles_animation_immerse_tf=",tf,"_restitution=",restitution)

record(fig, string(filename,".mp4"), frames, framerate=23) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
close(ds)



# Create 3D animation
# n = Observable(1)
# bₙ = @lift(b[:,:,:,$n])
# xₙ = @lift(x[:,$n])
# yₙ = @lift(y[:,$n])
# zₙ = @lift(z[:,$n])

# fig3D = Figure(resolution = (800, 600))
# ax3D = Axis3(fig3D[1,1]; 
#              xlabel = "x (m)",
#              ylabel = "y (m)", 
#              zlabel = "z (m)",
#              )

# title3D = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
# fig3D[1, :] = Label(fig3D, title3D, fontsize=20, tellwidth=false)

# # Create volume visualization
# zC = ds["zC"][:]  # Get z coordinates
# volumes = @lift begin
#     # Normalize buoyancy for better visualization
#     b_norm = ($bₙ .- minimum($bₙ)) ./ (maximum($bₙ) - minimum($bₙ))
#     volume = b_norm
# end

# # Plot volume rendering
# vol = volume!(ax3D, xC, yC, zC, volumes,
#               colormap = :thermal,
#               transparency = true,
#               alpha = 0.6)
# Colorbar(fig3D[1,2], vol, label="Normalized buoyancy")

# # Plot particles as spheres
# particles3D = scatter!(ax3D, xₙ, yₙ, zₙ, 
#                       color = :black,
#                       markersize = 15)

# # Adjust camera
# cam3D = cameracontrols(ax3D)
# cam3d!(ax3D, elevation=0.7, azimuth=0.3, roll=0.0)
# setperspective!(cam3D, 0.7)
# rotate_cam!(ax3D, 0.7, 0.3, 0.0)

# # Record animation
# filename3D = "my-tests/lagrangian-particles/particles3D"
# record(fig3D, string(filename3D,".mp4"), frames, framerate=23) do i
#     @info "Plotting 3D frame $i of $(frames[end])..."
#     n[] = i
    
#     # Slowly rotate camera during animation
#     # rotate_cam!(ax3D, 0.7, 0.3 + i*0.01, 0.0)
#     # cam3d!(ax3D, elevation=0.7, azimuth=0.3 + i*0.01, roll=0.0)

# end


