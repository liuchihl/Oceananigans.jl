using Random
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hour
using StructArrays: StructArray

# ------ Common configurations for both simulations ------
# Stretched grid
Lz = 4          # (m) domain depth
Lx = 16         # domain width    
Ly = 1
Nx = 64
Ny = 1
Nz = 40         # number of points in the vertical direction

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Vertically-stretched and uniform options
z_stretched(k) = Lz * (ζ₀(k) * Σ(k) - 1)
z_uniform = (-Lz, 0)

# Create base grid
grid = RectilinearGrid(; size = (Nx, Nz), halo=(4, 4),
                      x = (-Lx/2, Lx/2),
                      z = z_stretched, topology = (Bounded, Flat, Bounded))

@info "Build a grid:"
@show grid

# Create an immersed boundary grid with a rectangular obstacle
# Define a rectangular shape from the surface to the bottom
topo = ones(Nx,Ny)*-Lz
topo[Nx÷2-1:Nx÷2+1,1] .= -1
topo[Nx÷4-1:Nx÷4+1,1] .= -1
topo[3Nx÷4-1:3Nx÷4+1,1] .= -1

# Create immersed boundary grid
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(topo))

# ------ First simulation (without particles) ------
function run_first_simulation()
    @info "Running first simulation (without particles)"
    
    # Define tracers with only buoyancy
    tracers = (; b=CenterField(grid))
    
    # Convection boundary conditions
    b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))
    
    # Create model without particles
    model = NonhydrostaticModel(; grid,
                advection = WENO(),
                timestepper = :RungeKutta3,
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                closure = AnisotropicMinimumDissipation(),
                boundary_conditions = (; b=b_bcs))
    
    @info "Constructed model for first simulation"
    @show model
    
    # Initialize buoyancy
    bᵢ(x, z) = 1e-5 * z + 1e-9 * rand()
    set!(model, b=bᵢ)
    
    # Set up simulation
    iterations = 2500  # Adjust as needed
    simulation = Simulation(model, Δt=10, stop_iteration=iterations)
    
    # Add time step wizard
    wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    
    # Add buoyancy output writer
    b = model.tracers.b
    simulation.output_writers[:buoyancy] = 
                    NetCDFOutputWriter(model, (b=b,), 
                                      filename="my-tests/lagrangian-particles/b_first_sim.nc", 
                                      schedule=IterationInterval(10),
                                      overwrite_existing=true)
    
    # Add checkpointer to save pickup file
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                    schedule = IterationInterval(iterations),
                    dir="my-tests/lagrangian-particles/",
                    prefix = "first_sim",
                    cleanup = false)
    
    # Add progress callback
    progress_message(sim) = @info string("First sim - Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))
    
    # Run simulation
    run!(simulation)
    
    @info "First simulation completed"
    return iterations
end

    # Define custom particle structure
    struct CustomParticle
        x::Float64  # x-coordinate
        y::Float64  # y-coordinate
        z::Float64  # z-coordinate
        b::Float64  # buoyancy
    end
# ------ Second simulation (with particles) ------
function run_second_simulation(pickup_iteration)
    @info "Running second simulation (with particles)"
    

    
    # Step 1: Load the pickup file using a temporary model without particles
    @info "Creating temporary model to load pickup file"
    
    # Define tracers with buoyancy only (matching the first simulation)
    temp_tracers = (; b=CenterField(grid))
    
    # Convection boundary conditions
    b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-8))
    
    # Create temporary model without particles (same as first simulation)
    temp_model = NonhydrostaticModel(; grid,
                advection = WENO(),
                timestepper = :RungeKutta3,
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                closure = AnisotropicMinimumDissipation(),
                boundary_conditions = (; b=b_bcs))
    
    # Load the pickup file into temporary model
    pickup_file = "my-tests/lagrangian-particles/first_sim_iteration$(pickup_iteration).jld2"
    @info "Loading pickup file into temporary model: $pickup_file"
    set!(temp_model, pickup_file)
    
    # Step 2: Now create the actual model with particles
    @info "Creating actual model with particles"
    
    # Initialize particles
    restitution = 1  # Restitution coefficient for particle collisions
    
    Random.seed!(123)  # Set a fixed seed for reproducibility
    Nparticles = 30
    x₀ = Lx / 10 * (2rand(Nparticles) .- 1)
    y₀ = Ly / 10 * (2rand(Nparticles) .- 1)
    z₀ = - Lz / 10 * rand(Nparticles)
    b = 1e-5*ones(Nparticles)
    u = zeros(Nparticles)
    w = zeros(Nparticles)
    
    lagrangian_particles = StructArray{CustomParticle}((x₀, y₀, z₀, b))
    
    # Define tracers and tracked fields for the model with particles
    tracers = (; b=CenterField(grid), c=CenterField(grid))
    tracked_fields = (; b=tracers.b)
    
    # Create particles
    particles = LagrangianParticles(lagrangian_particles; tracked_fields=tracked_fields, restitution=restitution)
    
    # Create model with particles
    model = NonhydrostaticModel(; grid, particles,
                advection = WENO(),
                timestepper = :RungeKutta3,
                tracers = tracers,
                buoyancy = BuoyancyTracer(),
                closure = AnisotropicMinimumDissipation(),
                boundary_conditions = (; b=b_bcs))
    
    @info "Constructed model for second simulation"
    
    # Step 3: Copy the fields and velocities from temp_model to model
    @info "Copying fields from temporary model to particle model"
    copyto!(model.velocities.u, temp_model.velocities.u)
    copyto!(model.velocities.v, temp_model.velocities.v)
    copyto!(model.velocities.w, temp_model.velocities.w)
    copyto!(model.tracers.b, temp_model.tracers.b)
    
    # Set the clock to match
    model.clock.time = temp_model.clock.time
    model.clock.iteration = temp_model.clock.iteration
    
    # Set concentration field (not in pickup)
    cᵢ(x, z) = 1 * z + 1e-9 * rand()
    set!(model, c=cᵢ)
    
    # Set up simulation
    tf = 5000
    simulation = Simulation(model, Δt=10, stop_iteration=tf)
    
    # Add time step wizard
    wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    
    # Access fields
    b = model.tracers.b
    c = model.tracers.c
    
    # Add output writers
    simulation.output_writers[:particles] = 
                    NetCDFOutputWriter(model, model.particles, 
                                      filename=string("my-tests/lagrangian-particles/particles_immerse_",tf,"_restitution=",restitution,".nc"), 
                                      schedule=IterationInterval(10),
                                      overwrite_existing=true)
    
    simulation.output_writers[:buoyancy] = 
                    NetCDFOutputWriter(model, (b=b,c=c), 
                                      filename=string("my-tests/lagrangian-particles/b_immerse_",tf,"_restitution=",restitution,".nc"), 
                                      schedule=IterationInterval(10),
                                      overwrite_existing=true)
    
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                    schedule = IterationInterval(tf),
                    dir="my-tests/lagrangian-particles/",
                    prefix = "lagrangian_particles",
                    cleanup = false)
    
    # Add progress callback
    progress_message(sim) = @info string("Second sim - Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))
    
    # Run simulation
    run!(simulation)
    
    @info "Second simulation completed"
end

# ------ Main execution ------
# Run first simulation and get the iteration number of the pickup file
pickup_iteration = run_first_simulation()

# Run second simulation with the pickup file
run_second_simulation(pickup_iteration)

# ------ Post-simulation animation ------

using CairoMakie
using NCDatasets
    @info "Creating animation from simulation outputs"
    
    # Define paths to files
    fields_file1 = "my-tests/lagrangian-particles/b_first_sim.nc"
    particles_file2 = "my-tests/lagrangian-particles/particles_immerse_5000_restitution=1.nc"
    fields_file2 = "my-tests/lagrangian-particles/b_immerse_5000_restitution=1.nc"
    
    # Load data
    
    fields1 = NCDataset(fields_file1)
    fields2 = NCDataset(fields_file2)
    particles2 = NCDataset(particles_file2)
    # Extract coordinates
    xC = fields1["xC"][:]
    zC = fields1["zC"][:]
    t1 = fields1["time"][:]
    t2 = fields2["time"][:]
    Ny=1
    # Extract particle data
    p_times = particles2["time"][:]
    p_x = particles2["x"][:, :]  # Already 2D matrix [particles, time]
    p_z = particles2["z"][:, :]  # Already 2D matrix [particles, time]
    
    # Extract field data
    b1 = fields1["b"][:, 1:1, :, :]
    b2 = fields2["b"][:, 1:1, :, :]
    c2 = fields2["c"][:, 1:1, :, :]
    
b1[b1.==0] .= NaN
b2[b2.==0] .= NaN
n = Observable(1)
bxz1ₙ = @lift(b1[:,Ny,:,$n]) # Take center slice
bxz2ₙ = @lift(b2[:,Ny,:,$n]) # Take center slice
cxz2ₙ = @lift(c2[:,Ny,:,$n]) # Take center slice
xₙ = @lift(p_x[:,$n])
zₙ = @lift(p_z[:, $n])

tf=2500
restitution=1
if tf!==5000
    fig = Figure(resolution = (900, 500), figure_padding=(10, 40, 10, 10), fontsize=20);
    axis_kwargs = (xlabel = "x (m)",
                aspect = 1)

    title = @lift @sprintf("t=%1.2f hrs", t1[$n]/3600)
    fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false);

    # First column: x-z view of buoyancy
    ax1 = Axis(fig[2, 1]; title = "Buoyancy (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(zC[:]), 0)),
            axis_kwargs...)
    # Calculate global min/max for buoyancy and concentration
    b_global_min = minimum(filter(!isnan, b1))
    b_global_max = maximum(filter(!isnan, b1))
    # Plot buoyancy field in x-z plane
    hm1 = heatmap!(ax1, xC[:], zC[:], bxz1ₙ,
                colormap = :thermal,
                colorrange = (b_global_min, b_global_max),
                nan_color = :gray)
    # Add colorbars
    Colorbar(fig[2, 3], hm1, label = "b (m/s²)")

    # Plot particles in both views
else
    c2[c2.==0] .= NaN
    # Create animation
   
    cxzₙ = @lift(c2[:,Ny,:,$n]) # Take center slice
    fig = Figure(resolution = (900, 500), figure_padding=(10, 40, 10, 10), fontsize=20);
    axis_kwargs = (xlabel = "x (m)",
                aspect = 1)

    title = @lift @sprintf("t=%1.2f hrs", t2[$n]/3600)
    fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

    # First column: x-z view of buoyancy
    ax1 = Axis(fig[2, 1]; title = "Buoyancy (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(zC[:]), 0)),
            axis_kwargs...)

    # Second column: x-z view of concentration
    ax2 = Axis(fig[2, 2]; title = "Concentration (x-z plane)",
            ylabel = "z (m)",
            limits = ((minimum(xC), maximum(xC)), (minimum(zC[:]), 0)),
            axis_kwargs...)

    # Calculate global min/max for buoyancy and concentration
    b_global_min = minimum(filter(!isnan, b2))
    b_global_max = maximum(filter(!isnan, b2))
    c_global_min = minimum(filter(!isnan, c2))
    c_global_max = maximum(filter(!isnan, c2))

    # Plot buoyancy field in x-z plane
    hm1 = heatmap!(ax1, xC[:], zC[:], bxz2ₙ,
                colormap = :thermal,
                colorrange = (b_global_min, b_global_max),
                nan_color = :gray)

    # Plot concentration field in x-z plane
    hm2 = heatmap!(ax2, xC[:], zC[:], cxz2ₙ,
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
frames = 1:length(t2)
filename = string("my-tests/lagrangian-particles/particles_animation_immerse_tf=",tf,"_restitution=",restitution)

record(fig, string(filename,".mp4"), frames, framerate=23) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end