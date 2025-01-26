# this is a MWE for reproducing errors from using AveragedTimeInterval
using Oceananigans
using Printf

# rm("my-tests/checkpoint_test.nc")
""" Set up a simple simulation to test picking up from a checkpoint. """

    grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
    uᵢ(z) = 10
    T=2
    u_forcing(z, t) = 10*sin(2*pi/T*t)

    Δt = .01    # timestep (s)
    T1 = 1      # first simulation stop time (s)
    T2 = 2T1    # second simulation stop time (s)
    
    n_particles = 10  # number of particles to release
    x0 = zeros(n_particles)
    y0 = zeros(n_particles)
    z0 = collect(range(-0.9, 0.9, length=n_particles))
    # Create particles
    particles = LagrangianParticles(x=x0, y=y0, z=z0, restitution=0)

    model = NonhydrostaticModel(; grid, 
                                  forcing = (u = u_forcing,),
                                  timestepper = :RungeKutta3,
                                  particles=particles
                                  )

   
    set!(model, u=uᵢ)

    simulation = Simulation(model; Δt, stop_time=T1)
    u = model.velocities.u

    progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
    simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))

    simulation.output_writers[:timeavg] = NetCDFOutputWriter(model, (u=u,),
                                        filename = "my-tests/checkpoint-for-different-model/checkpoint_test.nc",
                                        schedule = TimeInterval(Δt),
                                        overwrite_existing = true)
                
    simulation.output_writers[:particles] = NetCDFOutputWriter(model, model.particles,
    filename = "my-tests/checkpoint-for-different-model/particles.nc",
    schedule = TimeInterval(Δt),
    overwrite_existing = true)

    checkpointer = Checkpointer(model,
                                schedule = TimeInterval(T1),
                                dir="my-tests/checkpoint-for-different-model/",
                                prefix = "checkpoint_test",
                                cleanup = true)

    simulation.output_writers[:checkpointer] = checkpointer


# run(`sh -c "rm test_iteration*.jld2"`)

# Run a simulation that saves data to a checkpoint
run!(simulation)

# change the model and see if picking up from the checkpoint works
# cd("my-tests/")

grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
uᵢ(z) = 10
T=2
u_forcing(z, t) = 10*sin(2*pi/T*t)

n_particles = 10  # number of particles to release
x0 = zeros(n_particles)
y0 = zeros(n_particles)
z0 = collect(range(-0.9, 0.9, length=n_particles))
# Create particles
particles = LagrangianParticles(x=x0, y=y0, z=z0, restitution=0)


f₀ = 1e-4
coriolis = ConstantCartesianCoriolis(f = f₀)
model = NonhydrostaticModel(; grid, 
                              forcing = (u = u_forcing,),
                              timestepper = :RungeKutta3,
                              coriolis = coriolis,
                              closure = SmagorinskyLilly(),
                              particles=particles
     )


set!(model, u=uᵢ)

simulation = Simulation(model; Δt, stop_time=T2)
u = model.velocities.u

progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))

simulation.output_writers[:timeavg] = NetCDFOutputWriter(model, (u=u,),
                                    filename = "my-tests/checkpoint-for-different-model/checkpoint_test.nc",
                                    schedule = TimeInterval(Δt),
                                    overwrite_existing = true)

simulation.output_writers[:particles] = NetCDFOutputWriter(model, model.particles,
                                    filename = "my-tests/checkpoint-for-different-model/particles.nc",
                                    schedule = TimeInterval(Δt),
                                    overwrite_existing = false)
                             
checkpointer = Checkpointer(model,
                            schedule = TimeInterval(T1),
                            prefix = "checkpoint_test",
                            dir="my-tests/checkpoint-for-different-model/",
                            overwrite_existing = true)

simulation.output_writers[:checkpointer] = checkpointer


# Run a simulation that saves data to a checkpoint
# run!(simulation,pickup="my-tests/checkpoint-for-different-model/checkpoint_test_iteration12.jld2")
run!(simulation,pickup=true)
# run!(simulation,pickup=11)
# cd("..")



# Visualization
using CairoMakie
using NCDatasets

# Load particle data
ds = Dataset("my-tests/checkpoint-for-different-model/particles.nc", "r")

# Get time steps and particle positions
times = ds["time"][:]
x = ds["x"][:,:]
y = ds["y"][:,:]
z = ds["z"][:,:]

n_frames = length(times)

# Create animation
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1],
          xlabel = "Time (s)",
          ylabel = "z",
          title = "Particle Trajectories")

limits!(ax, 0, T2, -1, 1)

# Initialize scatter plot
particles_plot = scatter!(ax, zeros(n_particles), zeros(n_particles),
                         color = :blue, markersize = 10)

# Create animation
record(fig, "my-tests/checkpoint-for-different-model/particle_animation.mp4", 1:n_frames; framerate = 30) do i
    # Extract z-coordinates for current time
    z_positions = z[:, i]
    # Update scatter plot
    # particles_plot[1] = (fill(times[i], n_particles), z_positions)
end

close(ds)

# println("Animation saved as 'particle_animation.mp4'")