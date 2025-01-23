# this is a MWE for reproducing errors from using AveragedTimeInterval
using Oceananigans
using Printf
using PyPlot
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner

# rm("my-tests/checkpoint_test.nc")
""" Set up a simple simulation to test picking up from a checkpoint. """

    grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
    uᵢ(z) = 0
    T=2
    u_forcing(z, t) = 10*sin(2*pi/T*t)

    Δt = .01    # timestep (s)
    T1 = .1      # first simulation stop time (s)
    T2 = 2T1    # second simulation stop time (s)
    
    model = NonhydrostaticModel(; grid, 
                                  forcing = (u = u_forcing,),
                                  timestepper = :RungeKutta3,
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
                                 
    checkpointer = Checkpointer(model,
                                schedule = TimeInterval(T1),
                                dir="my-tests/checkpoint-for-different-model/",
                                prefix = "checkpoint_test",
                                cleanup = true)

    simulation.output_writers[:checkpointer] = checkpointer


# run(`sh -c "rm test_iteration*.jld2"`)

# Run a simulation that saves data to a checkpoint
run!(simulation)

iter = iteration(simulation)

# change the model and see if picking up from the checkpoint works
# cd("my-tests/")

grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
uᵢ(z) = 0
T=2
u_forcing(z, t) = 10*sin(2*pi/T*t)


f₀ = 1e-4
coriolis = ConstantCartesianCoriolis(f = f₀)
model = NonhydrostaticModel(; grid, 
                              forcing = (u = u_forcing,),
                              timestepper = :RungeKutta3,
                              coriolis = coriolis,
                              closure = SmagorinskyLilly(),
                              pressure_solver = ConjugateGradientPoissonSolver(
                                grid; maxiter=100, preconditioner=AsymptoticPoissonPreconditioner(),
                                reltol=1e-10)
                
     )


set!(model, u=uᵢ)

simulation = Simulation(model; Δt, stop_time=T2)
u = model.velocities.u

progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))

simulation.output_writers[:timeavg] = NetCDFOutputWriter(model, (u=u,),
                                    filename = "my-tests/checkpoint-for-different-model/checkpoint_test.nc",
                                    schedule = TimeInterval(Δt),
                                    overwrite_existing = false)
                             
checkpointer = Checkpointer(model,
                            schedule = TimeInterval(T1),
                            prefix = "checkpoint_test",
                            dir="my-tests/checkpoint-for-different-model/",
                            cleanup = false)

simulation.output_writers[:checkpointer] = checkpointer


# Run a simulation that saves data to a checkpoint
# run!(simulation,pickup="checkpoint_test_iteration11.jld2")
run!(simulation,pickup=true)
# run!(simulation,pickup=11)
# cd("..")

