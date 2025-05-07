# this is a MWE for reproducing errors from using AveragedTimeInterval
using Oceananigans
using Printf
# using PyPlot
if isfile("my-tests/save-diffusivity-Smag/viscosity.nc")
    rm("my-tests/save-diffusivity-Smag/viscosity.nc")
end
""" Set up a simple simulation to test picking up from a checkpoint. """

grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
uᵢ(z) = 0
T=2
u_forcing(z, t) = 10*sin(2*pi/T*t)

closure = (SmagorinskyLilly(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
buoyancy = BuoyancyForce(BuoyancyTracer())
model = NonhydrostaticModel(; grid, 
                              advection = WENO(),
                              forcing = (u = u_forcing,),
                              timestepper = :RungeKutta3,
                              buoyancy = buoyancy,
                              closure = closure,
                              tracers = :b
     )


     
set!(model, u=uᵢ)
Δt = 0.01
T1 = 0.5
simulation = Simulation(model; Δt, stop_time=T1)
u = model.velocities.u
νₑ = simulation.model.diffusivity_fields[1].νₑ
progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", sim.model.clock.time)
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))

simulation.output_writers[:timeavg] = NetCDFOutputWriter(model, (; u=u,νₑ=νₑ,),
                                    filename = "my-tests/save-diffusivity-Smag/viscosity.nc",
                                    schedule = TimeInterval(Δt),
                                    overwrite_existing = true)
                             
# checkpointer = Checkpointer(model,
#                             schedule = TimeInterval(T1),
#                             prefix = "checkpoint_test",
#                             dir="my-tests/checkpoint-for-different-model/",
#                             cleanup = false)

# simulation.output_writers[:checkpointer] = checkpointer


# Run a simulation that saves data to a checkpoint
# run!(simulation,pickup="checkpoint_test_iteration11.jld2")
run!(simulation,pickup=false)
# run!(simulation,pickup=11)
# cd("..")

