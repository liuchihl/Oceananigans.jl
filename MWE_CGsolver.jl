
using Printf
using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
using Oceananigans.Utils: prettytime
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary


suffix = "0.1days"

## Simulation parameters
Nx = 4
Ny = 4

tᶠ = 5days # simulation run time
Δtᵒ = 180minutes # interval for saving output
H = 3kilometers # vertical extent
L = 2*2H # horizontal extent

# uniform grid works without blowing up
z_faces_array = [0., 50., 100, 150.]
# nonuniform grid blows up
# z_faces_array = [0., 50., 100, 200.]

grid = RectilinearGrid(size=(Nx, Ny, length(z_faces_array)-1), 
        x = (0, L),
        y = (0, L), 
        z = z_faces_array,
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)
h = 200meters # topographic height
topography = zeros(Nx,Ny)
topography[Nx÷2,Ny÷2] = h
grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(topography))

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
U₀=0.025
uᵢ(x, y, z) = -U₀
vᵢ(x, y, z) = 0.

model = NonhydrostaticModel(
    grid = grid_immerse,
    pressure_solver = ConjugateGradientPoissonSolver(grid_immerse))

set!(model, u=uᵢ, v=vᵢ)

Δt = 30
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|u|: %.2e, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.u),maximum(abs, model.velocities.w)) 
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))
run!(simulation)
