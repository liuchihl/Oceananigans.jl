# using Printf
# using Oceananigans
# using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
# using Oceananigans.Utils: prettytime
# using Oceananigans.Units
# using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

# suffix = "0.1days"

# ## Simulation parameters
# Nx = 4
# Ny = 4

# tᶠ = 5days # simulation run time
# Δtᵒ = 180minutes # interval for saving output
# H = 600meters # vertical extent
# L = 2*2H # horizontal extent

# # # uniform grid works without blowing up
# # z_faces_array = [0., 200., 400., 600.]
# # # nonuniform grid blows up
# z_faces_array = [0., 50., 200., 600.]

# grid = RectilinearGrid(size=(Nx, Ny, length(z_faces_array)-1), 
#         x = (0, L),
#         y = (0, L), 
#         z = z_faces_array,
#         halo = (4,4,4),
#         topology = (Periodic, Periodic, Bounded)
# )

# h = 200meters # topographic height
# topography = zeros(Nx,Ny)
# topography[Nx÷2,Ny÷2] = h
# grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(topography))

# # IC such that flow is in phase with predicted linear response, but otherwise quiescent
# U₀ = 0.025
# uᵢ(x, y, z) = -U₀
# vᵢ(x, y, z) = 0.
# preconditioner = fft_poisson_solver(grid)

# # Create an array to store results
# results = Dict{Float64, Union{Float64, String}}()

# # Test different regularizer values
# regularizer_values = -1:5:1000
# for reg_value in regularizer_values
#     @info "Testing regularizer value: $reg_value"
    
#     try
#         model = NonhydrostaticModel(
#             grid = grid_immerse,
#             pressure_solver = ConjugateGradientPoissonSolver(grid_immerse; maxiter=1, 
#             preconditioner=DiagnoallyDominantPreconditioner(),regularizer=reg_value)
#         )
#         set!(model, u=uᵢ, v=vᵢ)

#         Δt = 30
#         simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

#         # Modified progress message callback that checks for NaN
#         function progress_message(s)
#             max_u = maximum(abs, model.velocities.u)
#             max_w = maximum(abs, model.velocities.w)
            
#             # Check for NaN
#             if isnan(max_u) || isnan(max_w)
#                 @info "NaN detected for regularizer = $reg_value"
#                 results[reg_value] = "NaN"
#                 throw(:NaNDetected)
#             end
            
#             @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|u|: %.2e, max|w|: %.2e",
#                           100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
#                           s.model.clock.time, max_u, max_w)
#         end

#         simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))
        
#         # Run the simulation with exception handling
#         caught = catch_throw(run!(simulation))
        
#         # If simulation completed successfully, store the final maximum velocity
#         if caught === nothing
#             results[reg_value] = maximum(abs, model.velocities.u)
#             @info "Simulation completed successfully for regularizer = $reg_value"
#         end
        
#     catch e
#         if e == :NaNDetected
#             @info "Skipping to next regularizer value due to NaN"
#             continue
#         else
#             @warn "Unexpected error for regularizer = $reg_value: $e"
#             results[reg_value] = "Error"
#         end
#     end
# end

# # Print summary of results
# println("\nResults Summary:")
# for (reg_value, result) in sort(collect(results))
#     println("Regularizer = $reg_value: $result")
# end



using Printf
using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, 
                    fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner
using Oceananigans.Utils: prettytime
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: FFTBasedPoissonSolver

suffix = "0.1days"

## Simulation parameters
Nx = 4
Ny = 4

tᶠ = 5days # simulation run time
Δtᵒ = 180minutes # interval for saving output
H = 600meters # vertical extent
L = 2*2H # horizontal extent

# uniform grid works without blowing up
# z_faces_array = [0., 200., 400., 600.]
# nonuniform grid blows up
# z_faces_array = [range(0,300,13); range(330,600,6)]
# z_faces_array = [0., 40., 90, 150, 220, 300, 390, 490, 600.]
z_faces_array = [0., 50., 100, 150, 200., 400, 600.]

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
# preconditioner = fft_poisson_solver(grid)
preconditioner = AsymptoticPoissonPreconditioner()
regularizer = 1
model = NonhydrostaticModel(
    grid = grid_immerse,
    pressure_solver = ConjugateGradientPoissonSolver(grid_immerse; maxiter=100,preconditioner)
)
set!(model, u=uᵢ, v=vᵢ)

Δt = 30
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|u|: %.2e, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.u),maximum(abs, model.velocities.w)) 
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))
run!(simulation)


