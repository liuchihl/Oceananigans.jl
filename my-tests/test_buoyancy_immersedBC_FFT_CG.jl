##
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

underlying_grid = RectilinearGrid(size=(3,3,3),
    x=(0, 1),
    y=(0, 1),
    z=[0, 0.2, 0.4, 0.6, 0.8, 1],
    halo=(2, 2, 2),
    topology=(Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded)
)
bottom = [0.1 0.1 0.1;
          0.1 0.2 0.1;
          0.1 0.1 0.1]

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

θ = 0
ĝ = (sin(θ), 0, cos(θ)) # the vertical (oriented opposite gravity) unit vector in rotated coordinates
# tracer: no-flux boundary condition
N = 0.001
∂B̄∂z = N^2*cos(θ)
∂B̄∂x = N^2*sin(θ)
# In the following comments, ẑ=slope-normal unit vector and x̂=cross-slope unit vector
B_bcs_immersed = ImmersedBoundaryCondition(
        bottom = GradientBoundaryCondition(-∂B̄∂z), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
          west = GradientBoundaryCondition(-∂B̄∂x), # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x
          east = GradientBoundaryCondition(-∂B̄∂x)) # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x

B_bcs = FieldBoundaryConditions(
          bottom = GradientBoundaryCondition(-∂B̄∂z), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
             top = GradientBoundaryCondition(0.), # ∇B⋅ẑ = ∂B̄∂ẑ → ∂b∂z = 0 and ∂b∂x = 0 (periodic)
        immersed = B_bcs_immersed);
uᵢ(z) = 0
T = 2
U₀ = 0.1
u_forcing(x, y, z, t) = U₀ * sin(2π / T * t)

uᵢ(x, y, z) = 0.1

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, y, z) = -Uᵣ
vᵢ(x, y, z) = 0.
bᵢ(x, y, z) = 1e-9 * rand() # seed infinitesimal random perturbations in the buoyancy field



model = NonhydrostaticModel(;
    grid=grid,
    advection=WENO(),
    forcing=(u=u_forcing,),
    timestepper=:RungeKutta3,
    buoyancy=buoyancy,
    boundary_conditions=(u=u_bcs, v=v_bcs, b=B_bcs,),
    closure=closure,
    tracers=:b,
    # hydrostatic_pressure_anomaly=CenterField(grid),
    background_fields=Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field)
    # pressure_solver=ConjugateGradientPoissonSolver(
    #     grid; maxiter=100, preconditioner=AsymptoticPoissonPreconditioner(),
    #     reltol=tol),
)
set!(model, u=uᵢ)