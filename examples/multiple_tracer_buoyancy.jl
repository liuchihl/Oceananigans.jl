# Example:  Particle-driven flow with multiple sediment size classes
# This demonstrates how to use GeneralizedLinearEquationOfState for 
# simulations with multiple tracers contributing to buoyancy

using Oceananigans
using Oceananigans.BuoyancyFormulations

# Grid setup
grid = RectilinearGrid(size=(64, 64, 32), 
                       extent=(100, 100, 50))

# Define buoyancy with T, S, and two sediment classes
# Sediment particles with different densities contribute differently to buoyancy
eos = GeneralizedLinearEquationOfState(
    thermal_expansion = 2e-4,      # α for temperature
    haline_contraction = 8e-4,     # β for salinity
    tracer_buoyancy_coefficients = (
        sed1 = 0.6,  # γ₁ for fine sediment
        sed2 = 0.4   # γ₂ for coarse sediment
    )
)

buoyancy = SeawaterBuoyancy(equation_of_state=eos)

# Create model with all tracers
model = NonhydrostaticModel(
    grid = grid,
    buoyancy = buoyancy,
    tracers = (:T, :S, :sed1, :sed2),  # Temperature, Salinity, and two sediment classes
    advection = WENO()
)

# Set initial conditions
# Temperature gradient
set!(model, T = (x, y, z) -> 20 + 0.01 * z)

# Salinity gradient  
set!(model, S = (x, y, z) -> 35 + 0.001 * z)

# Sediment 1 (fine particles) - initially concentrated near surface
set!(model, sed1 = (x, y, z) -> exp((z + 50) / 10))

# Sediment 2 (coarse particles) - initially concentrated near bottom
set!(model, sed2 = (x, y, z) -> exp(-z / 5))

# Now the buoyancy will be calculated as: 
# b = g * (α*T - β*S + γ₁*sed1 + γ₂*sed2)

# Add different settling velocities for each sediment class
w_settling_1 = -0.001  # m/s (fine particles settle slowly)
w_settling_2 = -0.01   # m/s (coarse particles settle faster)

# You can add settling as a forcing term or advective velocity
# Example with forcing: 
using Oceananigans.Forcing

@inline settling_flux_sed1(i, j, k, grid, clock, fields) = 
    @inbounds w_settling_1 * fields. sed1[i, j, k]

@inline settling_flux_sed2(i, j, k, grid, clock, fields) = 
    @inbounds w_settling_2 * fields. sed2[i, j, k]

# Recreate model with settling
model = NonhydrostaticModel(
    grid = grid,
    buoyancy = buoyancy,
    tracers = (:T, :S, :sed1, :sed2),
    forcing = (
        sed1 = Forcing(settling_flux_sed1, discrete_form=true),
        sed2 = Forcing(settling_flux_sed2, discrete_form=true)
    )
)

# Run simulation
simulation = Simulation(model, Δt=10.0, stop_time=3600)

# Add diagnostics
using Oceananigans.OutputWriters

simulation.output_writers[:fields] = NetCDFOutputWriter(
    model, 
    merge(model.velocities, model.tracers),
    schedule = TimeInterval(60),
    filename = "multi_tracer_sediment.nc",
    overwrite_existing = true
)

# Print initial buoyancy statistics
b = buoyancy_field(model)
compute!(b)
@info "Initial buoyancy: min=$(minimum(b)), max=$(maximum(b))"

run!(simulation)

@info "Simulation complete!"