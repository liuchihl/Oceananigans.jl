using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
using Oceananigans.Utils: prettysummary

"""
    SeawaterBuoyancy{FT, EOS, T, S} <: AbstractBuoyancyFormulation{EOS}

Buoyancy formulation for seawater. `T` and `S` are either `nothing` if both
temperature and salinity are active, or of type `FT` if temperature
or salinity are constant, respectively.
"""
struct SeawaterBuoyancy{FT,EOS,T,S} <: AbstractBuoyancyFormulation{EOS}
    equation_of_state::EOS
    gravitational_acceleration::FT
    constant_temperature::T
    constant_salinity::S
end

# generalized EOS
function required_tracers(b::SeawaterBuoyancy)
    # if equation_of_state is a GeneralizedLinearEquationOfState, include tracer names
    if typeof(b.equation_of_state) <: GeneralizedLinearEquationOfState
        names = Symbol[]
        # include T/S if active
        if isnothing(b.constant_temperature)
            push!(names, :T)
        end
        if isnothing(b.constant_salinity)
            push!(names, :S)
        end
        return Tuple(names)
    else
        # fallback to original behavior
        return (:T, :S)
    end
end

Base.nameof(::Type{SeawaterBuoyancy}) = "SeawaterBuoyancy"
Base.summary(b::SeawaterBuoyancy) = string(nameof(typeof(b)), " with g=", prettysummary(b.gravitational_acceleration),
    " and ", summary(b.equation_of_state))

function Base.show(io::IO, b::SeawaterBuoyancy{FT}) where FT

    print(io, nameof(typeof(b)), "{$FT}:", "\n",
        "├── gravitational_acceleration: ", b.gravitational_acceleration, "\n")

    if !isnothing(b.constant_temperature)
        print(io, "├── constant_temperature: ", b.constant_temperature, "\n")
    end

    if !isnothing(b.constant_salinity)
        print(io, "├── constant_salinity: ", b.constant_salinity, "\n")
    end

    print(io, "└── equation_of_state: ", summary(b.equation_of_state))
end

"""
    SeawaterBuoyancy([FT = Float64;]
                     gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                     equation_of_state = LinearEquationOfState(FT),
                     constant_temperature = nothing,
                     constant_salinity = nothing)

Return parameters for a temperature- and salt-stratified seawater buoyancy model
with a `gravitational_acceleration` constant (typically called ``g``), and an
`equation_of_state` that related temperature and salinity (or conservative temperature
and absolute salinity) to density anomalies and buoyancy.

Setting `constant_temperature` to something that is not `nothing` indicates that buoyancy depends only on salinity.
For a nonlinear equation of state, the value provided `constant_temperature` is used as the temperature of the system.
Vice versa, setting `constant_salinity` indicates that buoyancy depends only on temperature.

For a linear equation of state, the values of `constant_temperature` or `constant_salinity`
are irrelevant.

Examples
========

The "TEOS10" equation of state, see https://www.teos-10.org

```jldoctest seawaterbuoyancy
julia> using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

julia> teos10 = TEOS10EquationOfState()
BoussinesqEquationOfState{Float64}:
├── seawater_polynomial: TEOS10SeawaterPolynomial{Float64}
└── reference_density: 1020.0
```

Buoyancy that depends on both temperature and salinity

```jldoctest seawaterbuoyancy
julia> using Oceananigans

julia> buoyancy = SeawaterBuoyancy(equation_of_state=teos10)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
└── equation_of_state: BoussinesqEquationOfState{Float64}
```

Buoyancy that depends only on salinity with temperature held at 20 degrees Celsius

```jldoctest seawaterbuoyancy
julia> salinity_dependent_buoyancy = SeawaterBuoyancy(equation_of_state=teos10, constant_temperature=20)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
├── constant_temperature: 20.0
└── equation_of_state: BoussinesqEquationOfState{Float64}
```

Buoyancy that depends only on temperature with salinity held at 35 psu

```jldoctest seawaterbuoyancy
julia> temperature_dependent_buoyancy = SeawaterBuoyancy(equation_of_state=teos10, constant_salinity=35)
SeawaterBuoyancy{Float64}:
├── gravitational_acceleration: 9.80665
├── constant_salinity: 35.0
└── equation_of_state: BoussinesqEquationOfState{Float64}
```
"""
function SeawaterBuoyancy(FT=Oceananigans.defaults.FloatType;
    gravitational_acceleration=Oceananigans.defaults.gravitational_acceleration,
    equation_of_state=LinearEquationOfState(FT),
    constant_temperature=nothing,
    constant_salinity=nothing)

    # Input validation: convert constant_temperature or constant_salinity = true to zero(FT).
    # This method of specifying constant temperature or salinity in a SeawaterBuoyancy model
    # should only be used with a LinearEquationOfState where the constant value of either temperature
    # or sailnity is irrelevant.
    constant_temperature = constant_temperature === true ? zero(FT) : constant_temperature
    constant_salinity = constant_salinity === true ? zero(FT) : constant_salinity
    equation_of_state = with_float_type(FT, equation_of_state)
    gravitational_acceleration = convert(FT, gravitational_acceleration)

    constant_temperature = isnothing(constant_temperature) ? nothing : convert(FT, constant_temperature)
    constant_salinity = isnothing(constant_salinity) ? nothing : convert(FT, constant_salinity)

    return SeawaterBuoyancy{FT,typeof(equation_of_state),typeof(constant_temperature),typeof(constant_salinity)}(
        equation_of_state, gravitational_acceleration, constant_temperature, constant_salinity)
end

const TemperatureSeawaterBuoyancy = SeawaterBuoyancy{FT,EOS,<:Nothing,<:Number} where {FT,EOS}
const SalinitySeawaterBuoyancy = SeawaterBuoyancy{FT,EOS,<:Number,<:Nothing} where {FT,EOS}

Base.nameof(::Type{TemperatureSeawaterBuoyancy}) = "TemperatureSeawaterBuoyancy"
Base.nameof(::Type{SalinitySeawaterBuoyancy}) = "SalinitySeawaterBuoyancy"

@inline get_temperature_and_salinity(::SeawaterBuoyancy, C) = C.T, C.S
@inline get_temperature_and_salinity(b::TemperatureSeawaterBuoyancy, C) = C.T, b.constant_salinity
@inline get_temperature_and_salinity(b::SalinitySeawaterBuoyancy, C) = b.constant_temperature, C.S

# Buoyancy perturbation
@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::SeawaterBuoyancy, C)
    eos = b.equation_of_state
    if eos isa GeneralizedLinearEquationOfState
        FT = eltype(grid)
        b′ = zero(FT)        # Thermal contribution
        @inbounds begin
            if isnothing(b.constant_temperature)
                b′ += eos.thermal_expansion * C.T[i, j, k]
            end
            # Haline contribution
            if isnothing(b.constant_salinity)
                b′ -= eos.haline_contraction * C.S[i, j, k]
            end
            # active tracer contributions
            for n = 1:length(eos.tracer_indices)
                idx = eos.tracer_indices[n]
                γ = eos.tracer_coefficients[n]
                b′ += γ * C.tracers[idx][i, j, k]
            end
        end
        return b.gravitational_acceleration * b′
    else # without active tracers
        T, S = get_temperature_and_salinity(b, C)
        return -(b.gravitational_acceleration * ρ′(i, j, k, grid, b.equation_of_state, T, S)
                 / b.equation_of_state.reference_density)
    end
end

#####
##### Buoyancy gradient components
#####

"""
    ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the ``x``-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_x b = g ( α ∂_x T - β ∂_x S + ∑γ_i ∂_x C_i ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity. The sum is over all active
tracers ``C_i`` with buoyancy coefficients ``γ_i``.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_x T`` (`∂x_T`), ``∂_x S`` (`∂x_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `x` and cell centers in `y` and `z`. Same is true for the active tracers.
"""

@inline function ∂x_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    eos = b.equation_of_state
    if eos isa GeneralizedLinearEquationOfState
        g = b.gravitational_acceleration
        FT = eltype(grid)
        ∂b = zero(FT)
        @inbounds begin
            if isnothing(b.constant_temperature)
                ∂b += thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, C.T) *
                      ∂xᶠᶜᶜ(i, j, k, grid, C.T)
            end
            if isnothing(b.constant_salinity)
                ∂b -= haline_contractionᶠᶜᶜ(i, j, k, grid, eos, C.S) *
                      ∂xᶠᶜᶜ(i, j, k, grid, C.S)
            end
            # tracer gradients
            for n = 1:length(eos.tracer_indices)
                idx = eos.tracer_indices[n]
                γ = eos.tracer_coefficients[n]
                ∂b += γ * ∂xᶠᶜᶜ(i, j, k, grid, C.tracers[idx])
            end
        end
        return g * ∂b
    else
        # fall back to existing method (let multiple dispatch handle it)
        return invoke(∂x_b, Tuple{Any,Any,Any,Any,SeawaterBuoyancy,Any},
            i, j, k, grid, b, C)
    end
end
"""
    ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the ``y``-derivative of buoyancy for temperature and salt-stratified water,

```math
∂_y b = g ( α ∂_y T - β ∂_y S + ∑γ_i ∂_y C_i ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_y T`` (`∂y_T`), ``∂_y S`` (`∂y_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `y` and cell centers in `x` and `z`. Same is true for the active tracers.
"""
@inline function ∂y_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    eos = b.equation_of_state
    if eos isa GeneralizedLinearEquationOfState
        g  = b.gravitational_acceleration
        FT = eltype(grid)
        ∂b = zero(FT)
        @inbounds begin
            if isnothing(b.constant_temperature)
                ∂b += thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, C.T) *
                      ∂yᶜᶠᶜ(i, j, k, grid, C.T)
            end

            if isnothing(b.constant_salinity)
                ∂b -= haline_contractionᶜᶠᶜ(i, j, k, grid, eos, C.S) *
                      ∂yᶜᶠᶜ(i, j, k, grid, C.S)
            end

            for n = 1:length(eos.tracer_indices)
                idx = eos.tracer_indices[n]
                γ   = eos.tracer_coefficients[n]
                ∂b += γ * ∂yᶜᶠᶜ(i, j, k, grid, C.tracers[idx])
            end
        end
        return g * ∂b
    else
        # fallback to original implementation
        return ∂y_b(i, j, k, grid, SeawaterBuoyancy(b), C)
    end
end
"""
    ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)

Returns the vertical derivative of buoyancy for temperature and salt-stratified water,

```math
∂_z b = N^2 = g ( α ∂_z T - β ∂_z S + ∑γ_i ∂_z C_i ) ,
```

where ``g`` is gravitational acceleration, ``α`` is the thermal expansion
coefficient, ``β`` is the haline contraction coefficient, ``T`` is
conservative temperature, and ``S`` is absolute salinity.

Note: In Oceananigans, `model.tracers.T` is conservative temperature and
`model.tracers.S` is absolute salinity.

Note that ``∂_z T`` (`∂z_T`), ``∂_z S`` (`∂z_S`), ``α``, and ``β`` are all evaluated at cell
interfaces in `z` and cell centers in `x` and `y`. Same is true for the active tracers.
"""
@inline function ∂z_b(i, j, k, grid, b::SeawaterBuoyancy, C)
    eos = b.equation_of_state
    if eos isa GeneralizedLinearEquationOfState
        g  = b.gravitational_acceleration
        FT = eltype(grid)
        ∂b = zero(FT)
        @inbounds begin
            if isnothing(b.constant_temperature)
                ∂b += thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, C.T) *
                      ∂zᶜᶜᶠ(i, j, k, grid, C.T)
            end
            if isnothing(b.constant_salinity)
                ∂b -= haline_contractionᶜᶜᶠ(i, j, k, grid, eos, C.S) *
                      ∂zᶜᶜᶠ(i, j, k, grid, C.S)
            end
            for n = 1:length(eos.tracer_indices)
                idx = eos.tracer_indices[n]
                γ   = eos.tracer_coefficients[n]
                ∂b += γ * ∂zᶜᶜᶠ(i, j, k, grid, C.tracers[idx])
            end
        end
        return g * ∂b
    else
        # fall back to existing definition
        return invoke(∂z_b,
                      Tuple{Int, Int, Int, typeof(grid), SeawaterBuoyancy, typeof(C)},
                      i, j, k, grid, b, C)
    end
end

#####
##### top buoyancy flux
#####

@inline get_temperature_and_salinity_flux(::SeawaterBuoyancy, bcs) = bcs.T, bcs.S
@inline get_temperature_and_salinity_flux(::TemperatureSeawaterBuoyancy, bcs) = bcs.T, NoFluxBoundaryCondition()
@inline get_temperature_and_salinity_flux(::SalinitySeawaterBuoyancy, bcs) = NoFluxBoundaryCondition(), bcs.S

@inline function top_bottom_buoyancy_flux(i, j, k, grid,
                                         b::SeawaterBuoyancy,
                                         top_bottom_tracer_bcs, clock, fields)
    eos = b.equation_of_state
    if eos isa GeneralizedLinearEquationOfState
        buoyancy_flux = zero(eltype(grid))

        # Temperature / salinity contributions (only if active)
        if isnothing(b.constant_temperature) || isnothing(b.constant_salinity)
            T, S = get_temperature_and_salinity(b, fields)
            T_flux_bc, S_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)
            if isnothing(b.constant_temperature)
                T_flux = getbc(T_flux_bc, i, j, grid, clock, fields)
                buoyancy_flux +=
                    thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, T, S) * T_flux
            end
            if isnothing(b.constant_salinity)
                S_flux = getbc(S_flux_bc, i, j, grid, clock, fields)
                buoyancy_flux -=
                    haline_contractionᶜᶜᶠ(i, j, k, grid, eos, T, S) * S_flux
            end
        end
        # Active tracer contributions
        @inbounds for n = 1:length(eos.tracer_indices)
            tracer_index = eos.tracer_indices[n]
            γ = eos.tracer_coefficients[n]

            tracer_bc = top_bottom_tracer_bcs.tracers[tracer_index]
            tracer_flux = getbc(tracer_bc, i, j, grid, clock, fields)

            buoyancy_flux += γ * tracer_flux
        end

        return b.gravitational_acceleration * buoyancy_flux
    else
        # without active tracers
        T, S = get_temperature_and_salinity(b, fields)
        T_flux_bc, S_flux_bc = get_temperature_and_salinity_flux(b, top_bottom_tracer_bcs)

        T_flux = getbc(T_flux_bc, i, j, grid, clock, fields)
        S_flux = getbc(S_flux_bc, i, j, grid, clock, fields)
        return b.gravitational_acceleration * (
            thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, T, S) * T_flux
          - haline_contractionᶜᶜᶠ(i, j, k, grid, eos, T, S) * S_flux
        )
    end
end

@inline top_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, grid.Nz + 1, grid, b, args...)
@inline bottom_buoyancy_flux(i, j, grid, b::SeawaterBuoyancy, args...) = top_bottom_buoyancy_flux(i, j, 1, grid, b, args...)

