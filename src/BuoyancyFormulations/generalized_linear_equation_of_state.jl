"""
    GeneralizedLinearEquationOfState{FT, N, NT} <: AbstractEquationOfState

Generalized linear equation of state for multiple tracers contributing to buoyancy. 
Supports the form:  b = α(T-T₀) + β(S-S₀) + ∑γᵢcᵢ
"""
struct GeneralizedLinearEquationOfState{FT, N, NT} <: AbstractEquationOfState
    thermal_expansion ::  FT
    haline_contraction :: FT
    tracer_buoyancy_coefficients :: NT  # NamedTuple of coefficients for each additional tracer
end

Base.summary(eos::GeneralizedLinearEquationOfState) =
    string("GeneralizedLinearEquationOfState(thermal_expansion=", prettysummary(eos.thermal_expansion),
           ", haline_contraction=", prettysummary(eos.haline_contraction),
           ", tracer_buoyancy_coefficients=", eos.tracer_buoyancy_coefficients, ")")

Base.show(io::IO, eos:: GeneralizedLinearEquationOfState) = print(io, summary(eos))

"""
    GeneralizedLinearEquationOfState([FT=Float64;] 
                                     thermal_expansion=1.67e-4, 
                                     haline_contraction=7.80e-4,
                                     tracer_buoyancy_coefficients=NamedTuple())

Create a generalized linear equation of state that supports multiple tracers contributing to buoyancy. 

# Arguments
- `FT`: Float type (default: Float64)
- `thermal_expansion`: Thermal expansion coefficient α (default: 1.67e-4 K⁻¹)
- `haline_contraction`: Haline contraction coefficient β (default:  7.80e-4 psu⁻¹)
- `tracer_buoyancy_coefficients`: NamedTuple of coefficients for additional tracers
  (e.g., `(c1=0.5, c2=0.3)` for tracers c1 and c2)

# Example
```julia
# For sediment with two size classes
eos = GeneralizedLinearEquationOfState(
    thermal_expansion = 2e-4,
    haline_contraction = 8e-4,
    tracer_buoyancy_coefficients = (sed1 = 0.6, sed2 = 0.4)
)
```
"""