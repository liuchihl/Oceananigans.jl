"""
    GeneralizedLinearEquationOfState{FT, N} <: AbstractEquationOfState

Generalized linear equation of state with multiple tracers contributing to buoyancy.

Buoyancy perturbation:
    b = g ( α T - β S + ∑ γᵢ cᵢ )
"""

struct GeneralizedLinearEquationOfState{FT,N,TI,TV} <: AbstractEquationOfState
    thermal_expansion::FT
    haline_contraction::FT
    tracer_indices::TI          # NTuple{N, Int}
    tracer_coefficients::TV     # NTuple{N, FT}
end


Base.summary(eos::GeneralizedLinearEquationOfState) =
    string("GeneralizedLinearEquationOfState(",
        "thermal_expansion=", prettysummary(eos.thermal_expansion),
        ", haline_contraction=", prettysummary(eos.haline_contraction),
        ", tracer_coefficients=", eos.tracer_coefficients,
        ")")

Base.show(io::IO, eos::GeneralizedLinearEquationOfState) =
    print(io, summary(eos))

@inline function Oceananigans.BuoyancyFormulations.with_float_type(
    ::Type{FT},
    eos::GeneralizedLinearEquationOfState) where FT
    return GeneralizedLinearEquationOfState(
        FT;
        thermal_expansion   = convert(FT, eos.thermal_expansion),
        haline_contraction  = convert(FT, eos.haline_contraction),
        tracer_indices      = eos.tracer_indices,
        tracer_coefficients = map(FT, eos.tracer_coefficients)
    )
end
"""
    GeneralizedLinearEquationOfState(FT=Float64; thermal_expansion=1.67e-4, haline_contraction=7.80e-4, tracer_indices=(), tracer_coefficients=())

Construct a GeneralizedLinearEquationOfState by specifying the floating point type, expansion/contraction coefficients, and tuples of tracer indices and coefficients.
"""
function GeneralizedLinearEquationOfState(
    FT=Oceananigans.defaults.FloatType;
    thermal_expansion=1.67e-4,
    haline_contraction=7.80e-4,
    tracer_indices=(),
    tracer_coefficients=()
)
    N = length(tracer_indices)
    return GeneralizedLinearEquationOfState{FT,N,typeof(tracer_indices),typeof(tracer_coefficients)}(
        convert(FT, thermal_expansion),
        convert(FT, haline_contraction),
        tracer_indices,
        map(FT, tracer_coefficients)
    )
end
# function GeneralizedLinearEquationOfState(
#     thermal_expansion = 1.67e-4,
#     haline_contraction = 7.80e-4,
#     tracer_buoyancy_coefficients = NamedTuple()
# )
#     tracer_names = keys(tracer_buoyancy_coefficients)
#     tracer_coeffs = values(tracer_buoyancy_coefficients)

#     tracer_indices = ntuple(i -> model.tracers[tracer_names[i]], length(tracer_names))

#     FT = eltype(tracer_coeffs)
#     N  = length(tracer_coeffs)

#     return GeneralizedLinearEquationOfState{FT, N,
#         typeof(tracer_indices),
#         typeof(tracer_coeffs)}(
#             thermal_expansion,
#             haline_contraction,
#             tracer_indices,
#             tracer_coeffs
#         )
# end

#####
##### Convinient aliases to dispatch on
#####

const GeneralizedLinearSeawaterBuoyancy = SeawaterBuoyancy{FT,<:GeneralizedLinearEquationOfState} where FT

#####
##### buoyancy perturbation
#####

@inline function buoyancy_perturbationᶜᶜᶜ(
    i, j, k, grid,
    b::GeneralizedLinearSeawaterBuoyancy,
    C
)
    eos = b.equation_of_state

    buoy = b.gravitational_acceleration *
           (eos.thermal_expansion * C.T[i, j, k] -
            eos.haline_contraction * C.S[i, j, k])

    @inbounds for n = 1:length(eos.tracer_indices)
        idx = eos.tracer_indices[n]
        buoy += b.gravitational_acceleration *
                eos.tracer_coefficients[n] *
                C[idx][i, j, k]
    end

    return buoy
end
