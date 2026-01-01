using Test
using Oceananigans
using Oceananigans.BuoyancyFormulations

@testset "GeneralizedLinearEquationOfState" begin
    
    @testset "Basic construction" begin
        # Test with no additional tracers (should behave like LinearEquationOfState)
        eos = GeneralizedLinearEquationOfState()
        @test eos.thermal_expansion == 1. 67e-4
        @test eos.haline_contraction == 7.80e-4
        @test length(eos.tracer_buoyancy_coefficients) == 0
        
        # Test with additional tracers
        eos_multi = GeneralizedLinearEquationOfState(
            tracer_buoyancy_coefficients = (c1=0.5, c2=0.3)
        )
        @test eos_multi.tracer_buoyancy_coefficients.c1 == 0.5
        @test eos_multi.tracer_buoyancy_coefficients.c2 == 0.3
    end
    
    @testset "Model integration" begin
        grid = RectilinearGrid(size=(4, 4, 4), extent=(10, 10, 10))
        
        eos = GeneralizedLinearEquationOfState(
            tracer_buoyancy_coefficients = (sed=0.6,)
        )
        
        buoyancy = SeawaterBuoyancy(equation_of_state=eos)
        
        model = NonhydrostaticModel(
            grid = grid,
            buoyancy = buoyancy,
            tracers = (:T, :S, :sed)
        )
        
        @test model.tracers isa NamedTuple{(:T, :S, :sed)}
        @test hasfield(typeof(model.buoyancy. formulation), :equation_of_state)
    end
    
    @testset "Buoyancy calculation" begin
        # TODO: Add tests for buoyancy_perturbationᶜᶜᶜ
        # and gradient functions
    end
end