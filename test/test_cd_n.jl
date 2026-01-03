"""
Tests for CD_n Function Implementation

Validates:
1. Boolean reference implementation correctness
2. Delay configuration logic
3. Spiking implementation accuracy
4. Exhaustive testing for small n
"""

using Test
using Statistics

# Include the modules
include(joinpath(@__DIR__, "..", "src", "spiking_neuron.jl"))
include(joinpath(@__DIR__, "..", "src", "cd_n.jl"))

@testset "CD_n Tests" begin

    @testset "Boolean Reference Implementation" begin
        # Basic cases for n=2
        @test cd_n_boolean([false, false], [false, false]) == false
        @test cd_n_boolean([true, false], [false, true]) == false  # No coincidence
        @test cd_n_boolean([true, false], [true, false]) == true   # Coincidence at pos 1
        @test cd_n_boolean([false, true], [false, true]) == true   # Coincidence at pos 2
        @test cd_n_boolean([true, true], [true, true]) == true     # Multiple coincidences

        # Single-vector interface
        @test cd_n_boolean([true, false, true, false]) == true   # n=2: x=[1,0], y=[1,0]
        @test cd_n_boolean([true, false, false, true]) == false  # n=2: x=[1,0], y=[0,1]
    end

    @testset "CD_n Symmetry Properties" begin
        # CD_n should be symmetric in x,y pairs (not across x and y)
        for _ in 1:10
            n = 4
            x = rand(Bool, n)
            y = rand(Bool, n)

            # Result should only depend on which positions have coincident 1s
            expected = any(x .& y)
            @test cd_n_boolean(x, y) == expected
        end
    end

    @testset "Delay Configuration" begin
        # Test for n=4
        config = configure_cd_n(4)

        @test config.n == 4
        @test length(config.delays_x) == 4
        @test length(config.delays_y) == 4

        # Delays should be matched
        @test config.delays_x == config.delays_y

        # Delays should be increasing
        @test issorted(config.delays_x)

        # Weight and threshold should satisfy: weight < threshold < 2*weight
        @test config.weight < config.threshold
        @test config.threshold < 2 * config.weight
    end

    @testset "Spiking CD_n Basic Cases" begin
        config = configure_cd_n(2)

        # No coincidence → output 0
        output1, _, _ = compute_cd_n_spiking([false, false], [false, false], config)
        @test output1 == false

        output2, _, _ = compute_cd_n_spiking([true, false], [false, true], config)
        @test output2 == false

        # Coincidence → output 1
        output3, spikes3, _ = compute_cd_n_spiking([true, false], [true, false], config)
        @test output3 == true
        @test length(spikes3) >= 1
    end

    @testset "Exhaustive Test n=2" begin
        # All 16 combinations for n=2
        results = test_cd_n_exhaustive(2; verbose=false)

        @test results["n"] == 2
        @test results["total_tests"] == 16
        @test results["accuracy"] == 1.0
        @test results["errors"] == 0
    end

    @testset "Exhaustive Test n=3" begin
        # All 64 combinations for n=3
        results = test_cd_n_exhaustive(3; verbose=false)

        @test results["n"] == 3
        @test results["total_tests"] == 64
        @test results["accuracy"] == 1.0
        @test results["errors"] == 0
    end

    @testset "Input Generation" begin
        inputs = generate_all_inputs(2)
        @test length(inputs) == 16  # 2^(2*2) = 16

        inputs3 = generate_all_inputs(3)
        @test length(inputs3) == 64  # 2^(2*3) = 64

        # Check all inputs are unique
        @test length(unique(inputs)) == length(inputs)
    end

    @testset "Precision Tracking" begin
        config = configure_cd_n(3)

        # Test with coincident inputs (should spike)
        x = [true, false, false]
        y = [true, false, false]

        output, spike_times, precision_info = compute_cd_n_spiking(x, y, config)

        @test output == true
        @test haskey(precision_info, "n_output_spikes")
        @test precision_info["n_output_spikes"] >= 1

        # If spikes occurred, check precision validation
        if precision_info["n_output_spikes"] > 0
            @test haskey(precision_info, "no_grid_quantization")
        end
    end

    @testset "Computational Requirements Analysis" begin
        analysis = analyze_cd_n_computational_requirements([2, 4, 6])

        @test length(analysis) == 3

        for result in analysis
            @test result["spiking_neurons"] == 1  # Always 1 output neuron
            @test result["spiking_synapses"] == 2 * result["n"]
        end
    end

end
