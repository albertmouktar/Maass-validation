"""
Tests for Spiking Neuron Implementation

Validates:
1. LIF neuron parameter handling
2. Alpha EPSP shape correctness
3. Spike detection accuracy
4. Continuous-time precision (no grid artifacts)
"""

using Test
using Statistics

# Include the module
include(joinpath(@__DIR__, "..", "src", "spiking_neuron.jl"))

@testset "Spiking Neuron Tests" begin

    @testset "LIF Parameters" begin
        # Default parameters
        params = LIFParams()
        @test params.τ_m == 5.0  # Matched to τ_s for efficient EPSP integration
        @test params.τ_s == 5.0
        @test params.V_th == 1.0
        @test params.V_rest == 0.0

        # Custom parameters
        custom = LIFParams(τ_m=10.0, V_th=0.5)
        @test custom.τ_m == 10.0
        @test custom.V_th == 0.5
        @test custom.τ_s == 5.0  # Unchanged default
    end

    @testset "Alpha EPSP Shape" begin
        τ = 5.0

        # Zero at t=0
        @test alpha_epsp(0.0, τ) == 0.0

        # Negative time returns zero
        @test alpha_epsp(-1.0, τ) == 0.0

        # Peak at t=τ
        peak = alpha_epsp(τ, τ)
        @test peak ≈ 1.0 atol=1e-10

        # Monotonic increase before peak
        t_before = 0.5 * τ
        @test alpha_epsp(t_before, τ) < peak

        # Decay after peak
        t_after = 2.0 * τ
        @test alpha_epsp(t_after, τ) < peak

        # Eventually decays toward zero
        t_late = 10.0 * τ
        @test alpha_epsp(t_late, τ) < 0.01
    end

    @testset "Synaptic Current Computation" begin
        τ_s = 5.0

        # Single input
        inputs = [SynapticInput(0.0, 1.0, 0.0)]  # spike at t=0, weight=1, delay=0

        # Before input arrives
        @test compute_synaptic_current(-1.0, inputs, τ_s) == 0.0

        # At peak (t = τ_s after arrival)
        current_at_peak = compute_synaptic_current(τ_s, inputs, τ_s)
        @test current_at_peak ≈ 1.0 atol=1e-10

        # Multiple inputs sum linearly
        inputs2 = [
            SynapticInput(0.0, 0.5, 0.0),
            SynapticInput(0.0, 0.5, 0.0)
        ]
        current_sum = compute_synaptic_current(τ_s, inputs2, τ_s)
        @test current_sum ≈ 1.0 atol=1e-10
    end

    @testset "Basic Simulation" begin
        params = LIFParams()

        # No inputs → no spikes
        sol_empty, spikes_empty = simulate_lif(params, SynapticInput[], 100.0)
        @test length(spikes_empty) == 0
        @test sol_empty.u[end][1] ≈ params.V_rest atol=1e-6

        # Strong input → spike
        strong_input = [SynapticInput(1.0, 2.0, 0.0)]  # weight=2 > threshold
        sol_spike, spikes = simulate_lif(params, strong_input, 50.0)
        @test length(spikes) >= 1

        # Weak input → no spike
        weak_input = [SynapticInput(1.0, 0.3, 0.0)]  # weight=0.3 < threshold
        sol_weak, spikes_weak = simulate_lif(params, weak_input, 50.0)
        @test length(spikes_weak) == 0
    end

    @testset "Coincidence Detection" begin
        # With τ_m = τ_s = 5.0: single input (w=0.6) → V_max ≈ 0.44
        # Two coincident (w=0.6 each) → V_max ≈ 0.88
        # Threshold = 0.65 is between them
        params = LIFParams(V_th=0.65)

        # Two coincident inputs → spike
        coincident = [
            SynapticInput(1.0, 0.6, 5.0),
            SynapticInput(1.0, 0.6, 5.0)
        ]
        _, spikes_coin = simulate_lif(params, coincident, 50.0)
        @test length(spikes_coin) >= 1

        # Two separated inputs → no spike
        separated = [
            SynapticInput(1.0, 0.6, 5.0),
            SynapticInput(1.0, 0.6, 25.0)  # Different delay
        ]
        _, spikes_sep = simulate_lif(params, separated, 100.0)
        @test length(spikes_sep) == 0
    end

    @testset "Spike Timing Precision" begin
        params = LIFParams()
        input = [SynapticInput(5.0, 2.0, 0.0)]

        # Run same simulation multiple times
        spike_times_runs = []
        for _ in 1:5
            _, spikes = simulate_lif(params, input, 50.0)
            push!(spike_times_runs, spikes)
        end

        # Should be perfectly reproducible
        @test all(spike_times_runs[i] == spike_times_runs[1] for i in 2:5)
    end

    @testset "Grid Quantization Check" begin
        params = LIFParams()

        # Generate spikes from various inputs
        all_spikes = Float64[]
        for offset in [0.0, 0.137, 0.291, 0.483, 0.719]
            input = [SynapticInput(1.0 + offset, 2.0, 0.0)]
            _, spikes = simulate_lif(params, input, 50.0)
            append!(all_spikes, spikes)
        end

        if length(all_spikes) >= 2
            # Verify no grid quantization
            is_valid, report = verify_no_grid_quantization(all_spikes)
            @test is_valid
            @test report["is_valid"] == true
        end
    end

    @testset "Continuous Time Validation" begin
        params = LIFParams()
        passed, report = validate_continuous_time_precision(params; n_tests=5)
        @test passed
        @test report["all_passed"] == true
        @test report["reproducibility_test"]["all_identical"] == true
    end

end
