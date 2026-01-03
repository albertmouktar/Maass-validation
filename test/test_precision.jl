"""
Tests for Precision Validation Module

Validates the audio DSP-inspired quality metrics:
1. SNR computation
2. Grid quantization detection
3. Aliasing artifact detection
4. Anti-Morrison validation
"""

using Test
using Statistics
using Dates

# Include the module
include(joinpath(@__DIR__, "..", "src", "precision_validation.jl"))

@testset "Precision Validation Tests" begin

    @testset "SNR Computation" begin
        # Perfect signal (no noise)
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        noise = [0.0, 0.0, 0.0, 0.0, 0.0]
        snr = compute_snr_db(signal, noise)
        @test isinf(snr)

        # Equal signal and noise → 0 dB
        signal = [1.0, 1.0, 1.0, 1.0]
        noise = [1.0, 1.0, 1.0, 1.0]
        snr = compute_snr_db(signal, noise)
        @test snr ≈ 0.0 atol=1e-10

        # 10x signal power → 10 dB
        signal = sqrt(10) .* [1.0, 1.0, 1.0, 1.0]
        noise = [1.0, 1.0, 1.0, 1.0]
        snr = compute_snr_db(signal, noise)
        @test snr ≈ 10.0 atol=0.1
    end

    @testset "Timing Precision Analysis" begin
        # Perfect timing
        computed = [1.0, 2.0, 3.0, 4.0, 5.0]
        reference = [1.0, 2.0, 3.0, 4.0, 5.0]

        analysis = analyze_timing_precision(computed, reference)
        @test isinf(analysis["snr_db"])
        @test analysis["max_absolute_error"] == 0.0
        @test analysis["precision_grade"] == "EXCELLENT"

        # Small errors
        computed_err = [1.001, 2.001, 3.001, 4.001, 5.001]
        analysis_err = analyze_timing_precision(computed_err, reference)
        @test analysis_err["snr_db"] > 60  # Still good precision
        @test analysis_err["max_absolute_error"] ≈ 0.001 atol=1e-10

        # Mismatched lengths
        analysis_mismatch = analyze_timing_precision([1.0, 2.0], [1.0])
        @test analysis_mismatch["error"] == "spike_count_mismatch"
    end

    @testset "Precision Grading" begin
        @test grade_precision(Inf) == "EXCELLENT"
        @test grade_precision(130.0) == "EXCELLENT"
        @test grade_precision(100.0) == "VERY_GOOD"
        @test grade_precision(70.0) == "GOOD"
        @test grade_precision(30.0) == "POOR"
    end

    @testset "Grid Quantization Detection" begin
        # Continuous-time spike times (not on any grid)
        continuous_times = [1.234567, 5.891234, 12.345678, 23.456789, 45.123456]
        result = detect_grid_quantization(continuous_times)

        @test result["is_quantized"] == false
        @test result["continuous_time_verified"] == true

        # Grid-quantized spike times (on 0.1ms grid)
        grid_times = [1.1, 2.3, 4.5, 7.8, 10.0]
        result_grid = detect_grid_quantization(grid_times)

        @test result_grid["grid_checks"][0.1]["is_quantized"] == true
        @test result_grid["is_quantized"] == true

        # 1ms grid
        grid_1ms = [1.0, 3.0, 5.0, 8.0, 12.0]
        result_1ms = detect_grid_quantization(grid_1ms)
        @test result_1ms["grid_checks"][1.0]["is_quantized"] == true
    end

    @testset "Aliasing Detection" begin
        # Random spike times (no aliasing)
        random_times = sort(rand(50) .* 100)  # Random times 0-100ms
        result = detect_aliasing_artifacts(random_times)

        @test result["n_spikes"] == 50
        # Random distribution should not be suspicious
        @test length(result["suspicious_frequencies"]) == 0 ||
              length(result["suspicious_frequencies"]) <= 1  # Allow some noise

        # Periodic spike times (would show aliasing)
        periodic_times = collect(0.0:1.0:50.0)  # Every 1ms
        result_periodic = detect_aliasing_artifacts(periodic_times;
            test_frequencies=[1000.0])  # Check 1ms grid

        # Periodic at 1ms should be detected at 1000 Hz
        @test result_periodic["frequency_analysis"][1000.0]["suspicious"] == true
    end

    @testset "ISI Analysis" begin
        # Varied ISIs (healthy)
        varied_times = [1.0, 3.5, 5.2, 9.8, 15.3, 18.7]
        result = analyze_isi_distribution(varied_times)

        @test result["n_intervals"] == 5
        @test result["cv"] > 0.1  # Not suspiciously regular
        @test result["continuous_time_verified"] == true

        # Perfectly regular ISIs (suspicious)
        regular_times = collect(0.0:5.0:50.0)
        result_regular = analyze_isi_distribution(regular_times)

        @test result_regular["cv"] ≈ 0.0 atol=1e-10
        @test result_regular["suspiciously_regular"] == true

        # Insufficient data
        result_short = analyze_isi_distribution([1.0])
        @test result_short["status"] == "insufficient_spikes"
    end

    @testset "Full Validation Suite" begin
        # Good continuous-time data
        good_times = [1.234, 5.678, 12.345, 23.456, 45.678, 67.891]
        validation = run_full_precision_validation(good_times)

        @test haskey(validation, "summary")
        @test validation["summary"]["checks_total"] == 3

        # With reference times
        reference = [1.234, 5.678, 12.345, 23.456, 45.678, 67.891]
        validation_ref = run_full_precision_validation(good_times; reference_times=reference)

        @test haskey(validation_ref, "precision")
        @test validation_ref["precision"]["precision_grade"] == "EXCELLENT"
    end

    @testset "Anti-Morrison Validation" begin
        # Good continuous data
        good_times = [1.23456789, 5.87654321, 12.34567890, 23.45678901]
        passed, report = validate_anti_morrison(good_times)

        @test passed == true
        @test report["assessment"] == "CONTINUOUS_TIME_VERIFIED"
        @test report["no_morrison_grid_artifacts"] == true

        # Data on 0.1ms grid (Morrison's common timestep)
        morrison_times = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
        passed_morrison, report_morrison = validate_anti_morrison(morrison_times)

        @test passed_morrison == false
        @test report_morrison["assessment"] == "POTENTIAL_DISCRETE_ARTIFACTS"
    end

    @testset "Morrison Precision Comparison" begin
        # High precision (better than Morrison)
        comparison = compare_to_morrison_precision(100.0)

        @test comparison["our_snr_db"] == 100.0
        @test comparison["continuous_advantage"] == true
        @test occursin("EXCELLENT", comparison["assessment"])

        # Perfect precision
        comparison_inf = compare_to_morrison_precision(Inf)
        @test occursin("INFINITE", comparison_inf["assessment"])
    end

end
