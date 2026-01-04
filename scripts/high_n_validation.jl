"""
High-n Validation of Maass Theorem

Tests spiking neuron CD_n for large n values (up to 500) using random sampling.
Exhaustive testing is impossible for n > ~12 (2^24 = 16M inputs).

For n=500:
- Input dimension: 1000 bits
- Possible inputs: 2^1000 (astronomically large)
- We sample 10,000-50,000 random inputs and measure accuracy
"""

using Pkg
Pkg.activate(".")

using Random
using Statistics
using JSON3
using Dates

include(joinpath(@__DIR__, "..", "src", "spiking_neuron.jl"))
include(joinpath(@__DIR__, "..", "src", "cd_n.jl"))

"""
    test_cd_n_sampled(n, n_samples; seed=42, verbose=false)

Test CD_n spiking implementation on random samples for large n.
"""
function test_cd_n_sampled(n::Int, n_samples::Int; seed::Int=42, verbose::Bool=false)
    Random.seed!(seed)

    config = configure_cd_n(n)

    correct = 0
    errors = 0

    # Track timing
    start_time = time()

    # Track positive/negative balance
    n_positive_expected = 0
    n_negative_expected = 0

    for i in 1:n_samples
        # Generate random input
        x = rand(Bool, n)
        y = rand(Bool, n)

        # Reference output (Boolean)
        expected = cd_n_boolean(x, y)
        if expected
            n_positive_expected += 1
        else
            n_negative_expected += 1
        end

        # Spiking output
        actual, spike_times, _ = compute_cd_n_spiking(x, y, config)

        if actual == expected
            correct += 1
        else
            errors += 1
            if verbose && errors <= 5
                println("  Error $errors: x=$x, y=$y, expected=$expected, got=$actual")
            end
        end

        # Progress
        if verbose && i % 1000 == 0
            elapsed = time() - start_time
            rate = i / elapsed
            println("  Progress: $i/$n_samples ($(round(100*i/n_samples, digits=1))%), " *
                    "accuracy=$(round(100*correct/i, digits=3))%, " *
                    "rate=$(round(rate, digits=1)) tests/sec")
        end
    end

    elapsed = time() - start_time
    accuracy = correct / n_samples

    results = Dict(
        "n" => n,
        "n_samples" => n_samples,
        "correct" => correct,
        "errors" => errors,
        "accuracy" => accuracy,
        "accuracy_percent" => round(100 * accuracy, digits=4),
        "elapsed_seconds" => round(elapsed, digits=2),
        "tests_per_second" => round(n_samples / elapsed, digits=1),
        "positive_samples" => n_positive_expected,
        "negative_samples" => n_negative_expected,
        "positive_fraction" => round(n_positive_expected / n_samples, digits=4),
        "synapses" => 2n,
        "max_delay_ms" => config.delays_x[end],
        "sim_time_ms" => config.T_sim
    )

    return results
end

"""
    run_high_n_experiment(; n_values, samples_per_n, verbose)

Run complete high-n validation experiment.
"""
function run_high_n_experiment(;
    n_values::Vector{Int} = [10, 20, 50, 100, 200, 500],
    samples_per_n::Int = 10000,
    verbose::Bool = true
)
    if verbose
        println("="^70)
        println("HIGH-N MAASS THEOREM VALIDATION")
        println("Testing spiking neuron CD_n for n up to $(maximum(n_values))")
        println("="^70)
        println()
        println("Configuration:")
        println("  n values: $n_values")
        println("  Samples per n: $samples_per_n")
        println("  Note: Exhaustive testing impossible for n > 12")
        println()
    end

    all_results = Dict{String, Any}(
        "metadata" => Dict(
            "timestamp" => string(Dates.now()),
            "experiment" => "high_n_spiking_validation",
            "samples_per_n" => samples_per_n,
            "n_values" => n_values
        ),
        "results" => []
    )

    for n in n_values
        if verbose
            println("-"^70)
            println("Testing n = $n")
            println("  Input dimension: $(2n) bits")
            println("  Synapses: $(2n)")
            println("  Theoretical sigmoid bound: √$n = $(round(sqrt(n), digits=2))")
            println("-"^70)
        end

        results = test_cd_n_sampled(n, samples_per_n; verbose=verbose)
        push!(all_results["results"], results)

        if verbose
            println()
            println("  RESULT for n=$n:")
            println("    Accuracy: $(results["accuracy_percent"])% ($(results["correct"])/$(results["n_samples"]))")
            println("    Errors: $(results["errors"])")
            println("    Time: $(results["elapsed_seconds"]) seconds")
            println("    Rate: $(results["tests_per_second"]) tests/sec")
            println("    Max delay: $(results["max_delay_ms"]) ms")
            println()
        end
    end

    # Summary
    if verbose
        println("="^70)
        println("SUMMARY")
        println("="^70)
        println()
        println("| n    | Samples | Accuracy      | Errors | Time (s) | Synapses |")
        println("|------|---------|---------------|--------|----------|----------|")
        for r in all_results["results"]
            println("| $(lpad(r["n"], 4)) | $(lpad(r["n_samples"], 7)) | " *
                    "$(lpad(string(r["accuracy_percent"]) * "%", 13)) | " *
                    "$(lpad(r["errors"], 6)) | " *
                    "$(lpad(r["elapsed_seconds"], 8)) | " *
                    "$(lpad(r["synapses"], 8)) |")
        end
        println()

        # Check if all passed
        all_perfect = all(r["accuracy"] == 1.0 for r in all_results["results"])
        if all_perfect
            println("✓ ALL TESTS PASSED: 100% accuracy across all n values")
        else
            println("⚠ SOME ERRORS DETECTED")
            for r in all_results["results"]
                if r["accuracy"] < 1.0
                    println("  n=$(r["n"]): $(r["errors"]) errors ($(r["accuracy_percent"])%)")
                end
            end
        end
    end

    # Save results
    results_path = joinpath(@__DIR__, "..", "data", "results",
                           "high_n_validation_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json")
    mkpath(dirname(results_path))
    open(results_path, "w") do f
        JSON3.write(f, all_results)
    end

    if verbose
        println()
        println("Results saved to: $results_path")
    end

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_high_n_experiment(
        n_values = [10, 20, 50, 100, 200, 500],
        samples_per_n = 10000,
        verbose = true
    )
end
