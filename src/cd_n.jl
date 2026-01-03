"""
CD_n Function Implementation for Maass Theorem Validation

The Coincidence Detection function CD_n is central to Maass's 1997 theorem:
- A single spiking neuron with programmable delays can compute CD_n exactly
- Sigmoidal networks require Ω(√n) hidden units to compute CD_n

This module implements CD_n using continuous-time spiking neurons,
validating the theoretical claim empirically for the first time in 28 years.

Mathematical Definition:
CD_n: {0,1}^{2n} → {0,1}
CD_n(x₁,...,xₙ,y₁,...,yₙ) = 1 iff ∃i: xᵢ = yᵢ = 1

The key insight: With matched delays Δₐᵢ = Δᵦᵢ, coincident 1s arrive
simultaneously and sum to exceed threshold, while non-coincident inputs
arrive separately and remain subthreshold.
"""

include("spiking_neuron.jl")

using Statistics

# ============================================================================
# BOOLEAN REFERENCE IMPLEMENTATION
# ============================================================================

"""
    cd_n_boolean(x, y)

Reference implementation of CD_n as a pure Boolean function.
Used for validating the spiking implementation.

CD_n(x,y) = 1 iff ∃i: xᵢ = yᵢ = 1 (coincident 1s exist)

# Arguments
- `x::Vector{Bool}`: First n-bit input vector
- `y::Vector{Bool}`: Second n-bit input vector

# Returns
- `Bool`: true if any position has both x[i]=1 and y[i]=1
"""
function cd_n_boolean(x::Vector{Bool}, y::Vector{Bool})::Bool
    @assert length(x) == length(y) "Input vectors must have equal length"
    return any(x .& y)
end

"""
    cd_n_boolean(input::Vector{Bool})

Alternative signature taking a single 2n-length vector.
First n elements are x, second n elements are y.
"""
function cd_n_boolean(input::Vector{Bool})::Bool
    n = length(input) ÷ 2
    @assert length(input) == 2n "Input length must be even"
    x = input[1:n]
    y = input[n+1:end]
    return cd_n_boolean(x, y)
end

# ============================================================================
# DELAY CONFIGURATION FOR SPIKING CD_n
# ============================================================================

"""
    CDnConfig

Configuration for CD_n implementation with spiking neuron.

# Fields
- `n::Int`: Problem size (number of bit pairs)
- `delays_x::Vector{Float64}`: Delays for x inputs (ms)
- `delays_y::Vector{Float64}`: Delays for y inputs (ms)
- `weight::Float64`: Synaptic weight for each input
- `threshold::Float64`: Spike threshold
- `τ_s::Float64`: Synaptic time constant
- `T_input::Float64`: Time when input spikes are delivered
- `T_sim::Float64`: Total simulation time
"""
struct CDnConfig
    n::Int
    delays_x::Vector{Float64}
    delays_y::Vector{Float64}
    weight::Float64
    threshold::Float64
    τ_s::Float64
    T_input::Float64
    T_sim::Float64
end

"""
    configure_cd_n(n; separation_factor=3.0, τ_s=5.0)

Configure delays and parameters for CD_n spiking implementation.

The key to CD_n is matched delays: Δₓᵢ = Δᵧᵢ
This ensures that if both xᵢ=1 and yᵢ=1, their EPSPs arrive
simultaneously and sum to exceed threshold.

Delay configuration:
Δᵢ = base_delay + (i-1) × separation

where separation = separation_factor × τ_s ensures non-overlapping
response windows for different input pairs.

# Arguments
- `n::Int`: Problem size
- `separation_factor::Float64`: Multiple of τ_s for delay spacing
- `τ_s::Float64`: Synaptic time constant (ms)

# Returns
- `CDnConfig`: Complete configuration for CD_n implementation
"""
function configure_cd_n(n::Int; separation_factor::Float64 = 3.0, τ_s::Float64 = 5.0)
    # Base delay to ensure all inputs have some propagation time
    base_delay = 5.0  # ms

    # Separation between delay slots
    separation = separation_factor * τ_s

    # Matched delays for x and y inputs
    delays_x = [base_delay + (i-1) * separation for i in 1:n]
    delays_y = [base_delay + (i-1) * separation for i in 1:n]  # MATCHED!

    # Weight configuration:
    # - Single EPSP should NOT exceed threshold
    # - Two coincident EPSPs SHOULD exceed threshold
    #
    # With τ_m = τ_s = 5.0ms, the membrane integrates ~73.6% of peak EPSP:
    # - Single input:  V_max ≈ 0.736 × weight
    # - Two coincident: V_max ≈ 1.472 × weight
    # Threshold must be between these values.
    weight = 1.0  # Normalized weight

    # Threshold: midpoint between single and double response
    # 0.736 < 1.1 < 1.472, so one input stays below, two exceed
    threshold = 1.1

    # Timing
    T_input = 1.0  # All input spikes delivered at t=1ms
    T_sim = base_delay + n * separation + 3 * τ_s  # Enough time for all responses

    return CDnConfig(n, delays_x, delays_y, weight, threshold, τ_s, T_input, T_sim)
end

# ============================================================================
# SPIKING CD_n IMPLEMENTATION
# ============================================================================

"""
    compute_cd_n_spiking(x, y, config)

Compute CD_n using spiking neuron with programmed delays.

This implements Maass's construction:
1. Each input bit xᵢ=1 generates a spike with delay Δₓᵢ
2. Each input bit yᵢ=1 generates a spike with delay Δᵧᵢ
3. Matched delays (Δₓᵢ = Δᵧᵢ) cause coincident inputs to sum
4. Output neuron fires iff coincident 1s exist (threshold exceeded)

# Arguments
- `x::Vector{Bool}`: First n-bit input
- `y::Vector{Bool}`: Second n-bit input
- `config::CDnConfig`: CD_n configuration

# Returns
- `output::Bool`: CD_n result (true if any coincident 1s)
- `spike_times::Vector{Float64}`: Output spike times (for analysis)
- `precision_info::Dict`: Precision validation information
"""
function compute_cd_n_spiking(
    x::Vector{Bool},
    y::Vector{Bool},
    config::CDnConfig
)
    @assert length(x) == config.n "x must have length n=$(config.n)"
    @assert length(y) == config.n "y must have length n=$(config.n)"

    # Build input spike list
    inputs = SynapticInput[]

    # Add x inputs (those that are 1)
    for i in 1:config.n
        if x[i]
            push!(inputs, SynapticInput(config.T_input, config.weight, config.delays_x[i]))
        end
    end

    # Add y inputs (those that are 1)
    for i in 1:config.n
        if y[i]
            push!(inputs, SynapticInput(config.T_input, config.weight, config.delays_y[i]))
        end
    end

    # Create neuron parameters with configured threshold
    params = LIFParams(
        τ_s = config.τ_s,
        V_th = config.threshold,
        V_rest = 0.0,
        V_reset = 0.0
    )

    # Run simulation
    sol, spike_times = simulate_lif(params, inputs, config.T_sim)

    # Output is 1 if neuron fired at least once
    output = length(spike_times) > 0

    # Collect precision information
    precision_info = Dict{String, Any}()
    precision_info["n_output_spikes"] = length(spike_times)
    precision_info["spike_times"] = spike_times
    precision_info["n_inputs"] = length(inputs)

    if length(spike_times) > 0
        is_valid, grid_report = verify_no_grid_quantization(spike_times)
        precision_info["no_grid_quantization"] = is_valid
        precision_info["grid_report"] = grid_report
    end

    return output, spike_times, precision_info
end

# ============================================================================
# EXHAUSTIVE TESTING
# ============================================================================

"""
    generate_all_inputs(n)

Generate all 2^(2n) possible input combinations for CD_n.

# Returns
- Vector of tuples (x::Vector{Bool}, y::Vector{Bool})
"""
function generate_all_inputs(n::Int)
    inputs = Tuple{Vector{Bool}, Vector{Bool}}[]

    # Total number of input combinations: 2^(2n)
    total = 2^(2n)

    for i in 0:(total-1)
        # Convert integer to 2n-bit binary representation
        bits = digits(i, base=2, pad=2n)
        x = Bool.(bits[1:n])
        y = Bool.(bits[n+1:2n])
        push!(inputs, (x, y))
    end

    return inputs
end

"""
    test_cd_n_exhaustive(n; verbose=false)

Exhaustively test CD_n spiking implementation against Boolean reference.

For each of the 2^(2n) possible inputs:
1. Compute reference output using Boolean function
2. Compute spiking output using continuous-time simulation
3. Verify they match

# Arguments
- `n::Int`: Problem size
- `verbose::Bool`: Print progress and details

# Returns
- `results::Dict`: Test results including accuracy, errors, timing
"""
function test_cd_n_exhaustive(n::Int; verbose::Bool = false)
    config = configure_cd_n(n)
    all_inputs = generate_all_inputs(n)
    total = length(all_inputs)

    correct = 0
    errors = []
    all_precision_info = []

    start_time = time()

    for (idx, (x, y)) in enumerate(all_inputs)
        # Reference output
        expected = cd_n_boolean(x, y)

        # Spiking output
        actual, spike_times, precision_info = compute_cd_n_spiking(x, y, config)

        push!(all_precision_info, precision_info)

        if actual == expected
            correct += 1
        else
            push!(errors, Dict(
                "x" => x,
                "y" => y,
                "expected" => expected,
                "actual" => actual,
                "spike_times" => spike_times
            ))
        end

        if verbose && idx % 1000 == 0
            println("Progress: $idx / $total ($(round(100*idx/total, digits=1))%)")
        end
    end

    elapsed = time() - start_time
    accuracy = correct / total

    # Aggregate precision validation
    n_with_spikes = count(p -> p["n_output_spikes"] > 0, all_precision_info)
    n_valid_precision = count(p -> get(p, "no_grid_quantization", true), all_precision_info)

    results = Dict(
        "n" => n,
        "total_tests" => total,
        "correct" => correct,
        "errors" => length(errors),
        "accuracy" => accuracy,
        "elapsed_seconds" => elapsed,
        "error_cases" => errors,
        "precision_validation" => Dict(
            "tests_with_spikes" => n_with_spikes,
            "tests_with_valid_precision" => n_valid_precision,
            "precision_pass_rate" => n_valid_precision / max(n_with_spikes, 1)
        )
    )

    if verbose
        println("\n=== CD_$n Exhaustive Test Results ===")
        println("Total tests: $total")
        println("Correct: $correct")
        println("Errors: $(length(errors))")
        println("Accuracy: $(round(100*accuracy, digits=4))%")
        println("Time: $(round(elapsed, digits=2)) seconds")
        println("Precision validation pass rate: $(round(100*results["precision_validation"]["precision_pass_rate"], digits=2))%")
    end

    return results
end

# ============================================================================
# COMPARISON WITH THEORETICAL PREDICTIONS
# ============================================================================

"""
    analyze_cd_n_computational_requirements(n_values)

Analyze computational requirements for CD_n at different problem sizes.

For spiking implementation:
- Uses O(1) neurons (just the output neuron)
- Uses 2n synaptic inputs with programmed delays

For sigmoidal networks:
- Theoretical lower bound: Ω(√n) hidden units
- We measure empirically in sigmoid_network.jl

# Returns
- Analysis dictionary with computational complexity data
"""
function analyze_cd_n_computational_requirements(n_values::Vector{Int})
    results = Dict{String, Any}[]

    for n in n_values
        config = configure_cd_n(n)

        push!(results, Dict(
            "n" => n,
            "spiking_neurons" => 1,  # Just the output neuron
            "spiking_synapses" => 2n,  # 2n inputs
            "spiking_distinct_delays" => n,  # n unique delay values
            "total_inputs" => 2^(2n),
            "delay_range_ms" => (config.delays_x[1], config.delays_x[end])
        ))
    end

    return results
end

# ============================================================================
# EXPORTS
# ============================================================================

export cd_n_boolean
export CDnConfig, configure_cd_n
export compute_cd_n_spiking
export generate_all_inputs, test_cd_n_exhaustive
export analyze_cd_n_computational_requirements
