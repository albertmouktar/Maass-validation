"""
Sigmoid Network Baseline for Maass Theorem Validation

This module implements sigmoidal feedforward networks to empirically
determine the minimum number of hidden units required to compute CD_n.

Maass's 1997 theorem claims Ω(√n) hidden units are necessary.
We test this empirically by incrementally increasing hidden units
until 100% accuracy is achieved.

Key insight: If empirical minimum hidden units scales as √n,
this confirms the theoretical lower bound and demonstrates
the computational advantage of temporal coding.
"""

using Flux
using Statistics
using Random

# ============================================================================
# DATASET GENERATION
# ============================================================================

"""
    generate_cd_n_dataset(n)

Generate complete dataset for CD_n classification.

# Arguments
- `n::Int`: Problem size

# Returns
- `X::Matrix{Float32}`: Input matrix (2n × 2^(2n)), each column is one input
- `Y::Vector{Float32}`: Labels (0.0 or 1.0)
"""
function generate_cd_n_dataset(n::Int)
    total = 2^(2n)

    # Preallocate
    X = zeros(Float32, 2n, total)
    Y = zeros(Float32, total)

    for i in 0:(total-1)
        # Convert to binary
        bits = digits(i, base=2, pad=2n)
        X[:, i+1] = Float32.(bits)

        # Compute CD_n label
        x_bits = Bool.(bits[1:n])
        y_bits = Bool.(bits[n+1:2n])
        Y[i+1] = any(x_bits .& y_bits) ? 1.0f0 : 0.0f0
    end

    return X, Y
end

"""
    generate_cd_n_dataset_sampled(n, n_samples; rng=Random.GLOBAL_RNG)

Generate sampled dataset for large n where exhaustive enumeration is impractical.

Uses stratified sampling to ensure balanced representation of positive/negative cases.

# Arguments
- `n::Int`: Problem size
- `n_samples::Int`: Number of samples to generate
- `rng`: Random number generator for reproducibility

# Returns
- `X::Matrix{Float32}`: Input matrix (2n × n_samples)
- `Y::Vector{Float32}`: Labels
"""
function generate_cd_n_dataset_sampled(n::Int, n_samples::Int; rng = Random.GLOBAL_RNG)
    X = zeros(Float32, 2n, n_samples)
    Y = zeros(Float32, n_samples)

    # Target roughly balanced dataset
    n_positive = 0
    n_negative = 0
    target_each = n_samples ÷ 2

    idx = 1
    while idx <= n_samples
        # Generate random input
        bits = rand(rng, Bool, 2n)
        X[:, idx] = Float32.(bits)

        # Compute label
        x_bits = bits[1:n]
        y_bits = bits[n+1:2n]
        label = any(x_bits .& y_bits) ? 1.0f0 : 0.0f0

        # Stratified sampling: don't oversample one class
        if label == 1.0f0 && n_positive < target_each
            Y[idx] = label
            n_positive += 1
            idx += 1
        elseif label == 0.0f0 && n_negative < target_each
            Y[idx] = label
            n_negative += 1
            idx += 1
        elseif n_positive >= target_each && n_negative >= target_each
            # Both met quota, accept any
            Y[idx] = label
            idx += 1
        end
        # Otherwise, reject and try again
    end

    return X, Y
end

# ============================================================================
# NETWORK CONSTRUCTION
# ============================================================================

"""
    create_sigmoid_network(input_dim, hidden_units)

Create a two-layer sigmoid network for binary classification.

Architecture: Input(2n) → Dense(h, sigmoid) → Dense(1, sigmoid)

# Arguments
- `input_dim::Int`: Input dimension (2n for CD_n)
- `hidden_units::Int`: Number of hidden units

# Returns
- Flux Chain model
"""
function create_sigmoid_network(input_dim::Int, hidden_units::Int)
    return Chain(
        Dense(input_dim, hidden_units, sigmoid),
        Dense(hidden_units, 1, sigmoid)
    )
end

# ============================================================================
# TRAINING
# ============================================================================

"""
    train_sigmoid_network!(model, X, Y;
        lr=0.01, max_epochs=5000, patience=100, target_accuracy=1.0)

Train sigmoid network with early stopping.

Uses binary cross-entropy loss and Adam optimizer.
Stops when target accuracy is reached or patience is exceeded.

# Arguments
- `model`: Flux model to train
- `X`: Input data (features × samples)
- `Y`: Labels (1 × samples or vector)
- `lr`: Learning rate
- `max_epochs`: Maximum training epochs
- `patience`: Early stopping patience (epochs without improvement)
- `target_accuracy`: Stop training when this accuracy is reached

# Returns
- `history::Dict`: Training history with loss and accuracy per epoch
- `final_accuracy::Float64`: Final accuracy on training data
"""
function train_sigmoid_network!(
    model,
    X::Matrix{Float32},
    Y::Vector{Float32};
    lr::Float64 = 0.1,
    max_epochs::Int = 10000,
    patience::Int = 500,
    target_accuracy::Float64 = 1.0
)
    # Reshape Y for Flux
    Y_matrix = reshape(Y, 1, :)

    # Loss function: binary cross-entropy
    loss(x, y) = Flux.Losses.binarycrossentropy(model(x), y)

    # Optimizer
    opt_state = Flux.setup(Adam(lr), model)

    # Training history
    history = Dict(
        "loss" => Float64[],
        "accuracy" => Float64[]
    )

    best_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in 1:max_epochs
        # Compute gradients and update
        grads = Flux.gradient(model) do m
            Flux.Losses.binarycrossentropy(m(X), Y_matrix)
        end
        Flux.update!(opt_state, model, grads[1])

        # Compute metrics
        predictions = model(X)
        current_loss = Flux.Losses.binarycrossentropy(predictions, Y_matrix)

        # Accuracy (threshold at 0.5)
        predicted_labels = vec(predictions) .> 0.5f0
        true_labels = Y .> 0.5f0
        current_accuracy = mean(predicted_labels .== true_labels)

        push!(history["loss"], current_loss)
        push!(history["accuracy"], current_accuracy)

        # Check for improvement
        if current_accuracy > best_accuracy
            best_accuracy = current_accuracy
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end

        # Early stopping conditions
        if current_accuracy >= target_accuracy
            break
        end

        if epochs_without_improvement >= patience
            break
        end
    end

    # Final accuracy
    predictions = model(X)
    predicted_labels = vec(predictions) .> 0.5f0
    true_labels = Y .> 0.5f0
    final_accuracy = mean(predicted_labels .== true_labels)

    return history, final_accuracy
end

# ============================================================================
# MINIMUM HIDDEN UNITS SEARCH
# ============================================================================

"""
    find_minimum_hidden_units(n;
        hidden_candidates=[1,2,4,8,16,32,64,128,256,512],
        num_trials=10,
        required_success_rate=0.8,
        rng=nothing)

Find minimum hidden units needed for sigmoid network to compute CD_n.

For each candidate hidden unit count:
1. Train multiple networks (num_trials)
2. Check if required_success_rate achieve 100% accuracy
3. Return first h that meets criterion

# Arguments
- `n::Int`: Problem size
- `hidden_candidates`: List of hidden unit counts to try
- `num_trials`: Number of training attempts per hidden unit count
- `required_success_rate`: Fraction of trials that must succeed
- `rng`: Random seed for reproducibility

# Returns
- `results::Dict`: Complete search results
"""
function find_minimum_hidden_units(
    n::Int;
    hidden_candidates::Vector{Int} = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    num_trials::Int = 10,
    required_success_rate::Float64 = 0.8,
    rng = nothing,
    verbose::Bool = false
)
    # Set random seed if provided
    if rng !== nothing
        Random.seed!(rng)
    end

    # Generate dataset (use exhaustive for small n, sampled for large n)
    if 2^(2n) <= 100000
        X, Y = generate_cd_n_dataset(n)
        dataset_type = "exhaustive"
    else
        X, Y = generate_cd_n_dataset_sampled(n, 50000)
        dataset_type = "sampled"
    end

    input_dim = 2n
    results = Dict{String, Any}(
        "n" => n,
        "input_dim" => input_dim,
        "dataset_size" => size(X, 2),
        "dataset_type" => dataset_type,
        "search_results" => Dict{Int, Any}()
    )

    minimum_h = nothing

    for h in hidden_candidates
        if verbose
            println("Testing h=$h hidden units...")
        end

        trial_results = []
        successes = 0

        for trial in 1:num_trials
            # Create fresh model
            model = create_sigmoid_network(input_dim, h)

            # Train
            history, final_accuracy = train_sigmoid_network!(model, X, Y)

            is_success = final_accuracy >= 0.9999  # Essentially 100%
            if is_success
                successes += 1
            end

            push!(trial_results, Dict(
                "final_accuracy" => final_accuracy,
                "epochs" => length(history["loss"]),
                "success" => is_success
            ))
        end

        success_rate = successes / num_trials

        results["search_results"][h] = Dict(
            "trials" => trial_results,
            "success_rate" => success_rate,
            "mean_accuracy" => mean([t["final_accuracy"] for t in trial_results]),
            "meets_threshold" => success_rate >= required_success_rate
        )

        if verbose
            println("  Success rate: $(round(100*success_rate, digits=1))%")
        end

        # Check if this is the minimum
        if minimum_h === nothing && success_rate >= required_success_rate
            minimum_h = h
            if verbose
                println("  → Found minimum: h=$h")
            end
        end
    end

    results["minimum_hidden_units"] = minimum_h
    results["theoretical_lower_bound"] = sqrt(n)

    if minimum_h !== nothing
        results["empirical_to_theoretical_ratio"] = minimum_h / sqrt(n)
    end

    return results
end

# ============================================================================
# SCALING ANALYSIS
# ============================================================================

"""
    analyze_scaling(n_values; kwargs...)

Analyze how minimum hidden units scales with problem size n.

Fits empirical data to h = c × n^α and compares to theoretical √n bound.

# Arguments
- `n_values::Vector{Int}`: Problem sizes to test

# Returns
- `analysis::Dict`: Scaling analysis results
"""
function analyze_scaling(n_values::Vector{Int}; verbose::Bool = false, kwargs...)
    results = Dict{String, Any}(
        "n_values" => n_values,
        "empirical_minimums" => Int[],
        "theoretical_bounds" => Float64[]
    )

    for n in n_values
        if verbose
            println("\n=== Analyzing n=$n ===")
        end

        search_results = find_minimum_hidden_units(n; verbose=verbose, kwargs...)

        if search_results["minimum_hidden_units"] !== nothing
            push!(results["empirical_minimums"], search_results["minimum_hidden_units"])
            push!(results["theoretical_bounds"], sqrt(n))
        end
    end

    # Fit scaling law: log(h) = log(c) + α*log(n)
    if length(results["empirical_minimums"]) >= 2
        log_n = log.(n_values[1:length(results["empirical_minimums"])])
        log_h = log.(results["empirical_minimums"])

        # Simple linear regression
        n_points = length(log_n)
        mean_log_n = mean(log_n)
        mean_log_h = mean(log_h)

        numerator = sum((log_n .- mean_log_n) .* (log_h .- mean_log_h))
        denominator = sum((log_n .- mean_log_n).^2)

        α = numerator / denominator  # Scaling exponent
        log_c = mean_log_h - α * mean_log_n
        c = exp(log_c)  # Scaling constant

        results["fitted_exponent"] = α
        results["fitted_constant"] = c
        results["theoretical_exponent"] = 0.5  # √n means α = 0.5

        if verbose
            println("\n=== Scaling Analysis ===")
            println("Fitted: h ≈ $(round(c, digits=2)) × n^$(round(α, digits=3))")
            println("Theoretical: h = Ω(n^0.5)")
            println("Exponent comparison: $(round(α, digits=3)) vs 0.5")
        end
    end

    return results
end

# ============================================================================
# EXPORTS
# ============================================================================

export generate_cd_n_dataset, generate_cd_n_dataset_sampled
export create_sigmoid_network
export train_sigmoid_network!
export find_minimum_hidden_units
export analyze_scaling
