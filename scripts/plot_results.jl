"""
Visualization for Maass Theorem Validation Results

Generates publication-quality figures:
1. Scaling comparison: Empirical minimum h vs theoretical Ω(√n)
2. Membrane potential traces with spike detection
3. Precision validation plots
"""

using DrWatson
@quickactivate "MaassTheorem"

using CairoMakie
using JSON3
using Statistics

# ============================================================================
# PUBLICATION THEME
# ============================================================================

function set_publication_theme!()
    set_theme!(Theme(
        fontsize = 14,
        Axis = (
            xlabelsize = 16,
            ylabelsize = 16,
            titlesize = 18,
            xticklabelsize = 12,
            yticklabelsize = 12,
        ),
        Legend = (
            framevisible = false,
            labelsize = 12,
        )
    ))
end

# ============================================================================
# SCALING PLOT
# ============================================================================

"""
    plot_scaling(results; save_path=nothing)

Plot empirical minimum hidden units vs theoretical √n bound.

Creates a log-log plot comparing:
- Empirical data points
- Theoretical Ω(√n) reference line
- Fitted scaling law
"""
function plot_scaling(results::Dict; save_path::Union{Nothing, String} = nothing)
    set_publication_theme!()

    # Extract data
    n_values = Int[]
    h_values = Float64[]

    for (n, r) in results["results_by_n"]
        h = r["sigmoid"]["minimum_hidden_units"]
        if h !== nothing
            push!(n_values, n)
            push!(h_values, Float64(h))
        end
    end

    if isempty(n_values)
        @warn "No valid results to plot"
        return nothing
    end

    # Sort by n
    perm = sortperm(n_values)
    n_values = n_values[perm]
    h_values = h_values[perm]

    # Theoretical bound
    n_theory = range(minimum(n_values), maximum(n_values), length=100)
    h_theory = sqrt.(n_theory)

    # Scale theoretical line to pass through first empirical point
    scale_factor = h_values[1] / sqrt(n_values[1])
    h_theory_scaled = scale_factor .* h_theory

    # Create figure
    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1],
        xlabel = "Problem size n",
        ylabel = "Hidden units h",
        title = "Sigmoid Network Requirements for CD_n",
        xscale = log10,
        yscale = log10
    )

    # Plot theoretical bound
    lines!(ax, n_theory, h_theory_scaled,
        color = :gray,
        linestyle = :dash,
        linewidth = 2,
        label = "Theoretical Ω(√n)")

    # Plot empirical data
    scatter!(ax, n_values, h_values,
        color = :blue,
        markersize = 15,
        label = "Empirical minimum")

    # Add fitted line if we have scaling analysis
    if haskey(results, "scaling_analysis")
        α = results["scaling_analysis"]["fitted_exponent"]
        c = results["scaling_analysis"]["fitted_constant"]
        h_fitted = c .* n_theory.^α

        lines!(ax, n_theory, h_fitted,
            color = :red,
            linewidth = 2,
            label = "Fitted: h ∝ n^$(round(α, digits=2))")
    end

    axislegend(ax, position = :lt)

    # Save if path provided
    if save_path !== nothing
        save(save_path, fig)
        # Also save PNG
        png_path = replace(save_path, ".pdf" => ".png")
        save(png_path, fig, px_per_unit=3)  # 300 DPI equivalent
    end

    return fig
end

# ============================================================================
# MEMBRANE TRACE PLOT
# ============================================================================

"""
    plot_membrane_trace(sol, params, spike_times; save_path=nothing)

Plot membrane potential trajectory with spike markers.

Shows:
- Membrane potential over time
- Threshold line
- Detected spike times
"""
function plot_membrane_trace(
    sol,
    params,
    spike_times::Vector{Float64};
    save_path::Union{Nothing, String} = nothing,
    title::String = "LIF Neuron Membrane Potential"
)
    set_publication_theme!()

    # Extract data
    t = sol.t
    V = [u[1] for u in sol.u]

    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Time (ms)",
        ylabel = "Membrane potential (normalized)",
        title = title
    )

    # Plot membrane potential
    lines!(ax, t, V,
        color = :blue,
        linewidth = 1.5,
        label = "V(t)")

    # Plot threshold
    hlines!(ax, [params.V_th],
        color = :red,
        linestyle = :dash,
        linewidth = 1.5,
        label = "Threshold")

    # Mark spike times
    if !isempty(spike_times)
        scatter!(ax, spike_times, fill(params.V_th, length(spike_times)),
            color = :red,
            marker = :star5,
            markersize = 15,
            label = "Spikes")
    end

    axislegend(ax, position = :rt)

    if save_path !== nothing
        save(save_path, fig)
        png_path = replace(save_path, ".pdf" => ".png")
        save(png_path, fig, px_per_unit=3)
    end

    return fig
end

# ============================================================================
# PRECISION VALIDATION PLOT
# ============================================================================

"""
    plot_precision_comparison(snr_values, labels; save_path=nothing)

Plot SNR comparison between our approach and Morrison's limits.
"""
function plot_precision_comparison(
    snr_values::Vector{Float64},
    labels::Vector{String};
    save_path::Union{Nothing, String} = nothing
)
    set_publication_theme!()

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Method",
        ylabel = "SNR (dB)",
        title = "Timing Precision Comparison",
        xticks = (1:length(labels), labels)
    )

    # Reference lines
    hlines!(ax, [96], color = :green, linestyle = :dash, label = "CD audio (16-bit)")
    hlines!(ax, [36], color = :orange, linestyle = :dash, label = "Morrison effective limit")

    # Bar plot
    barplot!(ax, 1:length(snr_values), snr_values,
        color = [:blue, :gray][1:length(snr_values)])

    axislegend(ax, position = :rt)

    if save_path !== nothing
        save(save_path, fig)
        png_path = replace(save_path, ".pdf" => ".png")
        save(png_path, fig, px_per_unit=3)
    end

    return fig
end

# ============================================================================
# RESULTS TABLE
# ============================================================================

"""
    print_results_table(results)

Print formatted results table.
"""
function print_results_table(results::Dict)
    println("\n" * "="^80)
    println("MAASS THEOREM VALIDATION RESULTS")
    println("="^80)

    println("\n┌─────┬──────────────────┬─────────────────┬───────────────┬─────────┐")
    println("│  n  │ Spiking Accuracy │ Min Hidden (h)  │ Theoretical √n│  Ratio  │")
    println("├─────┼──────────────────┼─────────────────┼───────────────┼─────────┤")

    for n in sort(collect(keys(results["results_by_n"])))
        r = results["results_by_n"][n]
        acc = round(100 * r["spiking"]["accuracy"], digits=1)
        h = r["sigmoid"]["minimum_hidden_units"]
        h_str = h !== nothing ? string(h) : "N/A"
        theory = round(sqrt(n), digits=2)
        ratio = h !== nothing ? round(h / sqrt(n), digits=2) : "N/A"

        println("│ $(lpad(n, 3)) │ $(lpad(acc, 14))% │ $(lpad(h_str, 15)) │ $(lpad(theory, 13)) │ $(lpad(ratio, 7)) │")
    end

    println("└─────┴──────────────────┴─────────────────┴───────────────┴─────────┘")

    if haskey(results, "scaling_analysis")
        sa = results["scaling_analysis"]
        println("\nScaling Analysis:")
        println("  Fitted exponent α = $(round(sa["fitted_exponent"], digits=3))")
        println("  Theoretical exponent = 0.5")
        println("  Consistent with theory: $(sa["exponent_consistent_with_theory"] ? "YES" : "NO")")
    end
end

# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

"""
    generate_all_plots(results_path; output_dir=nothing)

Generate all plots from experiment results.
"""
function generate_all_plots(results_path::String; output_dir::Union{Nothing, String} = nothing)
    # Load results
    results = JSON3.read(read(results_path, String))
    results = Dict(results)  # Convert to regular Dict

    # Set output directory
    if output_dir === nothing
        output_dir = datadir("figures")
    end
    mkpath(output_dir)

    println("Generating plots...")

    # 1. Scaling plot
    fig_scaling = plot_scaling(results;
        save_path = joinpath(output_dir, "scaling_comparison.pdf"))
    println("  ✓ Scaling comparison plot")

    # 2. Print results table
    print_results_table(results)

    println("\nPlots saved to: $output_dir")

    return fig_scaling
end

# ============================================================================
# ENTRY POINT
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Find most recent results file
    results_dir = datadir("results")

    if isdir(results_dir)
        files = filter(f -> endswith(f, ".json"), readdir(results_dir))

        if !isempty(files)
            latest = joinpath(results_dir, sort(files)[end])
            println("Plotting results from: $latest")
            generate_all_plots(latest)
        else
            println("No results files found in $results_dir")
            println("Run the experiment first: julia scripts/run_experiment.jl")
        end
    else
        println("Results directory not found: $results_dir")
        println("Run the experiment first: julia scripts/run_experiment.jl")
    end
end
