# Maass Theorem Empirical Validation

**First empirical test of Maass's 1997 foundational theorem on spiking neuron computational power, after 28 years and thousands of citations.**

[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [The Theorem](#the-theorem)
- [Key Results](#key-results)
- [Hardware Used](#hardware-used)
- [Installation](#installation)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Citation](#citation)
- [References](#references)

---

## The Theorem

Wolfgang Maass proved in 1997 that:

1. A **single spiking neuron** with programmable delays can compute the Coincidence Detection function CD_n
2. Any **sigmoidal feedforward network** requires Ω(√n) hidden units to compute CD_n

This demonstrates a fundamental computational advantage of temporal (spike-timing) coding over rate-based representations.

**CD_n Definition:**
```
CD_n: {0,1}^{2n} → {0,1}
CD_n(x₁,...,xₙ, y₁,...,yₙ) = 1  iff  ∃i: xᵢ = yᵢ = 1
```

---

## Key Results

| n | Spiking Neuron | Sigmoid Network | Theoretical √n |
|---|----------------|-----------------|----------------|
| 4 | 1 neuron, **100%** accuracy on 256 tests | 4 hidden units required | 2.00 |
| 6 | 1 neuron, **100%** accuracy on 4,096 tests | 16 hidden units required | 2.45 |
| 8 | 1 neuron, **100%** accuracy on 65,536 tests | 16 hidden units required | 2.83 |

**Total: 69,888 test cases with 100% accuracy using a single spiking neuron.**

---

## Hardware Used

This research was conducted on:

| Component | Specification |
|-----------|---------------|
| **Machine** | Mac Studio (2024) |
| **Chip** | Apple M3 Ultra |
| **CPU Cores** | 32 (24 performance + 8 efficiency) |
| **Memory** | 512 GB Unified Memory |
| **OS** | macOS 26.0.1 (Tahoe) |

**Note:** The core Maass validation (Julia) runs on any modern machine. The extended experiments (Python/PyTorch) benefit from GPU acceleration but work on CPU.

---

## Installation

### Prerequisites

- **Julia 1.10+** (tested on 1.12.3)
- **Python 3.11+** (tested on 3.13.7)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MaassTheorem.git
cd MaassTheorem
```

### Step 2: Set Up Julia Environment

```bash
# Start Julia
julia

# In Julia REPL:
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This installs all Julia dependencies:
- `DifferentialEquations.jl` - ODE solving with continuous callbacks
- `Flux.jl` - Neural network training for sigmoid baseline
- `CairoMakie.jl` - Plotting
- `DrWatson.jl` - Scientific project management
- `JSON3.jl` - Results serialization
- `StableRNGs.jl` - Reproducible random numbers

### Step 3: Set Up Python Environment (Optional - for extended experiments)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch>=2.0.0 transformers>=4.35.0 datasets>=2.15.0 \
            sentence-transformers>=2.2.0 scipy>=1.10.0 numpy>=1.24.0 tqdm>=4.65.0

# Or use requirements file
pip install -r pytorch/requirements_experiment.txt
```

### Step 4: Verify Installation

```bash
# Test Julia installation
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'

# Test Python installation (optional)
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Reproducing Results

### Quick Validation (< 1 minute)

```julia
# In Julia REPL
using Pkg; Pkg.activate(".")
include("scripts/run_experiment.jl")
quick_validation()
```

This tests n=4 only (256 input combinations).

### Full Maass Validation (< 5 minutes)

```julia
# In Julia REPL
using Pkg; Pkg.activate(".")
include("scripts/run_experiment.jl")
run_full_experiment()
```

This runs the complete validation for n ∈ {4, 6, 8}:
- Tests all 69,888 input combinations
- Trains sigmoid networks to find minimum hidden units
- Validates continuous-time precision
- Saves results to `data/results/`

### Individual Components

```julia
# Test spiking neuron CD_n for specific n
include("src/cd_n.jl")
results = test_cd_n_exhaustive(6; verbose=true)

# Find minimum sigmoid hidden units for specific n
include("src/sigmoid_network.jl")
results = find_minimum_hidden_units(6; verbose=true)

# Run precision validation
include("src/precision_validation.jl")
passed, report = validate_continuous_time_precision()
```

### View Existing Results

```julia
using JSON3
results = JSON3.read(read("data/results/maass_validation_20251130_manual.json", String))
println(results)
```

---

## Project Structure

```
MaassTheorem/
├── README.md                 # This file
├── Project.toml              # Julia dependencies
├── Manifest.toml             # Julia dependency lock file
│
├── src/                      # Core implementation
│   ├── spiking_neuron.jl     # LIF neuron with continuous-time ODE integration
│   ├── cd_n.jl               # CD_n function and exhaustive testing
│   ├── sigmoid_network.jl    # Sigmoid baseline for comparison
│   └── precision_validation.jl  # Anti-Morrison validation metrics
│
├── scripts/                  # Experiment runners
│   ├── run_experiment.jl     # Main Maass validation experiment
│   ├── plot_results.jl       # Visualization
│   ├── high_n_spiking_test.jl # Extended n tests
│   └── ...                   # Additional experiments
│
├── test/                     # Unit tests
│   ├── runtests.jl
│   ├── test_spiking.jl
│   ├── test_cd_n.jl
│   └── test_precision.jl
│
├── data/results/             # Experiment outputs
│   ├── maass_validation_20251130_manual.json  # Main results
│   └── ...
│
├── paper/                    # Publication materials
│   ├── maass_validation_paper.tex  # LaTeX paper
│   └── PAPER_SUMMARY.md
│
└── pytorch/                  # Extended Python experiments
    ├── requirements_experiment.txt
    └── ...
```

---

## How It Works

### Key Innovation: Continuous-Time Simulation

Unlike discrete-time simulators (which miss spikes with probability ~2.3×10⁻⁴ per timestep), we use:

1. **ODE Integration**: Vern8 solver (8th-order Runge-Kutta) with tolerance 10⁻¹⁰
2. **Exact Spike Detection**: ContinuousCallback with rootfinding
3. **Float64 Precision**: ~15 significant digits, matching ion channel dynamics

### CD_n Implementation

```
Input: x = [x₁,...,xₙ], y = [y₁,...,yₙ] ∈ {0,1}ⁿ

1. Assign matched delays: Δ_xᵢ = Δ_yᵢ = 5 + (i-1)×15 ms
2. Generate input spikes at t=1ms for all xᵢ=1 and yᵢ=1
3. If ∃i: xᵢ=yᵢ=1, both spikes arrive simultaneously → EPSPs sum → threshold exceeded
4. If no coincidence, individual EPSPs stay subthreshold
5. Output: 1 if neuron fires, 0 otherwise
```

### Anti-Morrison Validation

We verify continuous-time operation by checking:
- No grid quantization (spike times don't align with 0.1, 0.01, 0.001 ms grids)
- High SNR timing precision (>100 dB)
- Perfect reproducibility across runs

---

## Running Tests

```bash
# Run all Julia tests
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'

# Run specific test file
julia --project=. test/test_spiking.jl
julia --project=. test/test_cd_n.jl
julia --project=. test/test_precision.jl
```

---

## Troubleshooting

### Julia Package Installation Fails

```julia
# Clear package cache and retry
using Pkg
Pkg.gc()
Pkg.instantiate()
```

### DifferentialEquations.jl Compilation Slow

First run takes 2-5 minutes for precompilation. Subsequent runs are fast.

```julia
# Precompile explicitly
using Pkg; Pkg.activate(".")
using DifferentialEquations  # Wait for precompilation
```

### Out of Memory for Large n

For n > 10, use sampling instead of exhaustive testing:

```julia
include("src/sigmoid_network.jl")
X, Y = generate_cd_n_dataset_sampled(12, 50000)  # 50k samples instead of 2^24
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{maass_validation_2025,
  title={Empirical Validation of Maass's Theorem on the Computational Power of Spiking Neurons},
  author={Mouktar, Albert},
  year={2025},
  note={First empirical test of Maass 1997 theorem after 28 years}
}
```

---

## References

- **Maass, W. (1997)**. Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.

- **Maass, W. (1996)**. Lower bounds for the computational power of networks of spiking neurons. *Neural Computation*, 8(1), 1-40.

- **Morrison, A., et al. (2007)**. Exact subthreshold integration with continuous spike times in discrete-time neural network simulations. *Neural Computation*, 19(1), 47-79.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please open an issue or pull request.

Key areas for contribution:
- Testing larger n values with sampling
- Implementing delay learning (STDP)
- Neuromorphic hardware deployment
- Additional Boolean functions (Θ_k, etc.)
