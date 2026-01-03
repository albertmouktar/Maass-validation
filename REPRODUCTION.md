# Step-by-Step Reproduction Guide

This guide provides exact commands to reproduce the Maass theorem validation results.

---

## Prerequisites Checklist

- [ ] Julia 1.10+ installed (`julia --version`)
- [ ] Python 3.11+ installed (`python3 --version`)
- [ ] Git installed (`git --version`)
- [ ] ~2GB disk space for dependencies

---

## Step 1: Get the Code

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MaassTheorem.git
cd MaassTheorem
```

---

## Step 2: Install Julia Dependencies

```bash
# Start Julia
julia

# Inside Julia REPL, run these commands:
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Verify packages installed (this precompiles - takes 2-5 min first time)
using DifferentialEquations
using Flux
using JSON3
println("Julia setup complete!")
```

**Expected output:** No errors, "Julia setup complete!" printed.

---

## Step 3: Run Quick Validation (1 minute)

```julia
# Still in Julia REPL
include("scripts/run_experiment.jl")
quick_validation()
```

**Expected output:**
```
Running quick validation (n=4 only)...
...
QUICK VALIDATION RESULTS
==================================================
Spiking accuracy: 100.0%
✓ Spiking implementation validated
```

---

## Step 4: Run Full Experiment (5-10 minutes)

```julia
# Still in Julia REPL
run_full_experiment()
```

**Expected output:**
```
======================================================================
MAASS THEOREM EMPIRICAL VALIDATION
First empirical test in 28 years (since 1997)
======================================================================

--- Summary for n=4 ---
  Spiking: 1 neuron, 100.0% accuracy
  Sigmoid: 4 hidden units required
  Ratio to √n: 2.0×

--- Summary for n=6 ---
  Spiking: 1 neuron, 100.0% accuracy
  Sigmoid: 16 hidden units required
  Ratio to √n: 6.53×

--- Summary for n=8 ---
  Spiking: 1 neuron, 100.0% accuracy
  Sigmoid: 16 hidden units required
  Ratio to √n: 5.66×
```

---

## Step 5: Verify Results File

```julia
# Check results were saved
using JSON3
results = JSON3.read(read("data/results/maass_validation_20251130_manual.json", String))
println("Spiking accuracy n=8: ", results.results_by_n["8"].spiking.accuracy)
```

**Expected output:** `Spiking accuracy n=8: 1.0`

---

## Step 6: Run Unit Tests

```bash
# Exit Julia (Ctrl+D) and run from terminal
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```

**Expected output:** All tests pass.

---

## Step 7: (Optional) Python Extended Experiments

```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r pytorch/requirements_experiment.txt

# Run a Python experiment (example)
python3 scripts/generate_sbert_embeddings.py
```

---

## Validation Checklist

After completing all steps, verify:

| Check | Expected Result |
|-------|-----------------|
| `data/results/` contains JSON files | Yes |
| Spiking n=4 accuracy | 100% (256/256) |
| Spiking n=6 accuracy | 100% (4096/4096) |
| Spiking n=8 accuracy | 100% (65536/65536) |
| Sigmoid n=4 min hidden | 4 |
| Sigmoid n=6 min hidden | 16 |
| Sigmoid n=8 min hidden | 16 |
| All unit tests pass | Yes |

---

## Publishing to GitHub

### Initialize Git (if not already)

```bash
cd MaassTheorem
git init
git add .
git commit -m "Initial commit: Maass theorem empirical validation"
```

### Create GitHub Repository

1. Go to https://github.com/new
2. Name: `MaassTheorem`
3. Description: "First empirical validation of Maass 1997 theorem on spiking neuron computational power"
4. Public or Private (your choice)
5. Do NOT initialize with README (we have one)
6. Click "Create repository"

### Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/MaassTheorem.git
git branch -M main
git push -u origin main
```

---

## Troubleshooting

### "Package not found" errors

```julia
Pkg.gc()  # Clean cache
Pkg.instantiate()  # Reinstall
```

### Slow first run

DifferentialEquations.jl takes 2-5 minutes to precompile on first use. This is normal.

### Memory issues for large n

```julia
# For n > 10, use sampling
include("src/sigmoid_network.jl")
X, Y = generate_cd_n_dataset_sampled(12, 50000)
```

### Python import errors

```bash
# Make sure venv is activated
source .venv/bin/activate
pip install --upgrade pip
pip install -r pytorch/requirements_experiment.txt
```

---

## Hardware Requirements

### Minimum (Core Maass validation)
- 4GB RAM
- Any modern CPU
- ~2GB disk space

### Recommended (Extended experiments)
- 16GB+ RAM
- Multi-core CPU or GPU
- ~10GB disk space

### Used in paper
- Mac Studio M3 Ultra
- 512GB RAM
- 32 CPU cores

---

## Time Estimates

| Task | Time |
|------|------|
| Julia package install | 5-10 min |
| Quick validation (n=4) | < 1 min |
| Full experiment (n=4,6,8) | 5-10 min |
| Python setup | 5-10 min |
| Extended experiments | Hours |

---

## Contact

For issues reproducing results, please open a GitHub issue with:
1. Your OS and hardware
2. Julia/Python versions
3. Full error message
4. Steps you followed
