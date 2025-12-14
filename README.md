# Reasoning with Sampling - Reproduction Package

This repository contains a clean implementation of **Power Sampling via MCMC (Metropolis-Hastings)** for sampling from the power distribution π(x) ∝ p(x)^α, as described in the paper.

## Overview

Power Sampling uses MCMC to sample from a sharpened version of an LLM's distribution, which can improve reasoning quality by concentrating probability mass on higher-likelihood sequences.

### Key Features

- **Power Sampling MCMC** (Algorithm 1): Blockwise sequence growth with MH refinement
- **Baseline methods**: Single-shot generation and majority voting
- **Experiment framework**: Configurable experiments with CSV output
- **Plotting**: Accuracy vs N_mcmc, time-accuracy trade-offs

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- NumPy, Matplotlib

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch transformers numpy matplotlib pyyaml

# Optional: Install as editable package
pip install -e .
```

### Environment Variables

```bash
# Set HuggingFace cache directory (optional)
export HF_HOME=/path/to/cache

# For offline use, download models first
export TRANSFORMERS_OFFLINE=1
```

## Quick Start

### 1. Verify Installation

```bash
python scripts/verify_implementation.py --model gpt2
```

### 2. Run Quick Test

```bash
python scripts/run_experiment.py --config configs/quick_test.yaml
```

### 3. Run Full Experiment

```bash
python scripts/run_experiment.py --config configs/exp_nmcmc_sweep.yaml
```

### 4. Generate Plots

```bash
python scripts/plot_results.py --input results/ --output results/figures/
```

## Usage

### Power Sampling MCMC

```python
from rws.models import ModelWrapper
from rws.power_mcmc import power_sampling_mcmc

# Load model
model = ModelWrapper.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

# Run Power Sampling
result = power_sampling_mcmc(
    prompt="Solve: What is 2 + 2? Answer: ",
    model_wrapper=model,
    alpha=4.0,      # Power parameter
    B=192,          # Block size
    N_mcmc=10,      # MH iterations per block
    T_max=1024,     # Max sequence length
)

print(f"Answer: {result['final_answer']}")
print(f"Time: {result['wall_time_s']:.2f}s")
print(f"Accept rate: {result['accept_rate']:.2%}")
```

### Baseline Methods

```python
from rws.baselines import run_grpo_single, run_grpo_majority_vote

# Single generation
result = run_grpo_single(prompt, model, max_new_tokens=1024)

# Majority voting (10 samples)
result = run_grpo_majority_vote(
    prompt, model, 
    n_samples=10,
    tie_break="logprob"  # or "shortest"
)
```

### Running Experiments

```bash
# Using config file
python scripts/run_experiment.py --config configs/exp_nmcmc_sweep.yaml

# With CLI overrides
python scripts/run_experiment.py \
    --method power_mcmc \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --nmcmc 5 \
    --n_questions 20 \
    --seed 42
```

## Algorithm 1: Power Sampling MCMC

### Target Distribution

We sample from the power distribution:
```
π(x) ∝ p(x)^α
```
where p(x) is the base LM probability. In log-space:
```
log π(x) = α · log p(x) + C
```

### Proposal Distribution

The proposal uses the same model with temperature τ = 1/α:
```
q(token | context) ∝ p(token | context)^α
```
Operationally: `generate(..., temperature=1/α, do_sample=True)`

### Algorithm Structure

```
Input: prompt, base_model, α, B, N_mcmc, T_max
Output: sequence sampled from π

1. Initialize x = prompt
2. For block k = 0, 1, 2, ...:
   a. Set target length L_k = len(prompt) + (k+1) * B
   b. Extend x to length L_k using proposal q
   c. For i = 1 to N_mcmc:
      i.   Sample cut position m ~ Uniform(len(prompt)+1, L_k)
      ii.  Keep prefix x_{1:m-1}
      iii. Propose x' by resampling suffix from q(·|prefix)
      iv.  Compute acceptance:
           log A = [log π(x') - log π(x)] + [log q(x|x') - log q(x'|x)]
      v.   Accept x' with probability min(1, exp(log A))
   d. If EOS or L_k >= T_max: break
3. Return x
```

### MH Acceptance Computation

The acceptance ratio involves:
- **π ratio**: `α * (log p(x') - log p(x))` for the suffix part
- **Proposal ratio**: `log q(old_suffix | prefix) - log q(new_suffix | prefix)`

Both forward and reverse proposal probabilities are computed by:
1. Concatenating prefix + suffix
2. Running teacher-forced forward pass
3. Computing sum of log probs for suffix tokens under temperature-scaled logits

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 4.0 | Power parameter (higher = sharper) |
| `B` | 192 | Block size for sequence growth |
| `N_mcmc` | 10 | MH iterations per block |
| `T_max` | 1024 | Maximum sequence length |
| `top_p` | 1.0 | Nucleus sampling (1.0 = disabled) |

### Config File Format (YAML)

```yaml
model_name: "Qwen/Qwen2.5-Math-1.5B-Instruct"
dataset_path: "data/sample_math.jsonl"
n_questions: 50

methods:
  - power_mcmc
  - grpo_vote

nmcmc_sweep: [0, 1, 2, 5, 10]

alpha: 4.0
B: 192
T_max: 1024

seed: 42
```

## Output Format

### CSV Columns

| Column | Description |
|--------|-------------|
| `id` | Question ID |
| `method` | Method name |
| `is_correct` | 1 if correct, 0 otherwise |
| `pred_answer` | Predicted answer |
| `gold_answer` | Gold answer |
| `wall_time_s` | Wall clock time (seconds) |
| `output_tokens` | Number of output tokens |
| `internal_generated_tokens` | Total tokens (including MH proposals) |
| `accept_rate` | MH acceptance rate |
| `N_mcmc` | Number of MH iterations |

## Timing Methodology

Timing follows these rules for accurate measurement:

1. **CUDA synchronization**: `torch.cuda.synchronize()` is called before starting and stopping timers when using GPU
2. **Wall clock**: Uses `time.perf_counter()` for high-resolution timing
3. **End-to-end**: Includes all computation (generation, MH steps, answer extraction)
4. **Majority voting**: Total time includes all N samples + voting

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()

# ... generation code ...

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

## Project Structure

```
reasoning_with_sampling_repro/
├── src/rws/
│   ├── __init__.py
│   ├── models.py           # Model wrapper
│   ├── sampling.py         # Suffix sampling utilities
│   ├── power_mcmc.py       # Main MCMC algorithm
│   ├── baselines.py        # GRPO baselines
│   ├── answer_extraction.py # Answer parsing
│   ├── metrics.py          # Accuracy computation
│   └── utils.py            # Utilities
├── scripts/
│   ├── run_experiment.py   # Main experiment runner
│   ├── plot_results.py     # Plotting script
│   └── verify_implementation.py
├── configs/
│   ├── exp_nmcmc_sweep.yaml
│   ├── quick_test.yaml
│   └── baseline_comparison.yaml
├── data/
│   └── sample_math.jsonl   # Sample math problems
├── tests/
│   └── test_core.py        # Unit tests
└── results/                 # Output directory
```

## Testing

### Run Unit Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_core.py
```

### Verification Checks

The implementation includes these correctness checks:

1. **Log prob consistency**: `teacher_forced_logp` with T=1 matches `teacher_forced_logp_with_temperature(T=1)`
2. **Suffix log prob**: `logprob_suffix_given_prefix` matches direct computation
3. **MH stability**: Acceptance ratios are finite and numerically stable
4. **NMCMC=0**: Reduces to proposal-only sampling (no MH steps)

## Reproducing Results

### Full Reproduction

```bash
# 1. Run N_mcmc sweep experiment
python scripts/run_experiment.py --config configs/exp_nmcmc_sweep.yaml

# 2. Generate plots
python scripts/plot_results.py --input results/

# 3. View results
ls results/figures/
# - accuracy_vs_nmcmc.png
# - time_vs_accuracy.png
```

### Expected Outputs

- **accuracy_vs_nmcmc.png**: Shows accuracy improving with more MH iterations
- **time_vs_accuracy.png**: Trade-off between computation time and accuracy
- **CSV files**: Detailed per-question results

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `T_max` or `B`, or use smaller model
2. **Slow generation**: Check GPU is being used (`torch.cuda.is_available()`)
3. **Import errors**: Ensure `src/` is in Python path or install as package

### Model Loading

```python
# Check available memory
import torch
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use smaller dtype
model = ModelWrapper.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.bfloat16
)
```

## Citation

If you use this code, please cite the original paper.

## License

MIT License
