# DRAM Error Simulation

## Cloning Repo
```
git clone --recurse-submodules https://github.com/shadowpa0327/dram_noise_sim.git
```

If you forgot to clone the submodule, you can initialize and update it with:
```bash
git submodule update --init --recursive
```

## Build Environments
```bash
conda create -n dram_error python=3.11
conda activate dram_error
pip install -r requirements.txt
pip install -e 3rdparty/lm-evaluation-harness
```

## Usage

The main script `eval_llm_with_dram_error.py` evaluates language models using the lm-evaluation-harness framework with optional DRAM error simulation through monkey patching.

### Basic Usage

```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path <model_path> \
    --tasks <task1,task2,...> \
    --batch_size 8
```

### Parameters

- `--model_name_or_path`: Path to the model checkpoint or Hugging Face model name
- `--tasks`: Comma-separated list of evaluation tasks (e.g., "arc_easy")
- `--batch_size`: Batch size for evaluation (default: 8)
- `--verbose`: Enable verbose statistic. With this the scripts will dump the statistic number of bit flip (default: False)
- `--save_results`: Save evaluation results to JSON file
- `--output_dir`: Directory to save results (default: "./results")
- `--apply_monkey_patch`: Enable DRAM error simulation
- `--dram_error_prob`: DRAM error probability per bit (default: 1e-6)
- `--dram_error_prob_file`: Path to tensor file (.pt/.pth) containing per-layer error probabilities
- `--protect_sign_and_exponent`: Protect sign and exponent bits from DRAM errors (mantissa only)

### Examples

#### 1. Basic evaluation without DRAM errors:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --tasks arc_easy \
    --batch_size 16 \
    --verbose \
    --save_results
```

#### 2. Evaluation with uniform DRAM error simulation:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --tasks arc_easy \
    --apply_monkey_patch \
    --dram_error_prob 1e-4 \
    --save_results
```

#### 3. Evaluation with custom error probability tensor:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --tasks arc_easy \
    --apply_monkey_patch \
    --dram_error_prob_file ./error_probs.pt \
    --save_results
```

#### 4. Evaluation with protected sign and exponent bits:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --tasks arc_easy \
    --apply_monkey_patch \
    --dram_error_prob 1e-4 \
    --protect_sign_and_exponent \
    --save_results
```

#### 5. Evalaution and Dump the Statistic of Bit Flips
```
python eval_llm_with_dram_error.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --tasks arc_easy \
    --apply_monkey_patch \
    --dram_error_prob 1e-4 \
    --protect_sign_and_exponent \
    --save_results \
    --verbose
```

### Supported Tasks

The script supports all tasks available in the lm-evaluation-harness framework, including:
- `arc_easy`, `arc_challenge`: ARC reasoning tasks
- `hellaswag`: Commonsense reasoning
- `mmlu`: Massive multitask language understanding
- `gsm8k`: Grade school math problems
- And many more...

### Output

Results are displayed in a formatted table and optionally saved as JSON files in the specified output directory. The filename format is:
- Without DRAM errors: `{model_name}.json`
- With uniform DRAM errors: `{model_name}_dram_error_{probability}.json`
- With tensor DRAM errors: `{model_name}_dram_error_tensor_{filename}.json`
- With protected sign/exponent: `{model_name}_dram_error_{probability}_protected_se.json`

### DRAM Error Simulation

When `--apply_monkey_patch` is enabled, the script:
1. Patches all linear layers in the model (excluding lm_head layers)
2. Simulates bit-flip errors during matrix multiplication
3. Supports both uniform error rates and per-layer custom error probability tensors
4. Error probability tensors must have shape `(1024,)` with values in range `[0, 1]`
5. Optionally protects sign and exponent bits from errors (using `--protect_sign_and_exponent`)

