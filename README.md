# DRAM Error Simulation

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
- `--tasks`: Comma-separated list of evaluation tasks (e.g., "arc_easy,hellaswag")
- `--batch_size`: Batch size for evaluation (default: 8)
- `--verbose`: Enable verbose logging output
- `--save_results`: Save evaluation results to JSON file
- `--output_dir`: Directory to save results (default: "./results")
- `--apply_monkey_patch`: Enable DRAM error simulation
- `--dram_error_prob`: DRAM error probability per bit (default: 1e-5)
- `--dram_error_prob_file`: Path to tensor file (.pt/.pth) containing per-layer error probabilities

### Examples

#### 1. Basic evaluation without DRAM errors:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path microsoft/DialoGPT-medium \
    --tasks arc_easy,hellaswag \
    --batch_size 16 \
    --verbose \
    --save_results
```

#### 2. Evaluation with uniform DRAM error simulation:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path microsoft/DialoGPT-medium \
    --tasks arc_easy,hellaswag \
    --apply_monkey_patch \
    --dram_error_prob 1e-4 \
    --save_results
```

#### 3. Evaluation with custom error probability tensor:
```bash
python eval_llm_with_dram_error.py \
    --model_name_or_path microsoft/DialoGPT-medium \
    --tasks arc_easy,hellaswag \
    --apply_monkey_patch \
    --dram_error_prob_file ./error_probs.pt \
    --save_results
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

### DRAM Error Simulation

When `--apply_monkey_patch` is enabled, the script:
1. Patches all linear layers in the model (excluding lm_head layers)
2. Simulates bit-flip errors during matrix multiplication
3. Supports both uniform error rates and per-layer custom error probability tensors
4. Error probability tensors must have shape `(1024,)` with values in range `[0, 1]`

