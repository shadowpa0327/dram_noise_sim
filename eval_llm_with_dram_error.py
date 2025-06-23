# Import necessary modules
import argparse
import torch
import lm_eval
from tqdm import tqdm
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import os
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from monkey_patch import patch_model_linear_layers


def load_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def run_lm_eval_zero_shot(model, tokenizer, batch_size=64, task_list=["arc_easy", "hellaswag"], limit=None):
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # indexes all tasks from the lm_eval/tasks subdirectory.
    # Alternatively, you can set TaskManager(include_path="path/to/my/custom/task/configs")
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logging.info(f"Evaluation, Task(s): {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            #model_args= "add_bos_token=True" if model_type == "jamba" else "",
            tasks=task_list,
            task_manager=task_manager,
            log_samples=False,
            limit=limit
        ) 

    res = make_table(results)
    print(res)
    
    return results['results']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        '--tasks', type=lambda s: [item for item in s.split(',')], default=[],
        help='Task to be evaled'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='batch size for lm_eval tasks'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save the results or not."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the .json results."
    )
    parser.add_argument(
        "--apply_monkey_patch",
        action="store_true",
        help="Whether to apply DRAM error monkey patching to the model."
    )
    parser.add_argument(
        "--dram_error_prob",
        type=float,
        default=1e-6,
        help="DRAM error probability per bit (default: 1e-6)."
    )
    parser.add_argument(
        "--dram_error_prob_file",
        type=str,
        default=None,
        help="Path to file containing DRAM error probability tensor (.pt or .pth file). If provided, this overrides --dram_error_prob."
    )
    parser.add_argument(
        "--protect_sign_and_exponent",
        action="store_true",
        help="Whether to protect the sign and exponent bits from DRAM errors."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    
    # Apply monkey patching if requested
    if args.apply_monkey_patch:
        # Determine error probability (scalar or tensor)
        if args.dram_error_prob_file is not None:
            logging.info(f"Loading DRAM error probabilities from file: {args.dram_error_prob_file}")
            try:
                error_prob = torch.load(args.dram_error_prob_file, map_location='cpu')
                
                # Validate tensor shape and values
                if not isinstance(error_prob, torch.Tensor):
                    raise ValueError(f"File must contain a torch.Tensor, got {type(error_prob)}")
                
                if error_prob.shape != (1024,):
                    raise ValueError(f"Error probability tensor must have shape (1024,), got {error_prob.shape}")
                
                if torch.any(error_prob < 0) or torch.any(error_prob > 1):
                    raise ValueError("All error probabilities must be in range [0, 1]")
                
                # Move to same device as model
                device = next(model.parameters()).device
                error_prob = error_prob.to(device)
                
                logging.info(f"Loaded error probability tensor with shape {error_prob.shape}")
                logging.info(f"Error prob stats - Min: {error_prob.min():.2e}, Max: {error_prob.max():.2e}, Mean: {error_prob.mean():.2e}")
                
            except Exception as e:
                logging.error(f"Failed to load error probability file: {e}")
                raise
        else:
            error_prob = args.dram_error_prob
            logging.info(f"Using scalar DRAM error probability: {error_prob}")
        
        logging.info("Applying DRAM error monkey patching...")
        
        # Hard-coded layer filter to exclude lm_head layers
        def layer_filter(name, module):
            # Exclude lm_head layers from patching
            return "lm_head" not in name.lower()
        
        logging.info("Excluding lm_head layers from patching")
        
        # Apply monkey patching
        patched_layers = patch_model_linear_layers(
            model,
            error_prob=error_prob,
            layer_filter=layer_filter,
            protect_sign_and_exponent=args.protect_sign_and_exponent
        )
        logging.info(f"Successfully patched {len(patched_layers)} linear layers")
    else:
        logging.info("Skipping monkey patching (not requested)")
    
    logging.info("Start running lm_eval zero-shot evaluation...")
    res = run_lm_eval_zero_shot(model, tokenizer, args.batch_size, task_list=args.tasks)
    
    # Save results if requested
    if args.save_results:
        # Create directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Save results to JSON file
        model_name = args.model_name_or_path.split("/")[-1]
        suffix = ""
        if args.apply_monkey_patch:
            if args.dram_error_prob_file is not None:
                # Use filename without extension for suffix
                file_basename = os.path.splitext(os.path.basename(args.dram_error_prob_file))[0]
                suffix = f"_dram_error_tensor_{file_basename}"
            else:
                suffix = f"_dram_error_{args.dram_error_prob}"
            
            # Add hint if sign and exponent bits are protected
            if args.protect_sign_and_exponent:
                suffix += "_protected_se"
        
        output_file = os.path.join(args.output_dir, f"{model_name}{suffix}.json")
        with open(output_file, "w") as f:
            json.dump(res, f, indent=4)

        print(f"Results saved to {output_file}")
    else:
        logging.info("Results not saved (save_results=False)")
    