import os
import torch
import argparse

from lcb_runner.utils.scenarios import Scenario


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="If you have a local model, specify it here in conjunction with --model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="trust_remote_code option used in huggingface models",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--not_fast",
        action="store_true",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="whether to use CoT in code execution scenario",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--codegen_n",
        type=int,
        default=10,
        help="Number of samples for which code generation was run (used to map the code generation file during self-repair)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--multiprocess",
        default=0,
        type=int,
        help="Number of processes to use for generation (vllm runs do not use this)",
    )
    parser.add_argument(
        "--stop",
        default="###",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )
    parser.add_argument("--continue_existing", action="store_true")
    parser.add_argument("--continue_existing_with_eval", action="store_true")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use cache for generation"
    )
    parser.add_argument(
        "--cache_batch_size", type=int, default=100, help="Batch size for caching"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results")
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument("--timeout", type=int, default=6, help="Timeout for evaluation")
    parser.add_argument(
        "--openai_timeout", type=int, default=90, help="Timeout for requests to OpenAI"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=-1,
        help="Tensor parallel size for vllm",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching for vllm",
    )
    parser.add_argument(
        "--custom_output_file",
        type=str,
        default=None,
        help="Path to the custom output file used in `custom_evaluator.py`",
    )
    parser.add_argument(
        "--custom_output_save_name",
        type=str,
        default=None,
        help="Folder name to save the custom output results (output file folder modified if None)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for vllm")
    # Added to avoid running extra generations (it's slow for reasoning models)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--trt_rounds",
        type=int,
        default=3,
        help="Number of TRT rounds to run",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking for vllm",
    )
    parser.add_argument(
        "--multiprocess_oai",
        default=0,
        type=int,
        help="Number of processes to use for generation (vllm runs do not use this)",
    )
    parser.add_argument(
        "--logging_trace",
        action="store_true",
        help="Enable detailed trace logging for each solution attempt",
    )
    parser.add_argument(
        "--eval_all_rollouts",
        action="store_true",
        help="Evaluate all rollout solutions and log per-rollout correctness (requires --logging_trace)",
    )
    parser.add_argument(
        "--roll_out_n",
        type=int,
        default=1,
        help="Number of parallel solver roll-outs before ranking with the knowledge manager",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
        help="Filter benchmark to only run problems of specified difficulty (easy/medium/hard)",
    )
    parser.add_argument(
        "--reference_sol_in_solver",
        action="store_true",
        help="Show reference solution to the solver agent. If not set, uses placeholder text.",
    )
    parser.add_argument(
        "--enable_strategy",
        action="store_true",
        help="Enable strategy generation by KM to guide solver rollouts in subsequent rounds",
    )
    parser.add_argument(
        "--enable_test_gen",
        action="store_true",
        help="Enable self-generated test cases: KM generates tests, executes them, and ranks solutions by pass rate",
    )
    parser.add_argument(
        "--use_shared_knowledge",
        action="store_true",
        help="Enable saving and using shared knowledge (general reflections) across all problems. If not set, general_reflection is ignored.",
    )
    # Rate limiting arguments for smoother API call distribution
    parser.add_argument(
        "--stagger_delay",
        type=float,
        default=0.5,
        help="Delay in seconds between launching each problem task (default: 0.5). Higher values spread API calls more evenly.",
    )
    parser.add_argument(
        "--post_acquire_delay",
        type=float,
        default=0.2,
        help="Delay in seconds after acquiring semaphore before making API call (default: 0.2). Helps prevent burst when semaphore releases.",
    )
    parser.add_argument(
        "--rollout_stagger_min",
        type=float,
        default=2.0,
        help="Minimum stagger delay in seconds between rollouts within a problem (default: 2.0).",
    )
    parser.add_argument(
        "--rollout_stagger_max",
        type=float,
        default=5.0,
        help="Maximum stagger delay in seconds between rollouts within a problem (default: 5.0).",
    )

    args = parser.parse_args()

    args.stop = args.stop.split(",")

    if args.tensor_parallel_size == -1:
        args.tensor_parallel_size = torch.cuda.device_count()

    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()

    return args


def test():
    args = get_args()
    print(args)


if __name__ == "__main__":
    test()
