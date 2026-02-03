import os
import json
import re
from datetime import datetime

import numpy as np

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)
from lcb_runner.utils.extraction_utils import extract_code, LMStyle

def convert_messages_to_dict(harmony_messages):
    """Convert Message objects to JSON-serializable dictionaries."""
    serializable_messages = []
    for conversation in harmony_messages:
        conversation_dict = []
        for message in conversation:
            msg_dict = {}
            # Extract role
            if hasattr(message, 'author') and hasattr(message.author, 'role'):
                msg_dict['role'] = str(message.author.role)
                if hasattr(message.author, 'name') and message.author.name:
                    msg_dict['name'] = message.author.name

            # Extract content
            if hasattr(message, 'content'):
                content_list = []
                if isinstance(message.content, list):
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            content_list.append({'type': 'text', 'text': content_item.text})
                        else:
                            content_list.append(str(content_item))
                else:
                    content_list.append(str(message.content))
                msg_dict['content'] = content_list

            # Extract recipient if present (for MCP tool calls)
            if hasattr(message, 'recipient') and message.recipient:
                msg_dict['recipient'] = message.recipient

            # Extract tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                msg_dict['tool_calls'] = []
                for tool_call in message.tool_calls:
                    tool_dict = {
                        'id': getattr(tool_call, 'id', ''),
                        'type': getattr(tool_call, 'type', 'function'),
                    }
                    if hasattr(tool_call, 'function'):
                        tool_dict['function'] = {
                            'name': getattr(tool_call.function, 'name', ''),
                            'arguments': getattr(tool_call.function, 'arguments', ''),
                        }
                    msg_dict['tool_calls'].append(tool_dict)

            conversation_dict.append(msg_dict)
        serializable_messages.append(conversation_dict)
    return serializable_messages


def _evaluate_all_rollouts_batched(remaining_benchmark, all_rollout_solutions, args, model):
    """Evaluate all rollouts from all problems in a single batched codegen_metrics call.

    This batches all problems together into one call, creating only one ProcessPoolExecutor
    and parallelizing across all rollouts from all problems simultaneously.

    Args:
        remaining_benchmark: List of benchmark instances
        all_rollout_solutions: List of rollout solution lists (one per problem)
        args: Command line arguments
        model: The language model

    Returns:
        Dict mapping problem index to list of 0/1 correctness values
    """
    # Collect all problems and their codes
    eval_samples = []
    all_codes = []
    valid_indices_per_problem = []  # Track which rollout indices had valid code
    problem_indices = []  # Track original problem index for each entry

    for idx in range(len(remaining_benchmark)):
        if idx >= len(all_rollout_solutions) or not all_rollout_solutions[idx]:
            continue

        instance = remaining_benchmark[idx]
        rollout_solutions = all_rollout_solutions[idx]

        codes = []
        valid_indices = []
        for i, sol in enumerate(rollout_solutions):
            if sol and sol.strip():
                code = extract_code(sol, model.model_style)
                if code:
                    codes.append(code)
                    valid_indices.append(i)

        if codes:
            eval_samples.append(instance.get_evaluation_sample())
            all_codes.append(codes)
            valid_indices_per_problem.append((valid_indices, len(rollout_solutions)))
            problem_indices.append(idx)

    if not eval_samples:
        return {}

    # Single batched call for all problems
    try:
        print(f"[Trace] Evaluating {len(eval_samples)} problems with {sum(len(c) for c in all_codes)} total rollouts")
        _, results_dict, _ = codegen_metrics(
            eval_samples,
            all_codes,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"[Trace] Warning: Failed to evaluate rollout batch: {e}")
        return {}

    # Map results back to original problem indices
    results = {}
    for batch_idx, orig_idx in enumerate(problem_indices):
        valid_indices, num_rollouts = valid_indices_per_problem[batch_idx]
        rollout_accs = [0] * num_rollouts

        if batch_idx in results_dict:
            for j, orig_rollout_idx in enumerate(valid_indices):
                if j < len(results_dict[batch_idx]):
                    gen_results = np.array(results_dict[batch_idx][j])
                    rollout_accs[orig_rollout_idx] = 1 if np.all(gen_results > 0) else 0

        results[orig_idx] = rollout_accs

    return results


def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode with {args.trt_rounds} TRT rounds")
        benchmark = benchmark[-16:] # originally 15

    # Get original output path
    original_output_path = get_output_path(model.model_repr, args)

    # Create timestamp-based folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set timestamp for MCP server knowledge base (must be set before server connects)
    os.environ["KNOWLEDGE_BASE_TIMESTAMP"] = timestamp

    # Set shared knowledge flag for MCP server
    os.environ["USE_SHARED_KNOWLEDGE"] = "1" if args.use_shared_knowledge else "0"

    output_dir = os.path.dirname(original_output_path)
    output_filename = os.path.basename(original_output_path)

    # Create new directory with timestamp
    if args.reference_sol_in_solver:
        timestamp =  timestamp + "_ref_in"
    else:
        timestamp =  timestamp + "_no_ref"
    
    if args.roll_out_n > 1:
        timestamp =  timestamp + "_rolln_" + str(args.roll_out_n)

    if args.enable_strategy:
        timestamp = timestamp + "_strategyOn"

    if args.enable_test_gen:
        timestamp = timestamp + "_testOn"

    timestamped_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    print(f"Output files will be saved to: {timestamped_dir}")

    # Update output paths to use timestamp folder
    output_path = os.path.join(timestamped_dir, output_filename)
    eval_file = output_path.replace(".json", "_kflow_eval_results.json")
    eval_all_file = output_path.replace(".json", "_kflow_eval_all.json")
    eval_sample_file = output_path.replace(".json", "_kflow_eval_sample.json")

    # Create trace directory if logging is enabled
    if args.logging_trace:
        trace_dir = os.path.join(timestamped_dir, "trace")
        os.makedirs(trace_dir, exist_ok=True)
        print(f"Trace logging enabled. Logs will be saved to: {trace_dir}")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        # Build runner - MCP servers will persist across rounds
        runner = build_runner(args, model)

        # KNOWLEDGE FLOW ROUNDS
        # Each round, the runner reuses the same MCP servers
        # Knowledge accumulates in the persistent MCP servers
        for round_idx in range(args.trt_rounds):
            print(f"\n{'='*60}")
            print(f"TRT Round {round_idx + 1}/{args.trt_rounds}")
            print(f"{'='*60}")

            # Run the model
            # For round 1: Creates new MCP servers (empty knowledge)
            # For round 2+: Reuses existing MCP servers (accumulated knowledge)
            results_dict = runner.run_main(remaining_benchmark, format_prompt)

            # Extract outputs and messages
            # Knowledge is managed by persistent MCP servers, not returned here
            results = results_dict["outputs"]
            messages = results_dict.get("messages", [])

            # Combine and evaluate results
            combined_results = combine_results(
                args.scenario, results, model, args.cot_code_execution
            )
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)

            # Save metrics after each round
            eval_file_kflow = eval_file.replace(".json", f"_round{round_idx+1}.json")

            # Add conversation messages to metrics for analysis
            if metrics and len(metrics) > 0 and isinstance(metrics[0], dict):
                # Convert Message objects to JSON-serializable format
                # Messages include MCP tool calls (get_knowledge, add_knowledge, etc.)
                metrics[0]["messages"] = convert_messages_to_dict(messages)

                # Add error counts from results_dict (before pass@1)
                error_counts = results_dict.get("error_counts", {})
                if error_counts:
                    # Insert error_counts at the beginning of metrics dict (before pass@1)
                    metrics[0] = {"error_counts": error_counts, **metrics[0]}
                    print(f"  Error counts: {error_counts}")

            with open(eval_file_kflow, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"\nRound {round_idx+1} Results:")
            print(f"  Pass@1: {metrics[0]['pass@1']:.2%}")

            # Write trace logs if enabled (with immediate correctness evaluation)
            if args.logging_trace:
                round_eval_file = os.path.basename(eval_file_kflow)

                # Get reference_solution_is_wrong flags from results_dict if available
                ref_flags = results_dict.get("reference_solution_is_wrong", [False] * len(results))

                # Get all rollout solutions for per-rollout evaluation
                all_rollout_solutions = results_dict.get("rollout_solutions", [])

                # Extract per-problem correctness from metrics
                # metrics[0]["detail"]["pass@1"] contains per-problem pass@1 rates
                problem_correctness = {}
                if metrics and len(metrics) > 0 and isinstance(metrics[0], dict):
                    if "detail" in metrics[0] and "pass@1" in metrics[0]["detail"]:
                        per_problem_pass = metrics[0]["detail"]["pass@1"]
                        for problem_key, pass_rate in per_problem_pass.items():
                            try:
                                # Convert string key to integer index
                                prob_idx = int(problem_key)
                                # pass_rate is typically 1.0 (pass) or 0.0 (fail)
                                correctness = pass_rate > 0
                                problem_correctness[prob_idx] = correctness
                            except (ValueError, TypeError) as e:
                                print(f"[Trace] Warning: Could not parse problem {problem_key}: {e}")
                                continue

                # Evaluate all rollouts in one batched call (only if --eval_all_rollouts is set)
                rollout_results = {}
                if args.eval_all_rollouts and all_rollout_solutions:
                    rollout_results = _evaluate_all_rollouts_batched(
                        remaining_benchmark, all_rollout_solutions, args, model
                    )

                for idx, result in enumerate(results):
                    solution = result[0] if result and len(result) > 0 else ""
                    ref_solution_wrong = ref_flags[idx] if idx < len(ref_flags) else False
                    correctness = problem_correctness.get(idx, False)
                    rollout_accs = rollout_results.get(idx)  # Already computed in parallel

                    _write_trace_log(
                        trace_dir=trace_dir,
                        problem_idx=idx,
                        round_num=round_idx + 1,
                        solution=solution,
                        reference_solution_is_wrong=ref_solution_wrong,
                        round_eval_file=round_eval_file,
                        correctness=correctness,  # Immediate evaluation, no backfill needed
                        rollout_accs=rollout_accs  # Per-rollout correctness (list of 0/1)
                    )

                print(f"[Trace] Logged {len(results)} solutions for round {round_idx+1} with immediate correctness evaluation")

            # Extract and store solutions for next round
            # In the next round, agents will see these as "previous attempts"
            for i, instance in enumerate(remaining_benchmark):
                if results[i] and results[i][0]:
                    instance.last_solution = extract_code(results[i][0], LMStyle.GenericBase)
                else:
                    instance.last_solution = ""

        # Cleanup MCP servers after all rounds complete
        if hasattr(runner, 'cleanup_servers'):
            print("\nCleaning up MCP servers...")
            runner.cleanup_servers()
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)

        with open(eval_sample_file, "w") as f:
            json.dump(save_eval_results[-10:], f, indent=4)


def _write_trace_log(trace_dir: str, problem_idx: int, round_num: int,
                     solution: str, reference_solution_is_wrong: bool,
                     round_eval_file: str, correctness=None,
                     rollout_accs=None) -> None:
    """Write a trace log entry for a single problem-round attempt.

    Args:
        trace_dir: Directory to write trace logs
        problem_idx: Problem index
        round_num: Round number
        solution: The best solution selected by Knowledge Manager
        reference_solution_is_wrong: Whether reference solution was wrong
        round_eval_file: Name of the round evaluation file
        correctness: Whether the best solution is correct
        rollout_accs: List of 0/1 values for each rollout's correctness
    """
    problem_dir = os.path.join(trace_dir, f"problem_{problem_idx}")
    os.makedirs(problem_dir, exist_ok=True)

    trace_entry = {
        "problem_idx": problem_idx,
        "round": round_num,
        "solution": solution,
        "correctness": correctness,
        "reference_solution_is_wrong": reference_solution_is_wrong,
        "timestamp": datetime.now().isoformat(),
        "round_eval_file": round_eval_file
    }

    # Add per-rollout correctness if available
    if rollout_accs is not None:
        trace_entry["rollout_accs"] = rollout_accs

    trace_file = os.path.join(problem_dir, f"round_{round_num}.json")
    with open(trace_file, "w") as f:
        json.dump(trace_entry, f, indent=2)
        

if __name__ == "__main__":
    main()
