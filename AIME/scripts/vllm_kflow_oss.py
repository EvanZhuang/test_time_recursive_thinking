import argparse
import re, json, os, datetime
from datasets import load_dataset
import vllm
from vllm import LLM, SamplingParams
import transformers

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    ReasoningEffort,
    SystemContent,
    DeveloperContent,
) 

def parse_arguments():
    """Parse command-line arguments for the AIME evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a (cutting-edge) LLM on the AIME dataset using vLLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Hugging Face model name or path (e.g. 'gpt2', 'EleutherAI/gpt-neo-1.3B', etc.)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Which dataset split to evaluate on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate for each response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0.0 = greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling (-1 disables top-k)"
    )
    parser.add_argument(
        "--demo_size",
        type=int,
        default=0,
        help="Number of demo questions to show to the model"
    )
    parser.add_argument(
        "--reflex_size",
        type=int,
        default=1,
        help="Number of reflextion steps to perform (1 means no reflexion, 2 means one reflexion step, etc.)"
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default="",
        help="Postfix to append to the output files"
    )
    args = parser.parse_args()
    return args


def get_last_integer(text):
    """Extract the last integer from text, searching backwards. Handles negative numbers."""
    i = len(text) - 1
    while i >= 0:
        if text[i].isdigit():
            # We've found a digit; now move backwards to locate the start
            end = i
            while i >= 0 and text[i].isdigit():
                i -= 1
            # Check if there's a '-' immediately before the digits
            if i >= 0 and text[i] == '-':
                return text[i:end+1]
            else:
                return text[i+1:end+1]
        i -= 1
    return None


def extract_answer_integer(generated_text: str):
    """
    Extract integer answer from LLM output using multiple heuristics:
    1. Look for \boxed{X} pattern
    2. Find last integer in the text
    Returns integer if found, else None.
    """

    patterns = [
        r"\\boxed{(-?\d+)}",
    ]
    for pattern in patterns:
        match = re.search(pattern, generated_text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    return None


def remove_first_colon(text):
    """Remove the first colon and any leading spaces after it."""
    colon_index = text.find(':')
    if colon_index != -1:
        # Remove the colon and any leading spaces after it
        return text[:colon_index] + text[colon_index + 1:].lstrip()
    return text

def main():
    """Main evaluation loop with iterative refinement."""
    args = parse_arguments()

    # Load dataset
    dataset = load_dataset("yentinglin/aime_2025")['train']

    # Initialize vLLM model
    llm = LLM(args.model_name, dtype="bfloat16", tensor_parallel_size=8, max_model_len=args.max_new_tokens, gpu_memory_utilization=0.9, async_scheduling=True, enable_prefix_caching=False, cuda_graph_sizes=[2048], compilation_config={"pass_config":{"enable_fi_allreduce_fusion":True,"enable_noop":True},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"})
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Initialize tracking dictionaries for iterative refinement
    conversations = {k: [] for k in range(len(dataset))}
    summaries = {k: [] for k in range(len(dataset))}
    knowledges = {k: [] for k in range(len(dataset))}
    confidences = {k: [] for k in range(len(dataset))}
    drop_decisions = {k: [] for k in range(len(dataset))}
    answers_tracker = {k: [] for k in range(len(dataset))}

    # Setup Harmony encoding for model interaction
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(0.9*args.max_new_tokens),
        stop_token_ids=stop_token_ids,
    )

    # Define initial prompt for first iteration
    init_prompt = \
"""\n\nGuideline: Let's solve this problem. Be thorough. 

## Output format (Use exact headers including square brackets):
[Summary]: A paragraph of detailed step-by-step summary of your solution, write thoroughly and in details, note down every steps of calculation you did, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly.
""" 

    # Define refinement prompt for subsequent iterations
    cmd_prompt = \
"""\n\nLet's solve this problem. I have some additional information that might help. Examine them carefully and see if they can help you solve the problem more accurately.

{knowledge_text}

### Reference Solution 
Take these information with a grain of salt, they might be wrong or incomplete. Try to spot the mistakes in the solution and see if there is a more accurate approach.
{reference_solution}

### Output format (Use exact headers including square brackets):
[Why the reference solution is wrong?]: If you get a different solution than the reference solution, explain here in a stand-alone manner, you must explain what is the reference solution's final answer, and why is it incorrect. (or write "N/A" if you agree with the reference solution)
[Summary]: A paragraph of detailed step-by-step summary of your solution, write thoroughly and in details, note down every steps of calculation you did, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly."""

    REASONING_EFFORT = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }

    # Main refinement loop (iterative self-improvement)
    for reflex_ctr in range(args.reflex_size):
        all_questions = []
        all_prompts = []
        gold_answers = []
        predicted_solutions = []
        correct = 0
        total = 0
        
        # Process each problem in the dataset
        for example_ctr, example in enumerate(dataset):
            question = example["problem"]
            all_questions.append(question)
            gold_answer = example["answer"]
            gold_answers.append(gold_answer)

            # Initialize conversation on first iteration
            if reflex_ctr == 0:
                conversations[example_ctr] = [
                    {"role": "system", 
                    "content": "PLACEHOLDER"
                    },
                {
                    "role": "user",
                    'content': f"{question} \nPlease reason step by step, and put your final answer within \\boxed{{}}"
                }]
            else:
                # Extract and analyze previous response for refinement
                last_conv = conversations[example_ctr][-1]
                x = last_conv['content'].split("assistantfinal")[-1].strip() if "assistantfinal" in last_conv['content'] else last_conv['content']
                summary_text = x.split("[Summary]")[-1].strip().split("[Answer]")[0].strip()
                summary_text = remove_first_colon(summary_text)
                summary_text = summary_text[0].lower() + summary_text[1:] if len(summary_text) > 1 else summary_text.lower()
                summaries[example_ctr].append(summary_text)

                # Track answer changes and identify mistakes
                this_answer = extract_answer_integer(x.split("[Answer]")[-1].strip())
                if len(answers_tracker[example_ctr]) == 1 or this_answer is None:
                    pass
                else:
                    if int(this_answer) != int(answers_tracker[example_ctr][-2]) and "[Why the reference solution is wrong?]" in x:
                        drop_decisions[example_ctr].append(answers_tracker[example_ctr][-2])
                        new_knowledge = x.split("[Why the reference solution is wrong?]")[-1].strip()
                        knowledges[example_ctr].append(new_knowledge)


                conversations[example_ctr][-1]['content'] = last_conv['content']
                prompt = conversations[example_ctr]

            # Build prompt based on iteration
            if reflex_ctr == 0:
                tmp_conv = [
                    {"role": "system", 
                    "content": "PLACEHOLDER"
                    },
                {
                    "role": "user",
                    'content': f"{question}" + init_prompt
                }]
            else:
                # Build knowledge base from previous mistakes
                if len(knowledges[example_ctr]) == 0:
                    knowledge_text = "\n\n## Empirical Mistakes List\nN/A"
                else:
                    # Deduplicate wrong answers and build unique knowledge list
                    wrong_set = set()
                    unique_knowledges = []
                    for _ in range(len(knowledges[example_ctr])-1, -1, -1):
                        if int(drop_decisions[example_ctr][_]) in wrong_set:
                            continue
                        wrong_set.add(int(drop_decisions[example_ctr][_]))
                        unique_knowledges.append(knowledges[example_ctr][_])
                    knowledge_text = "## Empirical Mistakes List\n" + "\n".join([f"**Wrong Answer {i+1}**: {unique_knowledges[i]}" for i in range(len(unique_knowledges))])
                recall_text = ""
                summary_text = ""
                # Build reference solution from previous attempt
                if len(summaries[example_ctr]) > 0:
                    summary_text = "For your reference, this is a solution to solve the problem, try to improve it:{}".format(summaries[example_ctr][-1])
                tmp_conv = [{"role": "system", "content": "PLACEHOLDER"}, 
                {"role": "user", "content": question + cmd_prompt.replace("{knowledge_text}", knowledge_text).replace("{reference_solution}", summary_text)}]

            # Convert to Harmony encoding format
            harmony_messages = []
            for msg in tmp_conv:
                if msg['role'] == 'system':
                    harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(REASONING_EFFORT["high"])))
                elif msg['role'] == 'user':
                    harmony_messages.append(Message.from_role_and_content(Role.USER, msg['content']))
                elif msg['role'] == 'assistant':
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, msg['content']))
            
            convo = Conversation.from_messages(harmony_messages)
            prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            prefill_ids = vllm.inputs.TokensPrompt(prompt_token_ids=prefill_ids)
            all_prompts.append(prefill_ids)
        
        # Batch generate responses for all problems
        outputs = llm.generate(all_prompts, sampling_params=sampling_params)

        # Process outputs and evaluate
        predictions = []
        for example_ctr, (out, gold) in enumerate(zip(outputs, gold_answers)):
            output_tokens = out.outputs[0].token_ids
            generation = out.outputs[0].text
            if conversations[example_ctr][-1]['role'] == "user":
                conversations[example_ctr].append({"role": "assistant", "content": generation})
            else:
                conversations[example_ctr][-1]['content'] = generation
            predicted_solutions.append(conversations[example_ctr][-1]['content'])
            
            # Extract and evaluate predicted answer
            pred_answer = extract_answer_integer(generation)
            if pred_answer is not None:
                answers_tracker[example_ctr].append(pred_answer)
            predictions.append(pred_answer)
            if pred_answer is not None and int(pred_answer) == int(gold):
                correct += 1
            total += 1

        # Print iteration results
        accuracy = correct / total if total > 0 else 0
        print(f"Evaluation on split='{args.split}' with model='{args.model_name}':")
        print(f"  Total questions: {total}")
        print(f"  Correct answers: {correct}")
        print(f"  Accuracy: {accuracy:.2%}")

        # Save results to file
        out_json = [{"accuracy": accuracy, "total": total, "correct": correct}]
        out_json = out_json + [{"question": tokenizer.decode(q["prompt_token_ids"]), "gold": g, "predicted": p, "pred_solution": s} for q, g, p, s in zip(all_prompts, gold_answers, predictions, predicted_solutions)]
        model_base_name = args.model_name.split("/")[-1]

        os.makedirs(f"./results/vllm_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}", exist_ok=True)
        with open(f"./results/vllm_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}/predictions_{reflex_ctr}.json", "w") as f:
            json.dump(out_json, f, indent=2)

if __name__ == "__main__":
    main()