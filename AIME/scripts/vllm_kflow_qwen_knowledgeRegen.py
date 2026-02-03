import argparse
import re
import json
import os
import random
from datasets import load_dataset
import datasets
from vllm import LLM, SamplingParams
import transformers
import nltk
import math
import random
from nltk.tokenize import sent_tokenize

# Download the necessary tokenizer model
nltk.download('punkt_tab')

def parse_arguments():
    """Parse command-line arguments for the AIME evaluation script with knowledge regeneration."""
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
    1. Look for \boxed{X} pattern (takes last match)
    2. Find last integer in the text
    Returns integer if found, else None.
    """
    # Look for \boxed{X} pattern
    patterns = [
        r"\\boxed{(-?\d+)}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, flags=re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass

    # Fallback: parse the last integer found anywhere in the text
    last_int_str = get_last_integer(generated_text)
    if last_int_str:
        try:
            return int(last_int_str)
        except ValueError:
            pass

    return None


def fill_to_length(sent_lst, max_length, tokenizer):
    """
    Select the last K sentences that fit within max_length tokens.
    Works backwards from the end to preserve the most recent context.
    """
    if not sent_lst:
        return []
    
    selected_sentences = []
    current_length = 0
    
    for sentence in reversed(sent_lst):
        sentence_tokens = len(tokenizer.encode(sentence))
        
        if current_length + sentence_tokens <= max_length:
            selected_sentences.insert(0, sentence)
            current_length += sentence_tokens
        else:
            break
    
    return selected_sentences


def remove_first_colon(text):
    """Remove the first colon and any leading spaces after it from the text."""
    colon_index = text.find(':')
    if colon_index != -1:
        return text[:colon_index] + text[colon_index + 1:].lstrip()
    return text


def main():
    """Main evaluation loop with iterative knowledge regeneration based on mistakes."""
    args = parse_arguments()

    # Load dataset
    aime25 = load_dataset("yentinglin/aime_2025")['train']
    aime25 = aime25.remove_columns([col for col in aime25.column_names if col not in ['problem', 'answer']])
    dataset = aime25

    # Initialize vLLM model
    llm = LLM(
        args.model_name, 
        dtype="bfloat16", 
        tensor_parallel_size=8, 
        max_model_len=args.max_new_tokens, 
        gpu_memory_utilization=0.9, 
        trust_remote_code=True, 
        async_scheduling=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Initialize tracking structures for knowledge regeneration
    conversations = {k: [] for k in range(len(dataset))}
    summaries = {k: [] for k in range(len(dataset))}
    knowledges = {k: [] for k in range(len(dataset))}
    confidences = {k: [] for k in range(len(dataset))}
    answers_tracker = {k: [] for k in range(len(dataset))}
    drop_decisions = {k: [] for k in range(len(dataset))}

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(0.9 * args.max_new_tokens),
    )

    sampling_params_diffusion = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(args.max_new_tokens / 4),
        logprobs=20,
    )

    # Define prompts for initial and refinement iterations
    wait_text = ""
    init_prompt = """\n\nGuideline: Let's solve this problem. Be thorough. 

## Output format (Use exact headers including square brackets):
[Things to Avoid]: A standalone list of max 10 answers to avoid, note down what was not working for you, which answer is definitely wrong and why. Write it in a standalone manner.
[Summary]: A paragraph of detailed step-by-step summary of your solution, note down every reasoning step and calculation you did, where you might have gone wrong, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly.
"""

    cmd_prompt = """\n\nLet's solve this problem. Be thorough. Let's solve from scratch first. Try to avoid any heuristics, use solid logic and math. Then check against the Wrong Answer List and the reference solution, see which one is correct.

## Reference Solution
{reference_solution}

{knowledge_text}

### Output format (Use exact headers including square brackets):
[Things to Avoid]: A standalone list of max 10 answers to avoid, note down what was not working for you, which answer is definitely wrong and why. Write it in a standalone manner.
[Summary]: A paragraph of detailed step-by-step summary of your solution, write thoroughly and in details, note down every steps of reasoning you did, and what was the final answer you got. Write it in a standalone manner.
[Answer]: Therefore, final answer is \\boxed{<integer>}."""

    # Ensure chat template is available
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<|begin_of_thought|>' }}\n{%- endif %}\n"

    # Main refinement loop with knowledge regeneration
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
            last_action = ""

            # Initialize conversation on first iteration
            if reflex_ctr == 0:
                conversations[example_ctr].append({
                    "role": "user",
                    'content': f"{question}" + init_prompt
                })
                prompt = conversations[example_ctr]
            else:
                # Extend Generation
                last_conv = conversations[example_ctr][-1]
                x = last_conv['content'].split("</think>")[-1]
                summary_text = x.split("[Summary]")[-1].strip().split("[Answer]")[0].strip()
                summary_text = remove_first_colon(summary_text)
                summary_text = summary_text[0].lower() + summary_text[1:] if len(summary_text) > 1 else summary_text.lower()
                summaries[example_ctr].append(summary_text)
                new_knowledge = x.split("[Things to Avoid]")[-1].split("[Summary]")[0].strip()
                new_knowledge = remove_first_colon(new_knowledge)
                new_knowledge = new_knowledge[0].lower() + new_knowledge[1:]
                knowledges[example_ctr].append(new_knowledge)

                conversations[example_ctr][-1]['content'] = last_conv['content']
                prompt = conversations[example_ctr]

            if reflex_ctr == 0:
                # initial generation
                tmp_conv = [{"role": "user", "content": question + init_prompt}]
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True)
            else:
                knowledge_text = "## Wrong Answer List\n" + knowledges[example_ctr][-1]
                summary_text = "Note: No reference solution available."
                # if len(set(answers_tracker[example_ctr])) >= 2:
                #     wrong_answers = [str(x) for x in set(answers_tracker[example_ctr]) if x is not None and int(x) != int(answers_tracker[example_ctr][-1])]
                #     recall_text = " For your reference, we think these are wrong answers: " + ", ".join([str(x) for x in wrong_answers]) + ". Do not repeat previous mistakes."
                # if len(drop_decisions[example_ctr]) > 0:
                #     last_knowledge_text = "\n\n **Avoid adding this back** Last time you decided to drop: <" + drop_decisions[example_ctr][-1] + "> because it was incorrect or unhelpful, make sure to avoid it this time."
                #     knowledge_text = knowledge_text + last_knowledge_text
                if len(summaries[example_ctr]) > 0:
                    summary_text = "Reference Solution (check after you first solve independently):\n" + summaries[example_ctr][-1]
                tmp_conv = [{"role": "user", "content": question + cmd_prompt.replace("{knowledge_text}", knowledge_text).replace("{reference_solution}", summary_text)}]
                eps_text = "This is a complex problem. I sense a mistake in the reference solution, I will solve it from scratch first without referencing it. Then I will check against the Wrong Answer List and the Reference Solution to write an improved solution."
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True) + eps_text

            all_prompts.append(prompt)
        
        outputs = llm.generate(all_prompts, sampling_params)

        predictions = []
        # Each element in 'outputs' corresponds to one input prompt
        for example_ctr, (out, gold) in enumerate(zip(outputs, gold_answers)):
            # out is a RequestOutput containing .outputs (list of model generations)
            generation = out.outputs[0].text  # we only requested n=1
            if conversations[example_ctr][-1]['role'] == "user":
                conversations[example_ctr].append({"role": "assistant", "content": generation})
            else:
                # s1 generation
                conversations[example_ctr][-1]['content'] = generation
            predicted_solutions.append(generation)
            # Extract integer from the generated text
            pred_answer = extract_answer_integer(generation)
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
        output_confidences = [max(confidences[example_ctr]) if confidences[example_ctr] else -1 for example_ctr in range(len(confidences))]
        out_json = [{"accuracy": accuracy, "total": total, "correct": correct, "average_prompt_length": sum([len(tokenizer.encode(p)) for p in all_prompts]) / len(all_prompts)}]
        out_json = out_json + [
            {"question": q, "gold": g, "predicted": p, "conf": c, "pred_solution": s} 
            for q, g, p, c, s in zip(all_prompts, gold_answers, predictions, output_confidences, predicted_solutions)
        ]
        model_base_name = args.model_name.split("/")[-1]

        os.makedirs(f"./results/vllm_regen_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}", exist_ok=True)
        with open(f"./results/vllm_regen_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}/predictions_{reflex_ctr}.json", "w") as f:
            json.dump(out_json, f, indent=2)

if __name__ == "__main__":
    main()