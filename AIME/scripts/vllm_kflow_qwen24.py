import argparse
import re
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
import transformers


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

    # Fallback: find last integer anywhere in the text
    last_int_str = get_last_integer(generated_text)
    if last_int_str:
        try:
            return int(last_int_str)
        except ValueError:
            pass

    # If nothing found, return None
    return None


def remove_first_colon(text):
    """Remove the first colon and any leading spaces after it."""
    colon_index = text.find(':')
    if colon_index != -1:
        # Remove the colon and any leading spaces after it
        return text[:colon_index] + text[colon_index + 1:].lstrip()
    return text


def main():
    """Main evaluation loop with iterative refinement for AIME 2024 dataset."""
    args = parse_arguments()

    # Load AIME 2024 dataset
    dataset = load_dataset("HuggingFaceH4/aime_2024")['train']

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

    # Initialize tracking dictionaries for iterative refinement
    conversations = {k: [] for k in range(len(dataset))}
    summaries = {k: [] for k in range(len(dataset))}
    knowledges = {k: [] for k in range(len(dataset))}
    answers_tracker = {k: [] for k in range(len(dataset))}
    drop_decisions = {k: [] for k in range(len(dataset))}

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(0.9*args.max_new_tokens),
    )

    # Define initial prompt for first iteration
    init_prompt = \
"""\n\nGuideline: Let's solve this problem. Be thorough.

## Output format (Use exact headers including square brackets):
[Summary]: A paragraph of detailed step-by-step summary of your solution, note down every reasoning step and calculation you did, where you might have gone wrong, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly.
""" 

    # Define refinement prompt for subsequent iterations
    cmd_prompt = \
"""\n\nLet's solve this problem. Analyze the reference solution carefully, find mistakes or reasoning flaws (for example, wrong assumptions, miscalculations, missing edge cases). Make sure to check all the assumptions and reasoning, give me an extremely accurate solution.

## Reference Solution
{reference_solution}

{knowledge_text}

### Output format (Use exact headers including square brackets):
[Error Report]: If you get a different solution than your recalled solutions, explain why the old answer is wrong in details here, start with "The Answer cannot be <integer>, because..." end with "The correct approach is ..." without listing out the answer you got, explain in details in a standalone manner (or write "N/A" if you agree with the recalled solution)
[Summary]: A paragraph of detailed step-by-step summary of your solution, write thoroughly and in details, note down every steps of reasoning you did, and what was the final answer you got. Write it in a standalone manner.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly."""

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<|begin_of_thought|>' }}\n{%- endif %}\n"

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
                conversations[example_ctr].append({
                    "role": "user",
                    'content': f"{question}" + init_prompt
                })
                prompt = conversations[example_ctr]
            else:
                # Extract and analyze previous response for refinement
                last_conv = conversations[example_ctr][-1]
                x = last_conv['content'].split("</think>")[-1]
                summary_text = x.split("[Summary]")[-1].strip().split("[Answer]")[0].strip()
                summary_text = remove_first_colon(summary_text)
                summary_text = summary_text[0].lower() + summary_text[1:] if len(summary_text) > 1 else summary_text.lower()
                summaries[example_ctr].append(summary_text)

                # Track answer changes and identify mistakes
                this_answer = extract_answer_integer(x.split("[Answer]")[-1].strip())
                if len(answers_tracker[example_ctr]) <= 1:
                    pass
                else:
                    if int(this_answer) != int(answers_tracker[example_ctr][-2]) and "[Error Report]" in x:
                        drop_decisions[example_ctr].append(answers_tracker[example_ctr][-2])
                        print("Problem {}, {} | Changed answer from {} to {}".format(example_ctr, gold_answer, answers_tracker[example_ctr][-2], this_answer))
                        new_knowledge = x.split("[Error Report]")[-1].split("[Summary]")[0].strip()
                        new_knowledge = remove_first_colon(new_knowledge)
                        new_knowledge = new_knowledge[0].lower() + new_knowledge[1:]
                        knowledges[example_ctr].append(new_knowledge)
                    elif "[Error Report]" not in x:
                        print(f"Problem {example_ctr}, {gold_answer} | Not working)")

                conversations[example_ctr][-1]['content'] = last_conv['content']
                prompt = conversations[example_ctr]

            # Build prompt based on iteration
            if reflex_ctr == 0:
                # Initial generation - simple problem statement
                tmp_conv = [{"role": "user", "content": question + init_prompt}]
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True)
            else:
                # Build knowledge base from previous mistakes
                if len(knowledges[example_ctr]) == 0:
                    knowledge_text = "## Wrong Answer List\nN/A"
                else:
                    # Deduplicate wrong answers (keep up to 2 occurrences of each wrong answer)
                    wrong_set = {}
                    unique_knowledges = []
                    for _ in range(len(knowledges[example_ctr])-1, -1, -1):
                        if wrong_set.get(int(drop_decisions[example_ctr][_]), 0) >= 2:
                            continue
                        wrong_set[int(drop_decisions[example_ctr][_])] = wrong_set.get(int(drop_decisions[example_ctr][_]), 0) + 1
                        unique_knowledges.append(knowledges[example_ctr][_])
                    knowledge_text = "## Wrong Answer List\n" + "\n".join([f"**Wrong Answer {i+1}**: {unique_knowledges[i]}" for i in range(len(unique_knowledges))])
                
                # Build reference solution from previous attempt
                summary_text = "Note: No reference solution available."
                if len(summaries[example_ctr]) > 0:
                    summary_text = "Reference Solution (check after you first solve independently):\n" + summaries[example_ctr][-1]
                
                # Create refinement prompt with knowledge and reference solution
                tmp_conv = [{"role": "user", "content": question + cmd_prompt.replace("{knowledge_text}", knowledge_text).replace("{reference_solution}", summary_text)}]
                eps_text = "This is a complex problem. Oh the reference solution has one reasoning flaw! I will solve it from scratch first and challenge the reference solution's assumptions. Then I will check against the Wrong Answer List and the Reference Solution to write an improved and more accurate solution."
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True) + eps_text

            all_prompts.append(prompt)
        
        # Batch generate responses for all problems
        outputs = llm.generate(all_prompts, sampling_params)

        # Process outputs and evaluate
        predictions = []
        for example_ctr, (out, gold) in enumerate(zip(outputs, gold_answers)):
            generation = out.outputs[0].text
            if conversations[example_ctr][-1]['role'] == "user":
                conversations[example_ctr].append({"role": "assistant", "content": generation})
            else:
                conversations[example_ctr][-1]['content'] = generation
            predicted_solutions.append(generation)
            
            # Extract and evaluate predicted answer
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
        out_json = [{"accuracy": accuracy, "total": total, "correct": correct, "average_prompt_length": sum([len(tokenizer.encode(p)) for p in all_prompts])/len(all_prompts)}]
        out_json = out_json + [{"question": q,"gold": g, "predicted": p, "pred_solution": s} for q, g, p, s in zip(all_prompts, gold_answers, predictions, predicted_solutions)]
        model_base_name = args.model_name.split("/")[-1]

        os.makedirs(f"./results/vllm24_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}", exist_ok=True)
        with open(f"./results/vllm24_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}/predictions_{reflex_ctr}.json", "w") as f:
            json.dump(out_json, f, indent=2)


if __name__ == "__main__":
    main()
