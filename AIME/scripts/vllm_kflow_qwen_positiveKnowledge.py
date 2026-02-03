import argparse
import re
import json
import os
import random
import math
from datasets import load_dataset
import datasets
from vllm import LLM, SamplingParams
import transformers
import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary tokenizer model
nltk.download('punkt_tab')

def parse_arguments():
    """Parse command-line arguments for the AIME evaluation script with knowledge management."""
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
    1. Look for \boxed{X} pattern (returns last match if multiple found)
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
    """Main evaluation loop with iterative knowledge management and refinement."""
    args = parse_arguments()

    # Load dataset (keep only problem and answer fields)
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
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Initialize tracking structures for knowledge management
    conversations = {k: [] for k in range(len(dataset))}
    summaries = {k: [] for k in range(len(dataset))}
    knowledges = {k: [] for k in range(len(dataset))}
    confidences = {k: [] for k in range(len(dataset))}
    drop_decisions = {k: [] for k in range(len(dataset))}
    answers_tracker = {k: [] for k in range(len(dataset))}

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
    init_prompt = """\n\nGuideline: Let's solve this problem. Be thorough. 

For Experimental Knowledge, provide: (1) TRIGGER, When to use this - be concrete, (2) STEPS, Detailed procedure, (3) WORKED EXAMPLE, Concrete instance, (4) PITFALL, common mistakes to avoid.

Example of Good Experimental Knowledge:
Trigger: Look for [base]^[large_exponent] mod [prime]
When you see 'find a^b mod p' where p is prime:
→ By Fermat's Little Theorem: 7^10 ≡ 1 (mod 11)
→ So 7^100 = (7^10)^10 ≡ 1^10 ≡ 1 (mod 11)
PITFALL: "Applying when gcd(a,p) ≠ 1"
  Wrong: 22^100 mod 11 → "reduce 100 mod 10 = 0" → 1
  Right: 22 = 2×11 ≡ 0 (mod 11) → 22^100 ≡ 0 (mod 11)
FLT requires gcd(a,p) = 1!

## Output format (Use exact headers including square brackets):
[Novel Knowledge to Add to the List]: one key knowledge that other should know for solving the problem. Follow the format above.
[Summary]: A paragraph of detailed step-by-step summary that other should know for correctly solving the problem, write thoroughly and in details, note down every steps of reasoning and calculation you did, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.""" 

    cmd_prompt = """\n\nGuideline: Let's solve this problem. Look at the experimental knowledge list below, and see if any of them can help you solve the problem better or if you can disprove any of them.

Attempt to solve the problem without using the experimental knowledge first. Then solve the problem with the experimental knowledge. 
Cross reference to see if you get the same answer, and explain how KL helps you solve the problem. If you find some incorrect or unhelpful knowledge, explain why it is wrong or unhelpful.
    
Analyze the provided experimental knowledge (EK) to identify gaps and limitations for each of them, and make a judgement on whether some of them are redundant or incorrect, then generate genuinely novel insights that are more accurate or create a novel pathway to solve the problem, actionable for problem-solving. If you are disproving a KL, make sure to provide a concrete worked example that shows why it is wrong.

You can choose to add one new knowledge, drop one unhelpful knowledge, or do both. You can also choose to do nothing if you think the existing knowledge is sufficient and correct.

For Experimental Knowledge, provide: (1) TRIGGER, When to use this - be concrete, (2) STEPS, Detailed procedure, (3) WORKED EXAMPLE, Concrete instance, (4) PITFALL, common mistakes to avoid.

Example of Good Experimental Knowledge:
Trigger: Look for [base]^[large_exponent] mod [prime]
When you see 'find a^b mod p' where p is prime:
→ By Fermat's Little Theorem: 7^10 ≡ 1 (mod 11)
→ So 7^100 = (7^10)^10 ≡ 1^10 ≡ 1 (mod 11)
PITFALL: "Applying when gcd(a,p) ≠ 1"
  Wrong: 22^100 mod 11 → "reduce 100 mod 10 = 0" → 1
  Right: 22 = 2×11 ≡ 0 (mod 11) → 22^100 ≡ 0 (mod 11)
FLT requires gcd(a,p) = 1!

{knowledge_text}

## Reference Solution (check after you first solve independently):
{reference_solution}

## Output format (Use exact headers including square brackets):
[Solution without EK]: <your solution without using EK, if you can solve it>
[Adding?]: Yes / No
[Novel Knowledge to Add to the List]: one novel knowledge (not similar to any item in your existing knowledge list) for solving the problem, follow the format above. Write "N/A" if no significantly new knowledge is found.
[Reason to make this addition]: If updating Knowledge List, convince the judge why the new knowledge is considered significantly novel and useful? If No, explain why is it a minor update and not worth adding.
[Dropping?]: Yes / No
[Least Useful to Drop]: If you want to drop unhelpful knowledge, write down the ID of the incorrect/redundant item to Drop (format = "Dropping Experimental Knowledge 1/2/3/..."). If not, write 'N/A'.
[Why that knowledge should be dropped?]: If dropping from Knowledge List, explain why. If not, write 'N/A'.
[Summary]: A paragraph of detailed step-by-step summary that other should know for correctly solving the problem, write thoroughly and in details, note down every steps of reasoning and calculation you did, and what was the final answer you got.
[Answer]: Write your final answer in \\boxed{<integer>}."""

    # Set default chat template if not provided
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<|begin_of_thought|>' }}\n{%- endif %}\n"

    # Main refinement loop with knowledge management
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
                # Process previous generation to extract and manage knowledge
                last_conv = conversations[example_ctr][-1]
                x = last_conv['content'].split("</think>")[-1]
                
                if "[Novel Knowledge to Add to the List]" in x and "[Answer]" in x:
                    # Extract and store solution summary
                    summary_text = x.split("[Summary]")[-1].strip().split("[Answer]")[0].strip()
                    summary_text = remove_first_colon(summary_text)
                    summary_text = summary_text[0].lower() + summary_text[1:] if len(summary_text) > 1 else summary_text.lower()
                    summaries[example_ctr].append(summary_text)
                    
                    # Check if model decided to add new knowledge
                    if "[Adding?]" in x and "No" in x.split("[Adding?]")[-1].split("[")[0]:
                        pass  # No knowledge update
                    else:
                        # Extract and add new knowledge
                        if "[Reason to make this addition]" in x:
                            new_knowledge = x.split("[Novel Knowledge to Add to the List]")[-1].strip().split("[Reason to make this addition]")[0].strip()
                        elif "[Summary]" in x:
                            new_knowledge = x.split("[Novel Knowledge to Add to the List]")[-1].strip().split("[Summary]")[0].strip()
                        else:
                            new_knowledge = x.split("[Novel Knowledge to Add to the List]")[-1].strip().split("[")[0].strip()
                        
                        last_action += "Last time you added new knowledge: <" + new_knowledge + ">."
                        
                        # Validate knowledge length and add to list
                        if len(tokenizer.encode(new_knowledge)) <= 1024:
                            if new_knowledge and new_knowledge[0] == ':':
                                new_knowledge = new_knowledge[1:].strip()
                            knowledges[example_ctr].append(new_knowledge)
                    
                    # Check if model decided to drop knowledge
                    drop_id = 0
                    if "[Least Useful to Drop]" in x:
                        drop_text = x.split("[Least Useful to Drop]")[-1].strip().split("[Why that knowledge should be dropped?]")[0].strip()
                        if get_last_integer(drop_text) is not None:
                            drop_id = int(get_last_integer(drop_text))
                        
                        # Validate drop ID and remove knowledge if valid
                        if drop_id > 0 and drop_id <= len(knowledges[example_ctr]):
                            drop_reason = x.split("[Why that knowledge should be dropped?]")[-1].strip().split("[")[0].strip()
                            drop_reason = drop_reason[1:] if drop_reason and drop_reason[0] == ':' else drop_reason
                            drop_reason = drop_reason.strip()
                            
                            dropped_knowledge = knowledges[example_ctr].pop(drop_id - 1)
                            last_action += f" Last time you dropped: <{dropped_knowledge}>."

                        # Limit knowledge list size (remove oldest if exceeds limit)
                        if len(knowledges[example_ctr]) > 10:
                            knowledges[example_ctr].pop(0)
                drop_decisions[example_ctr].append(last_action)
                conversations[example_ctr][-1]['content'] = last_conv['content']
                prompt = conversations[example_ctr]

            # Build prompt based on iteration
            if reflex_ctr == 0 or len(knowledges[example_ctr]) == 0:
                tmp_conv = [{"role": "user", "content": question + init_prompt}]
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True)
            else:
                # Build knowledge list for refinement prompt
                knowledge_text = "## Experimental Knowledge List\n" + "\n".join(
                    [f"**EK {i+1}**: {knowledges[example_ctr][i]}" for i in range(len(knowledges[example_ctr]))]
                )
                
                summary_text = summaries[example_ctr][-1] if len(summaries[example_ctr]) > 0 else ""
                tmp_conv = [{"role": "user", "content": question + cmd_prompt.replace("{knowledge_text}", knowledge_text).replace("{reference_solution}", summary_text)}]
                    
                prompt = tokenizer.apply_chat_template(tmp_conv, tokenize=False, add_generation_prompt=True)

            all_prompts.append(prompt)
        
        outputs = llm.generate(all_prompts, sampling_params)
        predictions = []
        # Each element in 'outputs' corresponds to one input prompt
        for example_ctr, (out, gold) in enumerate(zip(outputs, gold_answers)):
            # out is a RequestOutput containing .outputs (list of model generations)
            generation = out.outputs[0].text  # we only requested n=1
            
            # Update conversation history
            if conversations[example_ctr][-1]['role'] == "user":
                conversations[example_ctr].append({"role": "assistant", "content": generation})
            else:
                conversations[example_ctr][-1]['content'] = generation
            predicted_solutions.append(generation)
            
            # Extract and evaluate predicted answer
            pred_answer = extract_answer_integer(generation)
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
        avg_prompt_length = sum([len(tokenizer.encode(p)) for p in all_prompts]) / len(all_prompts)
        out_json = [{"accuracy": accuracy, "total": total, "correct": correct, "average_prompt_length": avg_prompt_length}]
        out_json = out_json + [
            {"question": q, "gold": g, "predicted": p, "conf": c, "pred_solution": s} 
            for q, g, p, c, s in zip(all_prompts, gold_answers, predictions, output_confidences, predicted_solutions)
        ]
        model_base_name = args.model_name.split("/")[-1]

        os.makedirs(f"./results/vllm_posK_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}", exist_ok=True)
        with open(f"./results/vllm_posK_{args.max_new_tokens}_{args.output_postfix}/{model_base_name}/predictions_{reflex_ctr}.json", "w") as f:
            json.dump(out_json, f, indent=2)

if __name__ == "__main__":
    main()