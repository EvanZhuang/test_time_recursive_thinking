import argparse
import re
import json
import os
from datasets import load_dataset
import vllm
from vllm import LLM, SamplingParams
import transformers
import nltk
from nltk.tokenize import sent_tokenize

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

# Download the necessary tokenizer model
nltk.download('punkt_tab')

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
        match = re.search(pattern, generated_text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
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
    
    # Start from the end and work backwards
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

def main():
    """Main evaluation loop with iterative Markovian thinking refinement using Harmony encoding."""
    args = parse_arguments()

    # Load dataset
    dataset = load_dataset("yentinglin/aime_2025")['train']

    # Initialize vLLM model with optimized settings
    llm = LLM(
        args.model_name, 
        dtype="bfloat16", 
        tensor_parallel_size=8, 
        max_model_len=args.max_new_tokens, 
        gpu_memory_utilization=0.9, 
        async_scheduling=True, 
        enable_prefix_caching=False, 
        cuda_graph_sizes=[2048], 
        compilation_config={
            "pass_config": {"enable_fi_allreduce_fusion": True, "enable_noop": True},
            "custom_ops": ["+rms_norm"],
            "cudagraph_mode": "FULL_AND_PIECEWISE"
        }
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Initialize conversation tracking for iterative refinement
    conversations = {k: [] for k in range(len(dataset))}

    # Setup Harmony encoding for model interaction
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(0.9 * args.max_new_tokens),
        stop_token_ids=stop_token_ids,
    )

    sampling_params_diffusion = SamplingParams(
        temperature=args.temperature,
        max_tokens=int(args.max_new_tokens / 2),
    )

    # Text prompt for continued thinking in refinement iterations
    wait_text = "Wait, just keep thinking."
    
    # Set default chat template if not provided
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<|begin_of_thought|>' }}\n{%- endif %}\n"

    # Reasoning effort configuration
    REASONING_EFFORT = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }

    # Main refinement loop (iterative self-improvement via Markovian thinking)
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
                    {"role": "system", "content": "PLACEHOLDER"},
                    {
                        "role": "user",
                        'content': f"{question}"
                    }
                ]
                tmp_conv = [
                    {"role": "system", "content": "PLACEHOLDER"},
                    {"role": "user", 'content': f"{question}"}
                ]
            else:
                # Trim conversation to fit within token budget (keep most recent context)
                assert conversations[example_ctr][-1]['role'] == "assistant", "Last message should be from assistant"
                conversations[example_ctr][-1]['content'] = " ".join(
                    fill_to_length(
                        sent_tokenize(conversations[example_ctr][-1]['content']), 
                        int(args.max_new_tokens * 0.45), 
                        tokenizer
                    )
                )
                conversations[example_ctr][-1]['content'] += f"\n{wait_text}"
                tmp_conv = conversations[example_ctr]

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
            
            # Remove trailing stop token if present in refinement iterations
            if reflex_ctr > 0:
                if prefill_ids[-1] == 200007:
                    prefill_ids = prefill_ids[:-1]
            
            # Convert to vLLM tokens prompt format
            prefill_ids = vllm.inputs.TokensPrompt(prompt_token_ids=prefill_ids)
            all_prompts.append(prefill_ids)
        
        # Batch generate responses for all problems

        if reflex_ctr == 0:
            outputs = llm.generate(all_prompts, sampling_params)
        else:
            outputs = llm.generate(all_prompts, sampling_params_diffusion)

        # Process outputs and evaluate
        predictions = []
        for example_ctr, (out, gold) in enumerate(zip(outputs, gold_answers)):
            output_tokens = out.outputs[0].token_ids
            generation = out.outputs[0].text
            
            # Update conversation history
            if conversations[example_ctr][-1]['role'] == "user" or conversations[example_ctr][-1]['role'] == "system":
                conversations[example_ctr].append({"role": "assistant", "content": generation})
            else:
                conversations[example_ctr][-1]['content'] += generation.split("assistantfinal")[0]
            predicted_solutions.append(conversations[example_ctr][-1]['content'])
            
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
        out_json = [{"accuracy": accuracy, "total": total, "correct": correct}]
        out_json = out_json + [
            {"question": tokenizer.decode(q["prompt_token_ids"]), "gold": g, "predicted": p, "pred_solution": s} 
            for q, g, p, s in zip(all_prompts, gold_answers, predictions, predicted_solutions)
        ]
        model_base_name = args.model_name.split("/")[-1]

        os.makedirs(f"./results/buffer_halfoai_{args.max_new_tokens}/{model_base_name}", exist_ok=True)
        with open(f"./results/buffer_halfoai_{args.max_new_tokens}/{model_base_name}/predictions_{reflex_ctr}.json", "w") as f:
            json.dump(out_json, f, indent=2)

if __name__ == "__main__":
    main()