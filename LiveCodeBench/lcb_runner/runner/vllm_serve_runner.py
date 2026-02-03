import os, json
from time import sleep

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


def _safe_json_loads(maybe_json):
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {"_raw": maybe_json}
    return {}

class VLLMServeRunner(BaseRunner):
    client = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key=os.getenv("OPENAI_KEY", "testkey"),
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.GptOssVLLM:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "max_output_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "reasoning": {"effort": reasoning_effort},
                "timeout": args.openai_timeout,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }
        
        self.kflow_mem = []
    
    def _add_kflow_memory(self, id, knowledge: str):
        if id < len(self.kflow_mem):
            self.kflow_mem[id].append(knowledge)
        else:
            print(f"Warning: Invalid memory id {id}, max id is {len(self.kflow_mem)-1}")
    
    def _get_kflow_memory(self, id) -> str:
        if id >= len(self.kflow_mem) or len(self.kflow_mem[id]) == 0:
            return "## Knowledge Base is empty."
        return "## Knowledge Base:\n" + "\n".join(["Knowledge ID {}: {}".format(i, k) for i, k in enumerate(self.kflow_mem[id])])

    def _remove_kflow_memory(self, id, knowledge_id: int):
        if id < len(self.kflow_mem) and 0 <= knowledge_id < len(self.kflow_mem[id]):
            del self.kflow_mem[id][knowledge_id]
        else:
            print(f"Warning: Invalid knowledge_id {knowledge_id} for memory {id}")

    def _extract_tool_calls(self, response):
        """Return a list of {id, name, arguments} from Responses API output."""
        calls = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "content", None) is not None:
                item = item.content[0]
            if getattr(item, "type", None) == "function_call":
                # Responses API typically gives you:
                # item.id (tool_call_id), item.name, item.parameters (JSON string or dict)
                tool_call_id = getattr(item, "call_id", None)
                name = getattr(item, "name", None)
                args = getattr(item, "arguments", {})  # may be str or dict
                if isinstance(args, str):
                    try:
                        args = _safe_json_loads(args)
                    except Exception:
                        # leave as string if model emitted non-JSON (rare)
                        pass

                calls.append({"call_id": tool_call_id, "name": name, "arguments": args})
        return calls

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return []

        try:
            response = VLLMServeRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
        
    def run_batch(self, prompts: list[list[dict[str, str]]]) -> list[list[str]]:
        import asyncio
        import json
        from tqdm.asyncio import tqdm_asyncio
        if len(self.kflow_mem)  == 0 or len(self.kflow_mem) < len(prompts):
            self.kflow_mem = [[] for _ in range(len(prompts))]

        tool_functions = {"add_knowledge": self._add_kflow_memory, "remove_knowledge": self._remove_kflow_memory}
        tools = [
            {
                "type": "function",
                "name": "add_knowledge",
                "description": "Add knowledge to the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge": {"type": "string", "description": "The knowledge to add"}
                    },
                    "required": ["knowledge"]
                }
            },
            {
                "type": "function",
                "name": "remove_knowledge",
                "description": "Remove knowledge from the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge_id": {"type": "integer", "description": "The ID of the knowledge to remove"}
                    },
                    "required": ["knowledge_id"]
                }
            }
        ]
        
        async def _run_batch_async():
            # Create all API calls concurrently
            prompts_mem = []
            for id, prompt in enumerate(prompts):
                # Create a copy to avoid modifying the original
                prompt_copy = [msg.copy() for msg in prompt]
                prompt_copy[-1]['content'] += "\n\n" + self._get_kflow_memory(id)
                prompts_mem.append(prompt_copy)

            # Create tasks in order - asyncio.gather preserves order
            # so responses[i] will correspond to prompts_mem[i] and original prompts[i]
            tasks = [
                VLLMServeRunner.client.responses.create(
                    input=prompt[-1]['content'],
                    tools=tools,
                    tool_choice="auto",
                    instructions="Use the tools as needed to update the knowledge base. Rethink and improve.",
                    store=True,
                    **self.client_kwargs,
                )
                for prompt in prompts_mem
            ]
            # Wait for all completions with progress bar
            # gather() returns results in the same order as tasks
            responses = await tqdm_asyncio.gather(*tasks, desc="Processing batch")

            # Tool call handling and follow-up requests
            follow_up_tasks = []
            for idx, response in enumerate(responses):
                tool_calls = self._extract_tool_calls(response)
                
                if len(tool_calls) > 0:
                    # Gather conversation history for this tool call
                    conversation = []
                    
                    # Add the original prompt messages
                    for msg in prompts_mem[idx]:
                        conversation.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    # Add the assistant's response with tool calls
                    # Parse response.output to collect reasoning text and function calls
                    assistant_content = []
                    
                    for output in getattr(response, "output", []) or []:
                        # Handle content wrapper
                        if hasattr(output, "content") and output.content:
                            for c in output.content:
                                # Collect reasoning/output text
                                if getattr(c, "type", None) == "output_text" or getattr(c, "type", None) == "reasoning":
                                    text = getattr(c, "text", "")
                                    if text:
                                        assistant_content.append(text)
                                # Note: function_call items are already extracted by _extract_tool_calls
                        # Handle direct output items (if not wrapped in content)
                        elif getattr(output, "type", None) == "output_text" or getattr(output, "type", None) == "reasoning":
                            text = getattr(output, "text", "")
                            if text:
                                assistant_content.append(text)
                    
                    conversation.append({
                        "role": "assistant",
                        "content": "\n\n".join(assistant_content) if assistant_content else "",
                        "tool_calls": [
                            {
                                "id": call.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": call.get("name"),
                                    "arguments": json.dumps(call.get("arguments", {})) if isinstance(call.get("arguments", {}), dict) else call.get("arguments", "{}")
                                }
                            }
                            for call in tool_calls
                        ]
                    })
                    
                    # Execute tool calls and add results to conversation
                    for call in tool_calls:
                        name = call.get("name")
                        args = call.get("arguments", {})
                        call_id = call.get("call_id")
                        
                        if name in tool_functions:
                            try:
                                tool_functions[name](idx, **args)
                                tool_output = "Tool executed successfully"
                            except Exception as e:
                                print(f"Error executing tool {name}: {e}")
                                tool_output = f"Error: {e}"
                            
                            # Add tool call result to conversation
                            conversation.append({
                                "role": "tool",
                                "content": json.dumps({
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "name": name,
                                    "output": tool_output
                                })
                            })
                    
                    # Create follow-up request with conversation history
                    follow_up_tasks.append(
                        VLLMServeRunner.client.responses.create(
                            input=conversation,
                            tools=tools,
                            tool_choice="auto",
                            instructions="You've finished editing the knowledge base, now output your code in the specified format.",
                            **self.client_kwargs,
                        )
                    )
                else:
                    # No tools called, use None placeholder
                    follow_up_tasks.append(None)
            
            # Execute follow-up requests for responses that had tool calls
            num_follow_ups = sum(1 for t in follow_up_tasks if t is not None)
            print(f"[DEBUG] Processing {num_follow_ups} follow-up requests")
            
            final_responses = []
            if num_follow_ups > 0:
                # Execute follow-up tasks in parallel
                follow_up_results = await asyncio.gather(
                    *[task for task in follow_up_tasks if task is not None],
                    return_exceptions=True
                )
                
                # Match results back to original order
                result_idx = 0
                for task in follow_up_tasks:
                    if task is not None:
                        result = follow_up_results[result_idx]
                        if isinstance(result, Exception):
                            print(f"[ERROR] Follow-up request failed: {result}")
                            final_responses.append(None)
                        else:
                            final_responses.append(result)
                        result_idx += 1
                    else:
                        final_responses.append(None)
            else:
                final_responses = [None] * len(responses)

            output_texts = []

            # Extract text from final responses (after tool execution) or original responses
            for idx, (response, follow_up_response) in enumerate(zip(responses, final_responses)):
                toutput = []
                
                # First, collect any output from the original response (reasoning, initial thoughts)
                for output in getattr(response, "output", []) or []:
                    if hasattr(output, "content") and output.content:
                        for c in output.content:
                            if c.type == "output_text":
                                toutput.append(c.text)
                
                # If there's a follow-up response (tool was called), append its output
                if follow_up_response is not None:
                    for output in getattr(follow_up_response, "output", []) or []:
                        if hasattr(output, "content") and output.content:
                            for c in output.content:
                                if c.type == "output_text":
                                    toutput.append(c.text)
                
                # CRITICAL FIX: Return list[list[str]] not list[str]
                # Each response should be wrapped in a list to match expected format
                final_text = "\n\n".join(toutput) if toutput else ""
                
                # Ensure we always return non-None strings
                if not final_text:
                    print(f"[WARNING] Empty output for response {idx}")
                
                output_texts.append([final_text])  # Wrap in list!
            
            # Validate output format
            assert len(output_texts) == len(prompts), f"Output count mismatch: {len(output_texts)} != {len(prompts)}"
            assert all(isinstance(out, list) for out in output_texts), "All outputs must be lists"
            assert all(len(out) > 0 for out in output_texts), "All output lists must have at least one element"
            
            print(f"[DEBUG] Returning {len(output_texts)} outputs, format check: {type(output_texts[0]) if output_texts else 'N/A'}")
            print(f"[DEBUG] Sample output length: {len(output_texts[0][0]) if output_texts and output_texts[0] else 0} chars")
            print(f"[DEBUG] Output format validated: list[list[str]] with {len(output_texts)} items")
            import pdb; pdb.set_trace()
            return output_texts
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _run_batch_async())
                    return future.result()
            else:
                return loop.run_until_complete(_run_batch_async())
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(_run_batch_async())
        except openai.APITimeoutError as e:
            print(f"API Timeout Error: {e}")
            print("Consider increasing the timeout value in args.openai_timeout")
            raise