try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        TextContent,
        Author,
        Message,
        Role,
        ReasoningEffort,
        SystemContent,
        ToolNamespaceConfig,
        ToolDescription,
        DeveloperContent,
    )
    from vllm.inputs import TokensPrompt
    OPENAI_HARMONY_AVAILABLE = True
except ImportError:
    print("Cannot import openai_harmony")
    OPENAI_HARMONY_AVAILABLE = False

import json
from lcb_runner.runner.base_runner import BaseRunner


class VLLMKflowRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        if "__" in model_tokenizer_path:
            reasoning_effort = model_tokenizer_path.split("__")[-1]
            model_tokenizer_path = model_tokenizer_path.split("__")[0]
        if model_tokenizer_path.startswith("openai/"):
            self.llm = LLM(model_tokenizer_path, dtype="bfloat16", max_model_len=self.args.max_tokens,tensor_parallel_size=8, gpu_memory_utilization=0.9, async_scheduling=True, enable_prefix_caching=False, cuda_graph_sizes=[2048], compilation_config={"pass_config":{"enable_fi_allreduce_fusion":True,"enable_noop":True},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"})
        else:
            self.llm = LLM(
                model=model_tokenizer_path,
                tokenizer=model_tokenizer_path,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                enforce_eager=False,
                disable_custom_all_reduce=False,
                enable_prefix_caching=args.enable_prefix_caching,
                trust_remote_code=args.trust_remote_code,
            )
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=int(0.9*self.args.max_tokens),
            temperature=self.args.temperature,
            # top_p=self.args.top_p,
            stop_token_ids=stop_token_ids,
        )

        self.kflow_mem = []
        self.tool_functions = {"add_knowledge": self._add_kflow_memory, "remove_knowledge": self._remove_kflow_memory}
        self.tools = [
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
    
    def process_tool_calls(self, id: int, message: Message, tool_functions: dict) -> list[Message]:
        """
        Process tool calls in the message and return response messages.
        
        Args:
            id: The memory id for kflow operations
            message: The assistant message containing tool calls
            tool_functions: Dictionary mapping tool names to functions
            
        Returns:
            List of tool response messages
        """        
        response_messages = []
        
        # Check if message has recipients indicating tool calls
        if not hasattr(message, 'recipient') or not message.recipient:
            return response_messages
        
        # Parse the recipient to get tool namespace and function name
        # Format: "KnowledgeFlow.add_knowledge" or "KnowledgeFlow.remove_knowledge"
        recipient = message.recipient
        if '.' in recipient:
            namespace, tool_name = recipient.rsplit('.', 1)
        else:
            return response_messages
        
        # Get the tool function
        if tool_name not in tool_functions:
            print(f"Warning: Unknown tool {tool_name}")
            return response_messages
        
        # Parse the arguments from the message content
        # The content should be JSON formatted
        try:
            # Extract text content from the message
            text_content = None
            if hasattr(message, 'content') and isinstance(message.content, list):
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        text_content = content_item.text
                        break
            
            if text_content is None:
                print("Warning: No text content found in message")
                return response_messages
            
            # Parse JSON arguments
            args = json.loads(text_content)
            
            # Call the tool function with the id and parsed arguments
            tool_function = tool_functions[tool_name]
            if tool_name == "add_knowledge":
                knowledge = args.get("knowledge")
                if knowledge:
                    tool_function(id, knowledge)
                    # Create a success response message
                    response_msg = Message(author=Author(role=Role.TOOL, name="KnowledgeFlow"), content=[TextContent(text=f"Successfully added knowledge to the knowledge base.")])
                    response_messages.append(response_msg)
            elif tool_name == "remove_knowledge":
                knowledge_id = args.get("knowledge_id")
                if knowledge_id is not None:
                    tool_function(id, knowledge_id)
                    # Create a success response message
                    response_msg = Message(author=Author(role=Role.TOOL, name="KnowledgeFlow"), content=[TextContent(text=f"Successfully removed knowledge ID {knowledge_id} from the knowledge base.")])
                    response_messages.append(response_msg)
        
        except json.JSONDecodeError as e:
            print(f"Error parsing tool call arguments: {e}")
        except Exception as e:
            print(f"Error executing tool call: {e}")
        
        return response_messages


    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        # Disable Cache for KFlow runs
        if self.args.use_cache:
            raise ValueError("Cache is not supported for VLLMKflowRunner")
        # need to guarantee the prompts are in the same order across rounds!
        if len(self.kflow_mem) == 0:
            self.kflow_mem = {i: [] for i in range(len(prompts))}
        outputs = [None for _ in prompts]
        remaining_prompts = []
        harmony_messages = []

        # process the prompt in the openai format
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        REASONING_EFFORT = {
            "high": ReasoningEffort.HIGH,
            "medium": ReasoningEffort.MEDIUM,
            "low": ReasoningEffort.LOW,
        }
        system_message_content = SystemContent.new().with_reasoning_effort(REASONING_EFFORT["high"])
        tools= ToolNamespaceConfig(
            name="KnowledgeFlow", description="Knowledge Management Tools: add useful knowledge for other's to learn from, remove duplicate knowledges.", tools=[ToolDescription.new(name=tool['name'],
                                description=tool['description'],
                                parameters=tool['parameters']) for tool in self.tools]
        )

        system_message_content = system_message_content.with_tools(tools)

        for ctr, prompt in enumerate(prompts):
            harmony_message = []
            # system msg (for High Reasoning Effort)
            harmony_message.append(Message.from_role_and_content(Role.SYSTEM, system_message_content))
            # Developer content (for KFlow memory)
            harmony_message.append(Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions("If there is a given reference solution, examine it, rethink your approach. If you think the reference solution is wrong, use KnowledgeFlow to add the wrong solution and why it's wrong as knowledge to the knowledge base for future reference, if there is duplicate knowledge feel free to remove it. \n\n If no reference solution is given, try your best to solve the problem and output the solution in the right format.")))
            # KFlow memory content
            current_memory = self._get_kflow_memory(ctr)
            harmony_message.append(Message.from_role_and_content(Role.USER, prompt + "\n\n" + current_memory))
            convo = Conversation.from_messages(harmony_message)
            prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            # change it to tokens prompt
            token_prompt = TokensPrompt(prompt_token_ids=prefill_ids)
            harmony_messages.append(harmony_message)
            remaining_prompts.append(token_prompt)

        # processing prompts
        vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)

        # take the outputs
        for index, vllm_output in enumerate(vllm_outputs):
            outputs[index] = [o.text for o in vllm_output.outputs]

        # process the tool calls
        tbd_ids, tbd_convs = [], []
        for idx, vllm_output in enumerate(vllm_outputs):
            output_tokens = vllm_output.outputs[0].token_ids
            try:
                messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            except Exception as e:
                print(f"Error parsing messages from completion tokens: {e}")
                continue
            last_message = messages[-1]
            harmony_messages[idx].extend(messages)
            if hasattr(last_message, 'recipient') and last_message.recipient and "KnowledgeFlow" in last_message.recipient:
                response_messages = self.process_tool_calls(idx, last_message, self.tool_functions)
                # get the final response after tool calls
                for tool_response in response_messages:
                    # tool_response_msg = Message(author=Author(role=Role.TOOL, name="KnowledgeFlow"), content=[TextContent(text=tool_response)])
                    harmony_messages[idx].append(tool_response)
                
                # harmony_messages[idx].append(Message.from_role_and_content(Role.USER, "Please continue to complete the original task."))
                tbd_ids.append(idx)
                tbd_conv = Conversation.from_messages(harmony_messages[idx])
                tbd_conv = encoding.render_conversation_for_completion(tbd_conv, Role.ASSISTANT)
                # encoding.decode(tbd_convs[0]['prompt_token_ids'])
                tbd_conv = TokensPrompt(prompt_token_ids=tbd_conv)
                tbd_convs.append(tbd_conv)
        # re-process the to-be-done conversations
        if len(tbd_ids) > 0:
            tbd_vllm_outputs = self.llm.generate(tbd_convs, self.sampling_params)
            for i, idx in enumerate(tbd_ids):
                try:
                    messages = encoding.parse_messages_from_completion_tokens(tbd_vllm_outputs[i].outputs[0].token_ids, Role.ASSISTANT)
                    harmony_messages[idx].extend(messages)
                except Exception as e:
                    print(f"Error parsing messages from completion tokens: {e}")
                    continue
                outputs[idx] = [o.text for o in tbd_vllm_outputs[i].outputs]
        output_dict = {"outputs": outputs, "messages": harmony_messages, "kflow_mem": self.kflow_mem}
        return output_dict
