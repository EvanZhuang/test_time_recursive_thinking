"""
OpenAI Agent Runner with Knowledge Flow MCP Integration (Asyncio Version)

This runner connects to an existing MCP server (Master Control Program)
launched externally, typically via a script like `start_server.sh`.
The server runs as an SSE (Server-Sent Events) endpoint at a local address.

Architecture:
- Single shared MCP server for knowledge persistence.
- Solution evolution is tracked per-problem using a unique problem ID.
- The server persists solution history and reflections.
- Uses `asyncio` for asynchronous operations.

Three-Phase Workflow:


0. Fetch Knowledge (Runner -> MCP):
   - The runner directly calls the MCP server's `check_knowledge` tool
     before starting any agent work.
   - This retrieves all previous attempts and reflections for the specific problem.

1. Solver Agent (Pure Reasoning - NO MCP access):
   - A new, clean agent is created.
   - The knowledge context from Phase 0 is injected directly into its
     system prompt and first user message.
   - This agent is a pure reasoning agent with NO tool access.
   - It generates a solution based on the problem description and the
     in-context history of past (failed) attempts.

2. Knowledge Manager Agent (MCP `update_knowledge` access only):
   - A second, separate agent is created.
   - It receives the *original* knowledge context (from Phase 0) AND
     the *new* solution (from Phase 1).
   - Its sole purpose is to analyze the new solution in the context of
     the old ones and generate a reflection.
   - It then calls the `update_knowledge` tool on the MCP server, saving
     the new solution and the new reflection.

Benefits:
- Solver is a pure reasoning agent (no tool calls, faster, less complex).
- Knowledge is injected directly into the prompt (no extra latency during solve).
- Clean separation of concerns (solving vs. knowledge curation).
- 100% knowledge persistence (manager always runs if solver succeeds).
"""

import os
import re
import random
import asyncio
import json
import traceback
import contextlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from openai import AsyncAzureOpenAI
import httpx
import httpcore

# Import anyio utilities for MCP connection failure detection and cancel scope shielding
try:
    from anyio import ClosedResourceError, CancelScope
except ImportError:
    # Fallback if anyio is not installed (shouldn't happen in this environment)
    ClosedResourceError = Exception
    CancelScope = None

# Assuming 'agents' library and 'lcb_runner' are in the python path
from agents import Agent, Runner
from agents.mcp import MCPServer, MCPServerSse
from agents import set_tracing_disabled
from agents import ModelSettings
from agents.model_settings import Reasoning

from lcb_runner.utils.openai_reponse_custom import OpenAIResponsesModel
from lcb_runner.runner.base_runner import BaseRunner

# Import prompt templates
from lcb_runner.runner.agent_prompts import (
    SOLVER_SYSTEM_TEMPLATE,
    SOLVER_INPUT_TEMPLATE,
    SOLVER_WITH_STRATEGY_INPUT_TEMPLATE,
    KNOWLEDGE_MANAGER_SYSTEM,
    KNOWLEDGE_MANAGER_INPUT_TEMPLATE,
    KNOWLEDGE_MANAGER_WITH_STRATEGY_INPUT_TEMPLATE,
    KNOWLEDGE_MANAGER_WITHONLY_STRATEGY_INPUT_TEMPLATE,
    TEST_GENERATOR_SYSTEM,
    TEST_GENERATOR_INPUT_TEMPLATE,
    KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_TEMPLATE,
    KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_AND_STRATEGY_TEMPLATE
)

# Import helper functions
from lcb_runner.runner.agent_helpers import (
    log_critical_error_to_file,
    get_timestamp,
    is_valid_solution,
    sanitize_solution,
    extract_code_from_text,
    extract_code_for_evaluation,
    format_tool_arguments,
    extract_output_from_messages,
    parse_strategies_from_knowledge,
    format_strategy_performance,
    parse_strategy_history_from_knowledge,
    parse_reference_solution_from_knowledge,
    reorder_test_results,
    extract_fn_name_from_problem,
    extract_tool_call_arguments,
    handle_raw_response_event,
    handle_run_item_event,
    log_test_failure_to_disk,
    extract_update_knowledge_args_from_result,
    extract_content_from_tool_result,
    is_closed_resource_error,
)

AGENTS_SDK_AVAILABLE = True
# Disable agents SDK internal tracing for performance
set_tracing_disabled(True)


class OAIAgentRunner(BaseRunner):
    """
    Implements a BaseRunner for OpenAI Agents using the Knowledge Flow
    three-phase architecture and the `asyncio` async library.
    
    This runner orchestrates the three phases:
    1. Fetch knowledge from MCP
    2. Run a pure-reasoning Solver Agent
    3. Run a Knowledge Manager Agent to update MCP
    
    It manages concurrency using a `asyncio.Semaphore` and serializes
    MCP access using a `asyncio.Lock` to prevent race conditions.
    """

    # Azure endpoint and key from environment variables (required)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    subscription_key = os.getenv("OPENAI_API_KEY")

    # Default MCP server URL (assumes it's running locally)
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

    # Prompt templates are now imported from agent_prompts module

    def __init__(self, args, model):
        """
        Initializes the agent runner.

        Args:
            args: Command-line arguments or config object. Must contain:
                - model (str): Model name in format "model__effort"
                               (e.g., "gpt-4o__high").
                - openai_timeout (int): Timeout in seconds for individual agent runs.
                - multiprocess_oai (int): Concurrency limit for batch runs.
                - logging_trace (bool): Enable detailed trace logging.
            model: The model name (string), passed from BaseRunner.
        """
        super().__init__(args, model)

        # Store logging_trace flag
        self.logging_trace = getattr(args, 'logging_trace', False)

        # Store enable_strategy flag
        self.enable_strategy = getattr(args, 'enable_strategy', False)

        # Store enable_test_gen flag (KM generates and executes test cases for objective ranking)
        self.enable_test_gen = getattr(args, 'enable_test_gen', False)

        if not AGENTS_SDK_AVAILABLE:
            raise ImportError(
                "openai-agents-python SDK is required for OAIAgentRunner. "
                "Install with: pip install openai-agents-python"
            )

        # Parse model name and reasoning effort
        if "__" not in args.model:
            raise ValueError(
                f"Model {args.model} is not a valid OpenAI Reasoning model. "
                "It must include reasoning effort, e.g., 'gpt-4o__high'"
            )
        
        model_name, reasoning_effort = args.model.split("__")
        model_name = model_name.replace("agent", "")  # Clean up name if needed
        
        self.client_kwargs = {
            "model": model_name,
            "reasoning": {"effort": reasoning_effort},
            "timeout": args.openai_timeout,
            "max_tokens": args.max_tokens,
        }

        # Set a long streaming timeout for the HTTP client
        # At least 5 minutes, or twice the agent timeout
        # For long-running reasoning tasks, we need generous timeouts
        streaming_timeout = max(args.openai_timeout * 2, 300)

        # Also set a higher SDK-level timeout (this is separate from httpx timeout)
        # The SDK timeout should be at least as high as the httpx timeout
        # Add 60s buffer for overhead
        sdk_timeout = streaming_timeout + 60

        # Clients will be initialized in _init_clients() inside the event loop
        self.client = None
        self.http_client = None

        print(f"[Config] Agent model: {model_name}, Effort: {reasoning_effort}")
        print(f"[Config] Agent run timeout: {self.client_kwargs['timeout']}s")


        # The MCP server connection is managed per-task
        # Each task creates its own connection for isolation
        self.mcp_server: Optional[MCPServer] = None

        # Roll-out settings
        self.roll_out_n = max(1, getattr(args, "roll_out_n", 1))

        # Reference solution display settings
        self.reference_sol_in_solver = getattr(args, "reference_sol_in_solver", False)

        # Test failure logging directory
        # Uses the same timestamp as MCP server for consistency
        kb_timestamp = os.getenv("KNOWLEDGE_BASE_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.test_failure_logs_dir = Path("./knowledge_base") / kb_timestamp / "runner_test_logs"
        self.test_failure_logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Config] Test failure logs directory: {self.test_failure_logs_dir}")

    def _get_http_client_config(self) -> tuple:
        """
        Returns HTTP client configuration (timeout, limits) for the current settings.
        """
        streaming_timeout = max(self.client_kwargs["timeout"] * 2, 300)
        custom_timeout = httpx.Timeout(
            connect=3600.0,
            read=float(streaming_timeout),
            write=3600.0,
            pool=3600.0
        )
        # Connection pooling for high concurrency
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        return custom_timeout, limits, streaming_timeout

    def _init_openai_client(self, http_client: httpx.AsyncClient):
        """
        Initializes the OpenAI client with a provided HTTP client.
        This must be called inside the running asyncio loop.
        """
        streaming_timeout = max(self.client_kwargs["timeout"] * 2, 300)
        api_version = "2025-04-01-preview"
        sdk_timeout = streaming_timeout + 60

        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=self.endpoint.replace("/openai/v1/", ""),
            api_key=self.subscription_key,
            timeout=sdk_timeout,
            http_client=http_client,
        )
        self.http_client = http_client
        print(f"[Config] OpenAI SDK timeout: {sdk_timeout}s")

    def _create_task_client(self, http_client: httpx.AsyncClient) -> AsyncAzureOpenAI:
        """
        Creates a fresh OpenAI client for a specific task.

        Each concurrent task should have its own client to avoid shared state issues
        with response ID tracking in multi-turn conversations.

        Args:
            http_client: The HTTP client to use for this task's OpenAI client.

        Returns:
            A new AsyncAzureOpenAI client instance.
        """
        streaming_timeout = max(self.client_kwargs["timeout"] * 2, 300)
        api_version = "2025-04-01-preview"
        sdk_timeout = streaming_timeout + 60

        return AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=self.endpoint.replace("/openai/v1/", ""),
            api_key=self.subscription_key,
            timeout=sdk_timeout,
            http_client=http_client,
        )

    def cleanup_servers(self):
        """Clean up MCP server reference for the next round."""
        self.mcp_server = None
        print("[MCP] Server reference cleaned up for next round")

    # Helper methods are now imported from agent_helpers

    async def _log_test_failure(
        self,
        problem_idx: int,
        test_results_json: str,
        solutions: List[str],
        problem_description: str,
        phase: str = "test_generator",
        additional_context: Optional[Dict] = None,
    ) -> None:
        """Wrapper that delegates to the helper function."""
        await log_test_failure_to_disk(
            self.test_failure_logs_dir,
            problem_idx,
            test_results_json,
            solutions,
            problem_description,
            phase,
            additional_context,
        )



    async def _process_stream_events(self, result: Any, problem_idx: int) -> (str, List[str]):
        """
        Consumes the entire agent event stream, prints verbose logs,
        and extracts the final output text.
        
        This function is used when `STREAM_DEBUG` is true.

        Args:
            result: The streamable result object from `Runner.run_streamed`.
            problem_idx: The index of the problem for logging.

        Returns:
            A tuple of (final_output_text, tool_call_history).
        """
        output_text = ""
        tool_calls_count = 0
        response_count = 0
        in_output_mode = False
        tool_call_history = []
        check_knowledge_calls = 0
        debug_events = os.getenv("DEBUG_STREAMING", "false").lower() == "true"

        print(f"\n{'=' * 80}")
        print(f"┌─ Problem {problem_idx} │ Agent Execution Started │ {get_timestamp()}")
        print(f"{'=' * 80}\n")

        stream_start_time = datetime.now()
        print(f"[HTTP Streaming] Started at {stream_start_time.strftime('%H:%M:%S')}")

        async def _stream_events():
            """Inner function to process the stream."""
            nonlocal output_text, in_output_mode, response_count, \
                     tool_calls_count, check_knowledge_calls
            
            last_chunk_time = datetime.now()

            # The 'agents' library uses asyncio
            async for event in result.stream_events():
                current_time = datetime.now()
                chunk_gap = (current_time - last_chunk_time).total_seconds()
                last_chunk_time = current_time

                if chunk_gap > 30:  # Log if gap between chunks is > 30s
                    print(f"[HTTP Streaming] Long gap detected: {chunk_gap:.1f}s since last chunk")

                if debug_events:
                    print(f"\n[DEBUG] Event type: {event.type}, Event: {event}")

                # Handle raw response events (token-by-token)
                if event.type == "raw_response_event":
                    output_text, in_output_mode, response_count = handle_raw_response_event(
                        event, output_text, in_output_mode, response_count
                    )

                # Handle run item events (functional completion)
                elif event.type == "run_item_stream_event":
                    tool_calls_count, check_knowledge_calls = handle_run_item_event(
                        event.item, tool_calls_count, tool_call_history, check_knowledge_calls
                    )

                # Handle agent handoffs (if using multi-agent graphs)
                elif event.type == "agent_updated_stream_event":
                    if hasattr(event, 'agent') and hasattr(event.agent, 'name'):
                        print(f"\n┌─ [🔄 Agent Handoff] → {event.agent.name} │ {get_timestamp()}")

        # Wrap streaming in a try/except to log duration and provide detailed error info
        try:
            await _stream_events()
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"[✓ HTTP Streaming] Completed successfully after {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
        except asyncio.TimeoutError:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming TIMEOUT] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print("[INFO] The configured timeout was exceeded during streaming")
            print("[INFO] Consider increasing the timeout or checking the model response time")
            raise
        except httpx.ReadTimeout as e:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming ERROR] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print(f"[❌ Error Type] ReadTimeout: {str(e)}")
            print("[INFO] The HTTP read timeout (default 600s/10min) was exceeded")
            print("[INFO] This usually means:")
            print("       1. The model is taking too long to generate tokens")
            print("       2. The streaming connection was idle for too long")
            print("       3. Network issues interrupted the stream")
            print("[FIX] To resolve:")
            print("      1. Increase AsyncAzureOpenAI timeout parameter")
            print("      2. Use http_client with custom timeout in client initialization")
            print("      3. Check network connectivity and latency")
            raise
        except httpx.ReadError as e:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming ERROR] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print(f"[❌ Error Type] ReadError: {str(e)}")
            print("[INFO] The streaming connection was interrupted or closed unexpectedly")
            print("[INFO] This usually means:")
            print("       1. The server closed the connection (possibly due to server-side timeout)")
            print("       2. Network issues caused connection interruption")
            print("       3. The model endpoint had an internal error")
            print("[FIX] To resolve:")
            print("      1. Check if your timeout settings match server expectations")
            print("      2. Verify network stability between client and Azure endpoint")
            print("      3. Check Azure OpenAI service status and quotas")
            raise
        except httpx.ConnectTimeout as e:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming ERROR] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print(f"[❌ Error Type] ConnectTimeout: {str(e)}")
            print("[INFO] Could not establish connection to the model endpoint")
            raise
        except httpx.HTTPError as e:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming ERROR] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print(f"[❌ Error Type] {type(e).__name__}: {str(e)}")
            print("[INFO] HTTP error occurred during streaming")
            raise
        except Exception as e:
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            print(f"\n[❌ HTTP Streaming ERROR] After {stream_duration:.1f}s ({stream_duration/60:.1f} minutes)")
            print(f"[❌ Error Type] {type(e).__name__}: {str(e)}")
            raise  # Re-raise the exception

        # Print execution summary
        print(f"\n{'=' * 80}")
        print(f"└─ Problem {problem_idx} │ Execution Complete │ {get_timestamp()}")
        print("   📊 Summary:")
        print(f"        • Tool calls: {tool_calls_count}")
        print(f"        • Responses: {response_count}")
        print(f"        • Output length from streaming: {len(output_text)} characters")

        if tool_call_history:
            print(f"        • Tool sequence: {' → '.join(tool_call_history)}")
        
        # NOTE: Workflow warnings removed as requested by user.

        print(f"{'=' * 80}\n")

        return output_text, tool_call_history



    async def _run_solver_agent(
        self,
        prompt: List[Dict[str, str]],
        problem_idx: int,
        problem_description: str,
        knowledge_items: str,
        reference_solution: str,
        rollout_id: Optional[int] = None,
        num_rollouts: int = 1,
        strategy: Optional[Dict[str, str]] = None,
        openai_client: Optional[AsyncAzureOpenAI] = None,
    ) -> (str, List[Any]):
        """
        Executes Phase 1: The Solver Agent.

        This agent is pure reasoning and has NO tool access. The
        knowledge (lessons learned) and reference solution are injected
        directly into its prompt.

        Args:
            prompt: The original user prompt (OpenAI message format).
            problem_idx: The index of the problem for logging.
            problem_description: A short description of the problem.
            knowledge_items: Standalone lessons learned from previous attempts.
            reference_solution: The current best solution (code).
            rollout_id: Optional 0-indexed rollout number.
            num_rollouts: Total number of rollouts.
            strategy: Optional strategy dict with 'key_technique' and 'what_to_try'.
            openai_client: Per-task OpenAI client to avoid shared state issues.

        Returns:
            A tuple of (final_output_text, message_history_list).
        """
        # Use per-task client if provided, otherwise fall back to shared client
        client = openai_client if openai_client is not None else self.client
        input_text = prompt[-1]["content"] if prompt else ""
        rollout_index = (rollout_id or 0) + 1  # 1-indexed for display
        rollout_label = (
            f"{problem_idx}"
            if rollout_id is None
            else f"{problem_idx} (rollout {rollout_index})"
        )
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"

        # Use class-level prompt templates
        system_instruction = SOLVER_SYSTEM_TEMPLATE

        # only one rollout uses the reference solution, others start fresh
        # if rollout_id == 0:
        #     reference_solution = reference_solution
        # else:
        #     reference_solution = "# No reference solution provided for this rollout."

        # Use placeholder if reference_sol_in_solver flag is not set
        if not self.reference_sol_in_solver:
            reference_solution = "(No reference solution available)"

        # Use strategy-aware template if strategy is provided
        if strategy and self.enable_strategy:
            full_input = SOLVER_WITH_STRATEGY_INPUT_TEMPLATE.format(
                knowledge_items=knowledge_items,
                reference_solution=reference_solution,
                input_text=input_text,
                rollout_index=rollout_index,
                num_rollouts=num_rollouts,
                strategy_key_technique=strategy.get('key_technique', ''),
                strategy_what_to_try=strategy.get('what_to_try', ''),
            )
            print(f"[Problem {rollout_label}] Using strategy: {strategy.get('key_technique', 'unknown')}")
        else:
            full_input = SOLVER_INPUT_TEMPLATE.format(
                knowledge_items=knowledge_items,
                reference_solution=reference_solution,
                input_text=input_text,
                rollout_index=rollout_index,
                num_rollouts=num_rollouts,
            )

        # Create solver agent (NO mcp_servers - pure reasoning)
        agent = Agent(
            name="Solver Agent",
            model=OpenAIResponsesModel(
                model=self.client_kwargs["model"],
                openai_client=client,
            ),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.client_kwargs["reasoning"]["effort"]),
                max_output_tokens=self.client_kwargs["max_tokens"],
                tool_choice="none",  # Explicitly disable tools
            ),
            instructions=system_instruction,
            # NO mcp_servers parameter
        )

        output_text = ""
        tool_call_history = []
        
        # Run agent
        if stream_debug:
            # Debug mode: use streaming with verbose output
            print(f"[Problem {rollout_label}] Running solver agent (no tools, pure reasoning)...")
            result = Runner.run_streamed(
                starting_agent=agent,
                input=full_input,
            )
            output_text, tool_call_history = await self._process_stream_events(result, problem_idx)
        else:
            # Production mode: regular async execution
            print(f"[Problem {rollout_label}] Running solver agent (no tools, pure reasoning)...")
            result = await Runner.run(
                starting_agent=agent,
                input=full_input,
            )

        # Extract messages
        messages = result.messages if hasattr(result, "messages") else []
        if not stream_debug and messages:
            print(f"[INFO] Extracted {len(messages)} messages from result")

        # Fallback: extract from messages if streaming didn't capture output
        if not output_text:
            if not stream_debug:
                print("[INFO] Extracting output from messages...")
            output_text = extract_output_from_messages(messages)

        # Final fallback: use final_output attribute
        if not output_text and hasattr(result, "final_output"):
            output_text = str(result.final_output)
            if not stream_debug:
                print(f"[INFO] Extracted {len(output_text)} chars from final_output")

        if output_text and not stream_debug:
            print(f"[INFO] Total output length: {len(output_text)} characters")

        # Attempt to extract clean Python code from markdown code blocks
        if output_text:
            extracted_code = extract_code_from_text(output_text)
            if extracted_code and len(extracted_code) > 0.3 * len(output_text.strip()):
                # If extracted code is substantial portion of output, use it
                if not stream_debug:
                    print(f"[INFO] Extracted {len(extracted_code)} chars of Python code from response")
                output_text = extracted_code
            elif not is_valid_solution(output_text) and not stream_debug:
                print(f"[WARNING] Output doesn't contain clear Python code structure")

        return output_text, messages

    async def _run_test_generator_agent(
        self,
        problem_idx: int,
        problem_description: str,
        solutions: List[str],
        id_mapping: Optional[Dict[int, int]] = None,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        mcp_server: Optional[MCPServerSse] = None,
    ) -> Optional[str]:
        """
        Executes Phase 1.5: The Test Generator Agent.

        This agent generates test cases based on the problem description,
        then executes them against all candidate solutions. It returns
        the execution results as a formatted string to be passed to the
        Knowledge Manager.

        Args:
            problem_idx: The index of the problem for logging.
            problem_description: The full problem description.
            solutions: List of candidate solution strings.
            id_mapping: Optional mapping from solution index to original_id.
                        If None, uses identity mapping.
            openai_client: Per-task OpenAI client to avoid shared state issues.
            mcp_server: Per-task MCP server connection to avoid shared state issues.

        Returns:
            A formatted string containing test execution results, or None if failed.
        """
        # Use per-task client if provided, otherwise fall back to shared client
        client = openai_client if openai_client is not None else self.client
        # Use per-task MCP server if provided, otherwise fall back to shared
        mcp_server = mcp_server if mcp_server is not None else self.mcp_server
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"

        # Use provided id_mapping or fall back to identity mapping
        if id_mapping is None:
            id_mapping = {i: i for i in range(len(solutions))}

        # Register candidate solutions with MCP server first
        # (The execute_generated_tests tool will look them up by problem_id)
        # Call MCP tool directly on per-task connection
        try:
            await self.mcp_server.call_tool(
                "register_candidate_solutions",
                {
                    "problem_id": f"problem_{problem_idx}",
                    "solutions_json": json.dumps(solutions),
                    "id_mapping_json": json.dumps(id_mapping),
                }
            )
            print(f"[Test Generator] Registered {len(solutions)} candidate solutions with mapping: {id_mapping}")
        except Exception as e:
            print(f"[WARNING] Failed to register candidate solutions: {e}")
            return None

        # Extract function name from problem description
        fn_name = extract_fn_name_from_problem(problem_description)

        # Format fn_name argument for template
        if fn_name:
            fn_name_arg = f'"{fn_name}"'
        else:
            fn_name_arg = "null"

        # Format solutions code for display in template
        solutions_code_parts = []
        for i, sol in enumerate(solutions):
            # Use mapped solution ID for display
            display_id = id_mapping.get(i, i) + 1  # 1-indexed for display
            solutions_code_parts.append(f"### Solution {display_id}:\n```python\n{sol}\n```")
        solutions_code = "\n\n".join(solutions_code_parts)

        # Build input text from template
        input_text = TEST_GENERATOR_INPUT_TEMPLATE.format(
            problem_description=problem_description,
            num_solutions=len(solutions),
            solutions_code=solutions_code,
            problem_idx=problem_idx,
            fn_name_arg=fn_name_arg,
        )

        # Create test generator agent
        agent = Agent(
            name="Test Generator Agent",
            model=OpenAIResponsesModel(
                model=self.client_kwargs["model"],
                openai_client=client,
            ),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.client_kwargs["reasoning"]["effort"]),
                tool_choice="auto",
            ),
            instructions=TEST_GENERATOR_SYSTEM,
            mcp_servers=[mcp_server],  # Give access to execute_generated_tests tool
        )

        # Capture test results from tool call
        captured_test_results = None

        print(f"[Test Generator] Starting test generation for problem {problem_idx}... (fn_name={fn_name})")

        if stream_debug:
            print(f"[DEBUG] Starting streamed test generator run...")
            result = Runner.run_streamed(
                starting_agent=agent,
                input=input_text,
            )

            # Parse test results from stream
            event_count = 0
            event_types_seen = set()
            async for event in result.stream_events():
                event_count += 1
                event_type = type(event).__name__
                event_types_seen.add(event_type)

                if not hasattr(event, 'name'):
                    continue

                # Handle tool_output event - this contains the MCP tool results
                if event.name == "tool_output":
                    print(f"[DEBUG] Processing tool_output event...")
                    if hasattr(event, 'item') and hasattr(event.item, 'output'):
                        output = event.item.output
                        print(f"[DEBUG]   Raw output type: {type(output)}")
                        print(f"[DEBUG]   Raw output repr (first 1000): {repr(output)[:1000]}")

                        # Output is wrapped in {"type":"text","text":"..."} structure
                        # Extract the actual text content
                        output_text = None
                        if isinstance(output, dict) and 'text' in output:
                            output_text = output['text']
                            print(f"[DEBUG]   Extracted from dict['text']")
                        elif isinstance(output, str):
                            # Try to parse as JSON
                            try:
                                parsed = json.loads(output)
                                print(f"[DEBUG]   Parsed string as JSON: {type(parsed)}")
                                if isinstance(parsed, dict) and 'text' in parsed:
                                    output_text = parsed['text']
                                    print(f"[DEBUG]   Extracted from parsed dict['text']")
                                else:
                                    output_text = output
                                    print(f"[DEBUG]   Using raw string (parsed but no 'text' key)")
                            except json.JSONDecodeError as e:
                                output_text = output
                                print(f"[DEBUG]   Using raw string (JSON parse failed: {e})")
                        elif hasattr(output, 'text'):
                            output_text = output.text
                            print(f"[DEBUG]   Extracted from output.text attribute")
                        else:
                            output_text = str(output)
                            print(f"[DEBUG]   Converted to string")

                        print(f"[DEBUG]   Final output_text (first 500 chars): {output_text[:500] if output_text else 'None'}")

                        # Check if this contains test results
                        has_results = output_text and '"results"' in output_text
                        has_pass_rate = output_text and '"pass_rate"' in output_text
                        print(f"[DEBUG]   Contains 'results': {has_results}, Contains 'pass_rate': {has_pass_rate}")

                        if has_results and has_pass_rate:
                            captured_test_results = output_text
                            print(f"[Test Generator] Captured test results ({len(captured_test_results)} chars)")
                    else:
                        print(f"[DEBUG]   tool_output event missing item or output attr")
                        print(f"[DEBUG]   hasattr(event, 'item'): {hasattr(event, 'item')}")
                        if hasattr(event, 'item'):
                            print(f"[DEBUG]   hasattr(event.item, 'output'): {hasattr(event.item, 'output')}")

                # Handle tool_called event for logging
                elif event.name == "tool_called":
                    if hasattr(event, 'item') and hasattr(event.item, 'raw_item'):
                        raw_item = event.item.raw_item
                        if hasattr(raw_item, 'name') and raw_item.name == "execute_generated_tests":
                            print("[Test Generator] Calling execute_generated_tests...")

            print(f"[DEBUG] Stream finished. Total events: {event_count}")
            print(f"[DEBUG] Event types seen: {event_types_seen}")
        else:
            print(f"[DEBUG] Starting non-streamed test generator run...")
            result = await Runner.run(
                starting_agent=agent,
                input=input_text,
            )

            # Extract test results from new_items
            print(f"[DEBUG] Result type: {type(result).__name__}")
            print(f"[DEBUG] Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")

            if hasattr(result, 'new_items'):
                print(f"[DEBUG] new_items count: {len(result.new_items)}")
                for idx, item in enumerate(result.new_items):
                    item_type = type(item).__name__
                    print(f"[DEBUG]   Item {idx}: type={item_type}")
                    print(f"[DEBUG]     Item attributes: {[a for a in dir(item) if not a.startswith('_')]}")

                    if hasattr(item, 'output'):
                        output = item.output
                        print(f"[DEBUG]     output type: {type(output).__name__}")
                        output_str = str(output)
                        print(f"[DEBUG]     output str (first 500 chars): {output_str[:500]}")
                        has_results = '"results"' in output_str
                        has_pass_rate = '"pass_rate"' in output_str
                        print(f"[DEBUG]     Contains 'results': {has_results}, Contains 'pass_rate': {has_pass_rate}")

                        if has_results and has_pass_rate:
                            captured_test_results = output_str
                            print(f"[Test Generator] Captured test results ({len(captured_test_results)} chars)")
                    else:
                        print(f"[DEBUG]     No 'output' attribute")
            else:
                print(f"[DEBUG] Result has no 'new_items' attribute")

        if captured_test_results:
            print(f"[Test Generator] Successfully captured test results")
            return captured_test_results
        else:
            print(f"[WARNING] Test Generator did not return valid test results")
            return None

    async def _run_knowledge_manager_agent(
        self,
        problem_idx: int,
        problem_description: str,
        current_knowledge: str,
        new_solutions: List[str],
        num_rollouts: int = 1,
        test_results: Optional[str] = None,
        strategy_performance: Optional[List[Dict]] = None,
        successful_rollout_indices: Optional[List[int]] = None,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        mcp_server: Optional[MCPServerSse] = None,
    ) -> Any:
        """
        Executes Phase 2: The Knowledge Manager Agent.

        This agent's sole purpose is to reflect on the new solution,
        decide whether any prior knowledge should be disabled, and
        call the `update_knowledge` tool on the MCP server.

        Args:
            problem_idx: The index of the problem for logging.
            problem_description: A short description of the problem.
            current_knowledge: The string blob from Phase 0.
            new_solutions: List of new solution strings from Phase 1 rollouts.
            num_rollouts: Number of rollouts (used for strategy generation count).
            successful_rollout_indices: List of original rollout indices that succeeded.
                                        Used to build correct id_mapping.
            openai_client: Per-task OpenAI client to avoid shared state issues.
            mcp_server: Per-task MCP server connection to avoid shared state issues.

        Returns:
            The agent result object.
        """
        # Use per-task client if provided, otherwise fall back to shared client
        client = openai_client if openai_client is not None else self.client
        # Use per-task MCP server if provided, otherwise fall back to shared
        mcp_server = mcp_server if mcp_server is not None else self.mcp_server
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"

        # Use class-level prompt templates
        system_instruction = KNOWLEDGE_MANAGER_SYSTEM

        # Parse reference solution from knowledge context
        reference_solution, knowledge_without_ref = parse_reference_solution_from_knowledge(
            current_knowledge
        )
        if reference_solution:
            print(f"[Knowledge Manager] Found reference solution ({len(reference_solution)} chars)")

        # Create list of all solutions with original indices
        # original_id: 0 = reference solution, 1+ = rollout solutions (1-indexed by original rollout index)
        # NOTE: Shuffling is DISABLED for now to simplify debugging.
        # Solutions are presented in deterministic order:
        # - Solution 1 = reference (if exists)
        # - Solution 2+ = rollouts in order they appear in new_solutions
        all_solutions = []
        if reference_solution:
            all_solutions.append((0, reference_solution))  # (original_id, solution)

        # Use successful_rollout_indices if provided to get correct original_ids
        # Otherwise fall back to sequential indices (assumes all rollouts succeeded)
        if successful_rollout_indices is not None:
            for i, sol in enumerate(new_solutions):
                original_rollout_idx = successful_rollout_indices[i] if i < len(successful_rollout_indices) else i
                all_solutions.append((original_rollout_idx + 1, sol))  # original_id = rollout_idx + 1
        else:
            for idx, sol in enumerate(new_solutions):
                all_solutions.append((idx + 1, sol))  # original_id = idx + 1 for rollouts

        # DISABLED: Shuffle to remove positional bias
        # random.shuffle(all_solutions)
        # print(f"[Knowledge Manager] Shuffled solutions to remove positional bias")
        print(f"[Knowledge Manager] Using deterministic solution order (shuffle disabled)")

        # Create mapping (no shuffle means position in ordered list -> original_id)
        id_mapping = {}  # {idx: original_id}
        reverse_mapping = {}  # {original_id: idx}
        ordered_solutions = []
        for idx, (original_id, solution) in enumerate(all_solutions):
            id_mapping[idx] = original_id
            reverse_mapping[original_id] = idx
            ordered_solutions.append(solution)

        print(f"[Knowledge Manager] {len(ordered_solutions)} solutions. Mapping: {id_mapping}")

        # Since no shuffle, test_results already match the solution order
        # (both use: reference at position 1, rollouts at 2+)
        # No reordering needed - just pass through
        # Prefer formatted_results (markdown) if available for better KM readability
        reordered_test_results = test_results
        if test_results:
            try:
                results_data = json.loads(test_results) if isinstance(test_results, str) else test_results
                if isinstance(results_data, dict) and "formatted_results" in results_data:
                    reordered_test_results = results_data["formatted_results"]
                    print(f"[Knowledge Manager] Using formatted_results markdown ({len(reordered_test_results)} chars)")
                else:
                    print(f"[Knowledge Manager] Using test results as-is (no shuffle, no reorder needed)")
            except (json.JSONDecodeError, TypeError):
                print(f"[Knowledge Manager] Using test results as-is (no shuffle, no reorder needed)")

        # Register solutions with MCP server (for selected_solution_id lookup)
        # Call MCP tool directly on per-task connection
        try:
            await self.mcp_server.call_tool(
                "register_candidate_solutions",
                {
                    "problem_id": f"problem_{problem_idx}",
                    "solutions_json": json.dumps(ordered_solutions),
                    "id_mapping_json": json.dumps(id_mapping),  # Pass mapping for reverse lookup
                }
            )
            print(f"[Knowledge Manager] Registered {len(ordered_solutions)} candidate solutions")
        except Exception as e:
            print(f"[WARNING] Failed to register candidate solutions: {e}")

        # Format solutions block with anonymous labels (Solution 1, 2, 3, etc.)
        # No indication of which is reference vs rollout
        solutions_lines = []
        for sol_idx, solution in enumerate(ordered_solutions):
            # 1-indexed for readability in prompt
            solutions_lines.append(
                f"### Solution {sol_idx + 1}\n```python\n{solution}\n```"
            )
        solutions_block = "\n\n".join(solutions_lines)

        # Pre-compute strategy blocks (used by strategy-enabled templates)
        strategy_perf_block = ""
        strategy_history_block = ""
        if self.enable_strategy:
            if strategy_performance:
                strategy_perf_block = format_strategy_performance(strategy_performance)
                print(f"[Knowledge Manager] Including strategy performance for {len(strategy_performance)} strategies")
            strategy_history_block = parse_strategy_history_from_knowledge(current_knowledge)
            if strategy_history_block:
                print(f"[Knowledge Manager] Including strategy history ({len(strategy_history_block)} chars)")

        # Select appropriate template based on mode
        # Priority: test_results (from Phase 1.5) > strategy > default
        if reordered_test_results and self.enable_test_gen and self.enable_strategy:
            # Test results + strategy mode: combine both features
            input_text = KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_AND_STRATEGY_TEMPLATE.format(
                problem_description=problem_description,
                knowledge_items=knowledge_without_ref,
                strategy_performance_block=strategy_perf_block,
                strategy_history_block=strategy_history_block,
                solutions_block=solutions_block,
                test_results=reordered_test_results,
                problem_idx=problem_idx,
                num_rollouts=num_rollouts,
            )
            print(f"[Knowledge Manager] Using test results + strategy template (test_results={len(reordered_test_results)} chars, {num_rollouts} strategies)")
        elif reordered_test_results and self.enable_test_gen:
            # Test results provided from Phase 1.5 - use the simplified template
            input_text = KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_TEMPLATE.format(
                problem_description=problem_description,
                knowledge_items=knowledge_without_ref,
                solutions_block=solutions_block,
                test_results=reordered_test_results,
                problem_idx=problem_idx,
            )
            print(f"[Knowledge Manager] Using test results template (test_results={len(reordered_test_results)} chars)")
        elif self.enable_strategy:
            # Strategy mode without test results
            input_text = KNOWLEDGE_MANAGER_WITH_STRATEGY_INPUT_TEMPLATE.format(
                problem_description=problem_description,
                knowledge_items=knowledge_without_ref,
                strategy_performance_block=strategy_perf_block,
                strategy_history_block=strategy_history_block,
                solutions_block=solutions_block,
                problem_idx=problem_idx,
                num_rollouts=num_rollouts,
            )
            print(f"[Knowledge Manager] Using strategy-aware template (requesting {num_rollouts} strategies)")
        else:
            input_text = KNOWLEDGE_MANAGER_INPUT_TEMPLATE.format(
                problem_description=problem_description,
                knowledge_items=knowledge_without_ref,
                solutions_block=solutions_block,
                problem_idx=problem_idx
            )

        # Create knowledge manager agent
        agent = Agent(
            name="Knowledge Manager Agent",
            model=OpenAIResponsesModel(
                model=self.client_kwargs["model"],
                openai_client=client,
            ),
            model_settings=ModelSettings(
                # Knowledge management doesn't need high reasoning
                reasoning=Reasoning(effort=self.client_kwargs["reasoning"]["effort"]),
                tool_choice="auto",
            ),
            instructions=system_instruction,
            mcp_servers=[mcp_server], # Give it access to MCP
        )

        # Run agent and capture selected_solution_id + reference_solution_is_wrong from tool call
        captured_ref_flag = None
        captured_solution_id = None  # New: capture ID instead of full solution

        if stream_debug:
            print(f"\n[Knowledge Manager] Starting knowledge update for problem {problem_idx}... ")
            result = Runner.run_streamed(
                starting_agent=agent,
                input=input_text,
            )
            
            # Parse tool calls from stream
            async for event in result.stream_events():
                # check if event has item

                if not hasattr(event,'name'):
                    continue

                # Handle tool_called event structure
                if event.name == "tool_called":
                    # Check if the event has the nested structure
                    if hasattr(event, 'item') and hasattr(event.item, 'raw_item'):
                        raw_item = event.item.raw_item
                        # Check if this is the update_knowledge tool
                        if hasattr(raw_item, 'name') and raw_item.name == "update_knowledge":
                            print("[Knowledge Manager] Calling update_knowledge...")
                            # Get arguments from raw_item
                            if hasattr(raw_item, 'arguments'):
                                args = raw_item.arguments
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
                                if isinstance(args, dict):
                                    if "reference_solution_is_wrong" in args:
                                        captured_ref_flag = args["reference_solution_is_wrong"]
                                        print(f"[Knowledge Manager] Captured reference_solution_is_wrong: {captured_ref_flag}")
                                    if "selected_solution_id" in args and captured_solution_id is None:
                                        captured_solution_id = args["selected_solution_id"]
                                        print(f"[Knowledge Manager] Captured selected_solution_id: {captured_solution_id}")
        else:
            print(f"[Knowledge Manager] Updating knowledge for problem {problem_idx}...")
            result = await Runner.run(
                starting_agent=agent,
                input=input_text,
            )

            # Extract tool call arguments from result using helper
            captured_ref_flag, captured_solution_id = extract_update_knowledge_args_from_result(result)

            # Debug: Log if extraction failed
            if captured_solution_id is None:
                print(f"[WARNING] Could not extract selected_solution_id from Knowledge Manager result")
                print(f"[DEBUG] result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                if hasattr(result, 'messages'):
                    print(f"[DEBUG] messages count: {len(result.messages)}")
                    for i, msg in enumerate(result.messages[:3]):  # Show first 3
                        print(f"[DEBUG] msg[{i}] type: {type(msg).__name__}, attrs: {[a for a in dir(msg) if not a.startswith('_')][:10]}")

        print("[Knowledge Manager] Knowledge update complete.")

        # Map shuffled selected_solution_id back to original_id
        # Model returns 1-indexed ID (Solution 1, 2, 3) -> convert to 0-indexed shuffled_idx
        # Then use id_mapping to get original_id (0 = reference, 1+ = rollouts)
        if captured_solution_id is not None:
            # Convert model's 1-indexed ID to 0-indexed shuffled_idx
            shuffled_idx = captured_solution_id - 1 if captured_solution_id >= 1 else 0

            # Map shuffled index back to original ID using the mapping
            original_id = id_mapping.get(shuffled_idx, shuffled_idx)

            print(f"[Trace] selected_solution_id={captured_solution_id} -> shuffled_idx={shuffled_idx} -> original_id={original_id}")

            # Derive reference_solution_is_wrong from original_id (not shuffled)
            # original_id == 0 means reference was selected
            derived_ref_flag = (original_id != 0)
            result._captured_ref_flag = derived_ref_flag
            print(f"[Trace] Derived reference_solution_is_wrong={derived_ref_flag} from original_id={original_id}")

            # Map original_id to actual solution
            result._captured_solution_id = captured_solution_id  # Store the model's selection
            result._captured_original_id = original_id  # Store the original ID for reference

            # Use reverse_mapping and ordered_solutions to look up the solution
            # This correctly handles cases where rollouts failed and original_id doesn't
            # match the index in new_solutions
            if original_id in reverse_mapping:
                idx = reverse_mapping[original_id]
                result._captured_best_solution = ordered_solutions[idx]
                if original_id == 0:
                    print(f"[Knowledge Manager] Mapped original_id 0 -> reference solution ({len(result._captured_best_solution) if result._captured_best_solution else 0} chars)")
                else:
                    print(f"[Knowledge Manager] Mapped original_id {original_id} -> rollout solution ({len(result._captured_best_solution)} chars)")
            else:
                # Invalid original_id, fallback to first valid solution
                print(f"[WARNING] Invalid original_id {original_id} not in reverse_mapping {reverse_mapping}, falling back")
                result._captured_best_solution = ordered_solutions[0] if ordered_solutions else None
        else:
            result._captured_ref_flag = False
            result._captured_solution_id = None
            result._captured_original_id = None
            result._captured_best_solution = None
            print(f"[Trace] No selected_solution_id found, defaulting reference_solution_is_wrong to False")

        return result

    async def _run_knowledge_manager_strategy_only(
        self,
        problem_idx: int,
        problem_description: str,
        num_strategies: int = 1,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        mcp_server: Optional[MCPServerSse] = None,
    ) -> Any:
        """
        Executes a lightweight version of Phase 2: The Knowledge Manager Agent
        that only generates strategies without updating knowledge.

        This function is synchronous and does not use streaming.

        Args:
            problem_idx: The index of the problem for logging.
            problem_description: A short description of the problem.
            num_strategies: Number of strategies to generate.
            openai_client: Per-task OpenAI client to avoid shared state issues.
            mcp_server: Per-task MCP server connection to avoid shared state issues.

        Returns:
            The agent result object.
        """
        # Use per-task client if provided, otherwise fall back to shared client
        client = openai_client if openai_client is not None else self.client
        # Use per-task MCP server if provided, otherwise fall back to shared
        mcp_server = mcp_server if mcp_server is not None else self.mcp_server
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"
        input_text = KNOWLEDGE_MANAGER_WITHONLY_STRATEGY_INPUT_TEMPLATE.format(
            problem_description=problem_description,
            problem_idx=problem_idx,
            num_rollouts=num_strategies,
        )

        # Create knowledge manager agent
        agent = Agent(
            name="Knowledge Manager Agent",
            model=OpenAIResponsesModel(
                model=self.client_kwargs["model"],
                openai_client=client,
            ),
            model_settings=ModelSettings(
                # Knowledge management doesn't need high reasoning
                reasoning=Reasoning(effort=self.client_kwargs["reasoning"]["effort"]),
                tool_choice="auto",
            ),
            mcp_servers=[mcp_server], # Give it access to MCP
        )

        captured_solution_id = None


        if stream_debug:
            print(f"\n[Knowledge Manager] Starting knowledge update for problem {problem_idx}...")
            result = Runner.run_streamed(
                starting_agent=agent,
                input=input_text,
            )
            
            # Parse tool calls from stream
            async for event in result.stream_events():
                # check if event has item

                if not hasattr(event,'name'):
                    continue

                # Handle tool_called event structure
                if event.name == "tool_called":
                    # Check if the event has the nested structure
                    if hasattr(event, 'item') and hasattr(event.item, 'raw_item'):
                        raw_item = event.item.raw_item
                        # Check if this is the update_knowledge tool
                        if hasattr(raw_item, 'name') and raw_item.name == "update_knowledge":
                            print("[Knowledge Manager] Calling update_knowledge...")
                            # Get arguments from raw_item
                            if hasattr(raw_item, 'arguments'):
                                args = raw_item.arguments
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
                                if isinstance(args, dict):
                                    if "reference_solution_is_wrong" in args:
                                        captured_ref_flag = args["reference_solution_is_wrong"]
                                        print(f"[Knowledge Manager] Captured reference_solution_is_wrong: {captured_ref_flag}")
                                    if "selected_solution_id" in args and captured_solution_id is None:
                                        captured_solution_id = args["selected_solution_id"]
                                        print(f"[Knowledge Manager] Captured selected_solution_id: {captured_solution_id}")
        else:
            print(f"[Knowledge Manager] Updating knowledge for problem {problem_idx}...")
            result = await Runner.run(
                starting_agent=agent,
                input=input_text,
            )

            # Extract tool call arguments from result using helper
            captured_ref_flag, captured_solution_id = extract_update_knowledge_args_from_result(result)

            # Debug: Log if extraction failed
            if captured_solution_id is None:
                print(f"[WARNING] Could not extract selected_solution_id from Knowledge Manager result")
                print(f"[DEBUG] result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                if hasattr(result, 'messages'):
                    print(f"[DEBUG] messages count: {len(result.messages)}")
                    for i, msg in enumerate(result.messages[:3]):  # Show first 3
                        print(f"[DEBUG] msg[{i}] type: {type(msg).__name__}, attrs: {[a for a in dir(msg) if not a.startswith('_')][:10]}")

        print("[Knowledge Manager] Strategy update complete.")
        return result



    async def _run_single_async(
        self,
        prompt: List[Dict[str, str]],
        problem_idx: int,
        max_retries: int = 3,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        mcp_server: Optional[MCPServerSse] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete three-phase workflow for a single problem.

        This function is designed to be called concurrently in a batch.
        It handles its own retries, timeouts, and MCP lock acquisition.

        Args:
            prompt: The OpenAI message list for the problem.
            problem_idx: The unique index/ID for this problem.
            max_retries: The number of times to retry on failure.
            openai_client: Per-task OpenAI client to avoid shared state issues.
                          If None, falls back to self.client.
            mcp_server: Per-task MCP server connection to avoid shared state issues.
                       If None, falls back to self.mcp_server.

        Returns:
            A dictionary containing:
            - "output": A list containing the final solution string.
            - "messages": The list of message objects from the Solver agent.
            - "problem_idx": The index, for re-associating results.
        """
        assert isinstance(prompt, list)

        # Use per-task client if provided, otherwise fall back to shared client
        client = openai_client if openai_client is not None else self.client
        assert client is not None, "OpenAI client not initialized"
        timeout_seconds = self.client_kwargs["timeout"]

        # Extract a short problem description for MCP
        input_text = prompt[-1]["content"] if prompt else ""
        lines = input_text.split('\n')
        problem_description = input_text

        output_solutions: List[str] = []
        rollout_messages: List[List[Any]] = []
        best_ranked_solution: str = ""
        error_type: Optional[str] = None  # Track error type for this problem

        # Create per-task MCP connection for Agent usage
        # This prevents shared connection corruption between concurrent tasks
        # See: https://github.com/openai/openai-agents-python/issues/1288
        task_mcp_server = None
        try:
            task_mcp_server = MCPServerSse(
                name=f"Knowledge Flow Server (Task {problem_idx})",
                params={
                    "url": self.mcp_server_url,
                    "timeout": 60.0,
                    "sse_read_timeout": 600,
                },
                cache_tools_list=True,  # Cache tools to reduce list_tools() calls
                max_retry_attempts=2,   # Retry on transient failures
                retry_backoff_seconds_base=1.0,
            )
            await task_mcp_server.connect()
            print(f"[Task {problem_idx}] Created dedicated MCP connection")
        except Exception as e:
            print(f"[Task {problem_idx}] Failed to create dedicated MCP connection: {e}, using shared")
            task_mcp_server = mcp_server if mcp_server is not None else self.mcp_server

        if task_mcp_server is None:
            return {
                "output": [""],
                "messages": [],
                "problem_idx": problem_idx,
                "error_type": "mcp_connection_failed",
                "critical_failure": True,
            }

        # Retry loop for the entire 3-phase process
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n[RETRY] Attempt {attempt + 1}/{max_retries} for problem {problem_idx}...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            # --- PHASE 0: Fetch knowledge context ---
            print(f"\n{'=' * 80}")
            print(f"[PHASE 0: FETCH KNOWLEDGE] Problem {problem_idx}")
            print(f"{'=' * 80}")

            knowledge_context = ""

            # Use asyncio.timeout for timeout
            # Call MCP tool directly on per-task connection
            async with asyncio.timeout(timeout_seconds):
                tool_result = await self.mcp_server.call_tool(
                    "check_knowledge",
                    {
                        "problem_id": f"problem_{problem_idx}",
                        "problem_description": problem_description
                    }
                )
                knowledge_context = extract_content_from_tool_result(tool_result)

            # Sanitize knowledge context to prevent corrupted code from propagating
            if knowledge_context:
                knowledge_context = sanitize_solution(knowledge_context)

            # Reset roll-out containers for this attempt
            output_solutions = []
            rollout_messages = []
            best_ranked_solution = ""

            # --- PHASE 1: Run solver agent roll-outs ---
            print(f"\n{'=' * 80}")
            print(f"[PHASE 1: SOLVER] Problem {problem_idx}")
            print(f"{'=' * 80}")

            # Split knowledge_context into knowledge_items (lessons) and reference_solution (code)
            # This allows the solver to adaptively choose between its solution and the reference
            reference_solution, knowledge_items = parse_reference_solution_from_knowledge(knowledge_context)
            if reference_solution:
                print(f"[PHASE 1] Extracted reference solution ({len(reference_solution)} chars)")
            if knowledge_items:
                print(f"[PHASE 1] Extracted knowledge items ({len(knowledge_items)} chars)")

            # Parse strategies from knowledge context (only used when enable_strategy is True)
            # Strategies for round 2+ are generated by Phase 2 of the previous round.
            # For round 1 (no prior strategies), we generate initial strategies here.
            strategies = []
            if self.enable_strategy:
                strategies = parse_strategies_from_knowledge(knowledge_context)
                # Check if there's strategy history (indicates this is round 2+)
                has_strategy_history = "## Strategy History" in knowledge_context

                if strategies:
                    print(f"[PHASE 1] Found {len(strategies)} strategies for this round (from Phase 2 of previous round)")
                elif has_strategy_history:
                    # Strategy history exists but no new strategies - Phase 2 may have failed
                    print(f"[PHASE 1] WARNING: Strategy history exists but no strategies for this round")
                    print(f"[PHASE 1] Proceeding without strategies (will solve without strategy guidance)")
                else:
                    # Round 1: No strategy history and no strategies - generate initial strategies
                    try:
                        print("[PHASE 1] Round 1 detected (no strategy history)")
                        print("[PHASE 1] Generating initial strategies for first round...")
                        strategy_result = await self._run_knowledge_manager_strategy_only(
                            problem_idx=problem_idx,
                            problem_description=problem_description,
                            num_strategies=self.roll_out_n,
                            openai_client=client,
                            mcp_server=task_mcp_server,
                        )
                        # Parse strategies from result text
                        # Call MCP tool directly on per-task connection
                        async with asyncio.timeout(timeout_seconds):
                            tool_result_strategy = await self.mcp_server.call_tool(
                                "check_knowledge",
                                {
                                    "problem_id": f"problem_{problem_idx}",
                                    "problem_description": problem_description
                                }
                            )
                            # Extract string content from CallToolResult
                            if hasattr(tool_result_strategy, 'content'):
                                # Content might be a list of content blocks
                                if isinstance(tool_result_strategy.content, list):
                                    knowledge_context_strategy = "\n".join(
                                        str(block.text) if hasattr(block, 'text') else str(block)
                                        for block in tool_result_strategy.content
                                    )
                                else:
                                    knowledge_context_strategy = str(tool_result_strategy.content)
                            else:
                                knowledge_context_strategy = str(tool_result_strategy)
                            strategies = parse_strategies_from_knowledge(knowledge_context_strategy)
                            if strategies:
                                print(f"[PHASE 1] Generated {len(strategies)} initial strategies for round 1")
                    except Exception as e:
                        print(f"[PHASE 1] Initial strategy generation failed: {type(e).__name__}: {e}")
                        print("[PHASE 1] Proceeding without strategies for round 1")


            try:
                # Run all rollouts in parallel within this problem
                # Concurrency is controlled at the problem level by the outer semaphore
                # Add staggered delays to smooth out API bursts

                # Configurable rollout stagger delay (default 2-5 seconds per rollout)
                rollout_stagger_min = getattr(self.args, 'rollout_stagger_min', 2.0)
                rollout_stagger_max = getattr(self.args, 'rollout_stagger_max', 5.0)

                async def run_rollout_with_delay(r_idx: int):
                    """Wrapper that adds a small random delay before starting each rollout."""
                    # Stagger rollouts with configurable delay per rollout index
                    # First rollout (r_idx=0) has minimal delay, subsequent rollouts are staggered
                    if r_idx > 0:
                        delay = r_idx * random.uniform(rollout_stagger_min, rollout_stagger_max)
                        await asyncio.sleep(delay)

                    # Assign strategy to this rollout if available (Round 2+ only)
                    strategy = None
                    if strategies and r_idx < len(strategies):
                        strategy = strategies[r_idx]

                    return await self._run_solver_agent(
                        prompt,
                        problem_idx,
                        problem_description,
                        knowledge_items,
                        reference_solution,
                        rollout_id=r_idx,
                        num_rollouts=self.roll_out_n,
                        strategy=strategy,
                        openai_client=client,
                    )

                # Create rollout tasks
                rollout_tasks = [
                    asyncio.create_task(run_rollout_with_delay(r_idx))
                    for r_idx in range(self.roll_out_n)
                ]

                rollout_results = await asyncio.gather(*rollout_tasks, return_exceptions=True)

                output_solutions = []
                rollout_messages = []
                successful_rollout_indices = []  # Track which rollouts succeeded (for correct id mapping)
                failed_rollouts = 0
                closed_resource_detected = False  # Flag for critical MCP connection failure

                for r_idx, res in enumerate(rollout_results):
                    if isinstance(res, Exception):
                        failed_rollouts += 1
                        error_type = type(res).__name__

                        # Check for ClosedResourceError - indicates MCP server connection died
                        if isinstance(res, ClosedResourceError) or is_closed_resource_error(res):
                            print(f"\n[🚨 CRITICAL] ClosedResourceError detected in rollout {r_idx + 1}/{self.roll_out_n}!")
                            print(f"[🚨 CRITICAL] MCP server connection has been lost. This is an unrecoverable error.")

                            # Log detailed error information for forensic analysis
                            log_critical_error_to_file(
                                error_type="ClosedResourceError",
                                error_message=str(res),
                                problem_idx=problem_idx,
                                phase="Phase1_Solver_Rollout",
                                additional_context={
                                    "rollout_idx": r_idx,
                                    "total_rollouts": self.roll_out_n,
                                    "attempt": attempt,
                                    "max_retries": max_retries,
                                    "successful_rollouts_so_far": len(output_solutions),
                                    "failed_rollouts_so_far": failed_rollouts,
                                },
                                output_dir=getattr(self, 'output_dir', None),
                            )

                            closed_resource_detected = True
                            break  # Stop processing more rollouts

                        # Check for streaming/connection errors specifically
                        elif "ReadError" in error_type or "ReadTimeout" in error_type:
                            print(f"[❌ ROLLOUT ERROR] Rollout {r_idx + 1}/{self.roll_out_n} failed due to streaming error: {error_type}")
                            print(f"[INFO] Connection was interrupted - this is usually a server-side timeout or network issue")
                        else:
                            print(f"[❌ ROLLOUT ERROR] Rollout {r_idx + 1}/{self.roll_out_n} failed: {error_type}: {res}")
                        continue
                    sol, msgs = res
                    # Sanitize solution to remove corrupted characters (null bytes, malformed newlines)
                    sanitized_sol = sanitize_solution(sol) if sol else sol
                    output_solutions.append(sanitized_sol)
                    rollout_messages.append(msgs)
                    successful_rollout_indices.append(r_idx)  # Track original rollout index

                # If ClosedResourceError was detected, break out of the retry loop immediately
                if closed_resource_detected:
                    print(f"\n[🚨 ABORT] Breaking out of retry loop due to ClosedResourceError.")
                    print(f"[🚨 ABORT] MCP server connection is dead. Cannot continue.")
                    print(f"[🚨 ABORT] Check critical_errors log for details.")
                    return {
                        "output": output_solutions if output_solutions else [""],
                        "messages": rollout_messages,
                        "problem_idx": problem_idx,
                        "error_type": "ClosedResourceError",
                        "critical_failure": True,
                    }

                # Log summary of rollout results
                if failed_rollouts > 0:
                    print(f"[SUMMARY] {failed_rollouts}/{self.roll_out_n} rollouts failed, {len(output_solutions)} succeeded")
                    if failed_rollouts == self.roll_out_n:
                        print(f"[WARNING] ALL rollouts failed! Will attempt recovery from knowledge context.")

            except asyncio.TimeoutError:
                print(f"[❌ TIMEOUT ERROR] Phase 1 (Solver) timed out after {timeout_seconds}s for problem {problem_idx}")
                error_type = "TimeoutError"
                if attempt < max_retries - 1:
                    print("[RETRY] Will retry with exponential backoff...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Returning empty response.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": error_type,
                    }
            except httpx.ReadTimeout as e:
                print(f"[❌ HTTP TIMEOUT ERROR] HTTP read timeout in Phase 1 (Solver) for problem {problem_idx}: {e}")
                print("[INFO] This usually means the model took longer than the configured timeout to generate a response")
                error_type = "ReadTimeout"
                if attempt < max_retries - 1:
                    print("[RETRY] Will retry with exponential backoff...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Returning empty response.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": error_type,
                    }
            except httpx.ReadError as e:
                print(f"[❌ HTTP READ ERROR] Connection interrupted in Phase 1 (Solver) for problem {problem_idx}: {e}")
                print("[INFO] The connection was closed unexpectedly (possibly server-side timeout)")
                error_type = "ReadError"
                if attempt < max_retries - 1:
                    print("[RETRY] Will retry with exponential backoff...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Returning empty response.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": error_type,
                    }
            except httpx.HTTPError as e:
                print(f"[❌ HTTP ERROR] HTTP error in Phase 1 (Solver) for problem {problem_idx}: {e}")
                error_type = "HTTPError"
                if attempt < max_retries - 1:
                    print("[RETRY] Will retry with exponential backoff...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Returning empty response.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": error_type,
                    }
            except ClosedResourceError as e:
                # ClosedResourceError is a critical failure - MCP server connection died
                print(f"\n[🚨 CRITICAL] ClosedResourceError caught at top level for problem {problem_idx}")
                log_critical_error_to_file(
                    error_type="ClosedResourceError",
                    error_message=str(e),
                    problem_idx=problem_idx,
                    phase="Phase1_Solver_TopLevel",
                    additional_context={
                        "attempt": attempt,
                        "max_retries": max_retries,
                    },
                    output_dir=getattr(self, 'output_dir', None),
                )
                print(f"[🚨 ABORT] MCP server connection is dead. Cannot continue.")
                return {
                    "output": [""],
                    "messages": [],
                    "problem_idx": problem_idx,
                    "error_type": "ClosedResourceError",
                    "critical_failure": True,
                }
            except Exception as e:
                error_type = type(e).__name__
                # Check if this is a ClosedResourceError wrapped in another exception
                if is_closed_resource_error(e):
                    print(f"\n[🚨 CRITICAL] ClosedResourceError detected in exception for problem {problem_idx}")
                    log_critical_error_to_file(
                        error_type="ClosedResourceError",
                        error_message=str(e),
                        problem_idx=problem_idx,
                        phase="Phase1_Solver_WrappedException",
                        additional_context={
                            "original_exception_type": error_type,
                            "attempt": attempt,
                            "max_retries": max_retries,
                        },
                        output_dir=getattr(self, 'output_dir', None),
                    )
                    print(f"[🚨 ABORT] MCP server connection is dead. Cannot continue.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": "ClosedResourceError",
                        "critical_failure": True,
                    }

                print(f"[❌ UNEXPECTED ERROR] Phase 1 (Solver) failed for problem {problem_idx}: {error_type}: {e}")
                if attempt < max_retries - 1:
                    print("[RETRY] Will retry with exponential backoff...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Returning empty response.")
                    return {
                        "output": [""],
                        "messages": [],
                        "problem_idx": problem_idx,
                        "error_type": error_type,
                    }

            # Check for empty or trivial responses across roll-outs
            if not any(output_solutions) or all(len(sol) < 10 for sol in output_solutions if sol):
                print(f"\n[WARNING] Short/empty responses across roll-outs. Count={len(output_solutions)}")
                if attempt < max_retries - 1:
                    print("[RETRY] Retrying due to short responses...")
                    continue
                else:
                    print("[FAIL] Max retries reached. Using available short responses.")

            # --- PHASE 1.5: Run test generator (if enabled) ---
            test_results = None
            if self.enable_test_gen and output_solutions:
                print(f"\n{'=' * 80}")
                print(f"[PHASE 1.5: TEST GENERATOR] Problem {problem_idx}")
                print(f"{'=' * 80}")

                # Extract raw code from solutions for execution (done once before retry loop)
                solutions_for_testing = [
                    extract_code_for_evaluation(sol) for sol in output_solutions
                ]
                # Also include reference solution if available
                if reference_solution:
                    solutions_for_testing.insert(0, extract_code_for_evaluation(reference_solution))

                # Build correct id_mapping for Phase 1.5
                # Maps solution index in solutions_for_testing to original_id
                # original_id: 0 = reference, 1+ = rollout indices (1-indexed)
                phase15_id_mapping = {}
                if reference_solution:
                    phase15_id_mapping[0] = 0  # Reference at index 0 -> original_id 0
                    for i, rollout_idx in enumerate(successful_rollout_indices):
                        # Rollout at index i+1 (after reference) -> original_id = rollout_idx + 1
                        phase15_id_mapping[i + 1] = rollout_idx + 1
                else:
                    for i, rollout_idx in enumerate(successful_rollout_indices):
                        # Rollout at index i -> original_id = rollout_idx + 1
                        phase15_id_mapping[i] = rollout_idx + 1
                print(f"[PHASE 1.5] ID mapping for test generator: {phase15_id_mapping}")

                try:
                    async with asyncio.timeout(timeout_seconds):
                        test_results = await self._run_test_generator_agent(
                            problem_idx=problem_idx,
                            problem_description=problem_description,
                            solutions=solutions_for_testing,
                            id_mapping=phase15_id_mapping,
                            openai_client=client,
                            mcp_server=task_mcp_server,
                        )

                    if test_results:
                        print(f"[PHASE 1.5] Test generation completed successfully")
                        # Log test failures to disk for debugging
                        await self._log_test_failure(
                            problem_idx=problem_idx,
                            test_results_json=test_results,
                            solutions=solutions_for_testing,
                            problem_description=problem_description,
                            phase="test_generator",
                            additional_context={
                                "num_rollouts": self.roll_out_n,
                                "has_reference": bool(reference_solution),
                            },
                        )
                    else:
                        print(f"[PHASE 1.5] Test generation did not return valid results. Falling back to non-test-based ranking")

                except asyncio.TimeoutError:
                    print(f"[PHASE 1.5] Test generation timed out after {timeout_seconds}s. Falling back to non-test-based ranking")
                    test_results = None
                except Exception as e:
                    print(f"[PHASE 1.5] Test generation failed with {type(e).__name__}: {e}. Falling back to non-test-based ranking")
                    test_results = None

                print(f"{'=' * 80}\n")

            # --- PHASE 2: Run knowledge manager ---
            reference_solution_is_wrong = False  # Default value
            best_ranked_solution = next((s for s in output_solutions if s), "")

            # Build strategy performance mapping (for KM to see which strategies worked)
            # Use successful_rollout_indices to correctly map strategies to their test results
            strategy_performance = []
            if strategies and self.enable_strategy:
                for i, original_rollout_idx in enumerate(successful_rollout_indices):
                    # Get the strategy for this rollout (if it exists)
                    if original_rollout_idx >= len(strategies):
                        continue
                    strat = strategies[original_rollout_idx]

                    perf = {
                        "id": strat.get("id", original_rollout_idx + 1),
                        "key_technique": strat.get("key_technique", "Unknown"),
                        "what_to_try": strat.get("what_to_try", ""),
                        "rollout_id": original_rollout_idx + 1,  # Original rollout number (1-indexed)
                    }

                    # Add test results if available
                    # Test results use solution_id based on position in solutions_for_testing:
                    # - solutions_for_testing[0] = reference (if exists) -> solution_id=1
                    # - solutions_for_testing[1+i] = output_solutions[i] -> solution_id=2+i (if reference exists)
                    # - OR solutions_for_testing[i] = output_solutions[i] -> solution_id=1+i (if no reference)
                    if test_results:
                        try:
                            results_data = json.loads(test_results) if isinstance(test_results, str) else test_results
                            for res in results_data.get("results", []):
                                # i is the index in output_solutions/successful_rollout_indices
                                if reference_solution:
                                    target_solution_id = i + 2  # Skip reference at position 1
                                else:
                                    target_solution_id = i + 1
                                if res.get("solution_id") == target_solution_id:
                                    perf["pass_rate"] = res.get("pass_rate", 0)
                                    perf["passed"] = res.get("passed", 0)
                                    perf["failed"] = res.get("failed", 0)
                                    break
                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"[WARNING] Could not parse test results for strategy performance: {e}")

                    strategy_performance.append(perf)

                if strategy_performance:
                    print(f"[PHASE 2] Built strategy performance for {len(strategy_performance)} strategies (from {len(successful_rollout_indices)} successful rollouts)")

            if output_solutions:
                print(f"\n{'=' * 80}")
                print(f"[PHASE 2: KNOWLEDGE MANAGER] Problem {problem_idx}")
                print(f"{'=' * 80}")

                try:
                    # Use asyncio.timeout for timeout
                    async with asyncio.timeout(timeout_seconds):
                        km_result = await self._run_knowledge_manager_agent(
                            problem_idx, problem_description,
                            knowledge_context, output_solutions,
                            num_rollouts=self.roll_out_n,
                            test_results=test_results,  # Pass test results from Phase 1.5
                            strategy_performance=strategy_performance,  # Pass strategy performance
                            successful_rollout_indices=successful_rollout_indices,  # For correct id mapping
                            openai_client=client,
                            mcp_server=task_mcp_server,
                        )

                    # Extract reference_solution_is_wrong from the captured flag
                    if hasattr(km_result, '_captured_ref_flag'):
                        reference_solution_is_wrong = km_result._captured_ref_flag
                        print(f"[Trace] Using captured reference_solution_is_wrong: {reference_solution_is_wrong}")
                    else:
                        print(f"[Trace] No captured reference_solution_is_wrong flag, defaulting to False")

                    # Use the best-ranked solution from the knowledge manager if available AND valid
                    if hasattr(km_result, "_captured_best_solution") and km_result._captured_best_solution:
                        candidate = km_result._captured_best_solution
                        if is_valid_solution(candidate):
                            best_ranked_solution = candidate
                            print(f"[Trace] Using best-ranked solution from knowledge manager ({len(candidate)} chars).")
                        else:
                            print(f"[WARNING] Knowledge Manager returned invalid solution ({len(candidate) if candidate else 0} chars).")
                            print("[Trace] Falling back to first valid rollout solution.")
                            fallback = next((s for s in output_solutions if is_valid_solution(s)), None)
                            if fallback:
                                best_ranked_solution = fallback
                                print(f"[Trace] Using fallback rollout solution ({len(fallback)} chars).")

                    # Store strategy performance history to MCP
                    # Call MCP tool directly on per-task connection
                    if self.enable_strategy and strategy_performance:
                        try:
                            # Get the selected original_id from km_result
                            selected_original_id = getattr(km_result, '_captured_original_id', 0)
                            await self.mcp_server.call_tool(
                                "store_strategy_performance",
                                {
                                    "problem_id": f"problem_{problem_idx}",
                                    "strategy_performance": json.dumps(strategy_performance),
                                    "selected_original_id": selected_original_id,
                                }
                            )
                            print(f"[PHASE 2] Stored strategy performance history for {len(strategy_performance)} strategies")
                        except Exception as e:
                            print(f"[WARNING] Failed to store strategy performance: {e}")
                except asyncio.TimeoutError:
                    print(f"[❌ TIMEOUT ERROR] Phase 2 (Knowledge Manager) timed out after {timeout_seconds}s for problem {problem_idx}")
                    print("[WARNING] Solution generated but NOT saved to knowledge base. Continuing anyway.")
                except httpx.ReadTimeout as e:
                    print(f"[❌ HTTP TIMEOUT ERROR] HTTP read timeout in Phase 2 (Knowledge Manager) for problem {problem_idx}: {e}")
                    print("[WARNING] Solution generated but NOT saved to knowledge base. Continuing anyway.")
                except httpx.HTTPError as e:
                    print(f"[❌ HTTP ERROR] HTTP error in Phase 2 (Knowledge Manager) for problem {problem_idx}: {e}")
                    print("[WARNING] Solution generated but NOT saved to knowledge base. Continuing anyway.")
                except Exception as e:
                    print(f"[❌ UNEXPECTED ERROR] Phase 2 (Knowledge Manager) failed for problem {problem_idx}: {type(e).__name__}: {e}")
                    print("[WARNING] Solution generated but NOT saved to knowledge base. Continuing anyway.")
                print(f"{'=' * 80}\n")

            # Success! Break out of the retry loop.
            break

        # Final validation and recovery for empty/invalid solutions
        final_solution = best_ranked_solution
        if not is_valid_solution(final_solution):
            print(f"[CRITICAL] No valid solution for problem {problem_idx} after all attempts!")

            # Recovery attempt 1: Try to find any valid solution from rollouts
            for sol in output_solutions:
                if is_valid_solution(sol):
                    final_solution = sol
                    print(f"[RECOVERY] Found valid solution in rollouts ({len(sol)} chars).")
                    break

            # Recovery attempt 2: Try to extract code from knowledge context
            if not is_valid_solution(final_solution) and knowledge_context:
                extracted = extract_code_from_text(knowledge_context)
                if extracted:
                    final_solution = extracted
                    print(f"[RECOVERY] Extracted code from knowledge context ({len(extracted)} chars).")

            # Final check
            if not is_valid_solution(final_solution):
                print(f"[FAIL] Could not recover valid solution for problem {problem_idx}.")

        # Final sanitization before returning
        if final_solution:
            final_solution = sanitize_solution(final_solution)

        # Clean up per-task MCP connection if we created one
        # Use cleanup() method which is designed for safe teardown
        # Only cleanup if we created a dedicated connection (not using shared)
        if task_mcp_server is not None and task_mcp_server is not self.mcp_server:
            try:
                await task_mcp_server.cleanup()
                print(f"[Task {problem_idx}] Closed dedicated MCP connection")
            except Exception as e:
                # Suppress cleanup errors - connection may already be dead
                # This is expected when the connection failed during use
                print(f"[Task {problem_idx}] MCP cleanup (ignored): {type(e).__name__}")

        # This return is for success or final failure
        return {
            "output": [final_solution] if final_solution else [""],
            "rollout_solutions": output_solutions,  # All rollout solutions for per-rollout evaluation
            "messages": rollout_messages,
            "problem_idx": problem_idx,
            "reference_solution_is_wrong": reference_solution_is_wrong,
            "error_type": error_type,  # None if successful, error type string if failed
        }

    async def _asyncio_wrapper_single(self, prompt: List[Dict[str, str]]) -> List[str]:
        """
        An async wrapper for running a single prompt, used by `_run_single`.
        This manages HTTP client and MCP server connection lifecycle for a single run.
        Uses direct per-task MCP connection.
        """
        # Get HTTP client configuration
        custom_timeout, limits, streaming_timeout = self._get_http_client_config()
        print(f"[HTTP Client] Creating with read timeout: {streaming_timeout}s")

        # Use context managers for guaranteed cleanup
        async with httpx.AsyncClient(timeout=custom_timeout, limits=limits) as http_client:
            print(f"[HTTP Client] Created with connection pooling (max={limits.max_connections})")
            self._init_openai_client(http_client)

            # Create direct MCP connection per task
            mcp_context = MCPServerSse(
                name="Knowledge Flow Server (Single Run)",
                params={
                    "url": self.mcp_server_url,
                    "timeout": 60.0,
                    "sse_read_timeout": 600,
                },
            )

            try:
                # Establish connection
                try:
                    await mcp_context.connect()
                    self.mcp_server = mcp_context
                    print(f"Connected to MCP server at {self.mcp_server_url}")
                except httpx.ConnectTimeout as e:
                    print(f"[❌ MCP CONNECTION ERROR] Cannot connect to MCP server at {self.mcp_server_url}")
                    print(f"[ERROR] {e}")
                    print("[INFO] Please ensure the MCP server is running (e.g., via start_server.sh)")
                    raise
                except httpx.HTTPError as e:
                    print(f"[❌ MCP CONNECTION ERROR] HTTP error connecting to MCP server: {e}")
                    raise
                except Exception as e:
                    print(f"[❌ MCP CONNECTION ERROR] Failed to connect to MCP server: {type(e).__name__}: {e}")
                    raise

                # Run the single task
                result = await self._run_single_async(prompt, 0)
                return result["output"]
            finally:
                # Ensure MCP disconnect
                try:
                    print("Disconnecting from MCP server...")
                    await mcp_context.cleanup()
                    self.mcp_server = None
                    print("MCP server disconnected.")
                except Exception as e:
                    print(f"[WARNING] Error during MCP server disconnect: {type(e).__name__}: {e}")
                    self.mcp_server = None
        # HTTP client automatically closed by context manager
        print("[HTTP Client] Closed.")


    def _run_single(self, prompt: List[Dict[str, str]]) -> List[str]:
        """
        Synchronous wrapper for running a single problem.

        This is the main entry point for `lcb_runner` when not in batch mode.
        It spins up a new asyncio event loop just for this one problem.

        Args:
            prompt: The OpenAI message list for the problem.

        Returns:
            A list containing the final solution string.
        """
        # Use asyncio.run to execute the async wrapper
        return asyncio.run(self._asyncio_wrapper_single(prompt))

    async def _run_batch_async_inner(self, prompts: List[List[Dict[str, str]]]) -> Dict[str, List]:
        """
        The core async logic for running a batch of problems using asyncio.

        This function:
        1. Creates HTTP client with connection pooling.
        2. Connects to the MCP server *once*.
        3. Sets up an `asyncio.Semaphore` to limit concurrency.
        4. Uses asyncio tasks to spawn all tasks concurrently.
        5. Waits for all tasks to complete.
        6. Disconnects from the MCP server and HTTP client.

        Args:
            prompts: A list of prompts to run in batch.

        Returns:
            A dictionary in the format expected by `trt.py`:
            - "outputs": List of output lists
            - "messages": List of message history lists
        """
        outputs = [None] * len(prompts)
        all_messages = [None] * len(prompts)
        all_rollout_solutions = [None] * len(prompts)  # Store all rollout solutions for per-rollout evaluation
        results_storage = []

        # Get HTTP client configuration
        custom_timeout, limits, streaming_timeout = self._get_http_client_config()
        print(f"[HTTP Client] Creating with read timeout: {streaming_timeout}s")

        # Use context manager for guaranteed HTTP client cleanup
        async with httpx.AsyncClient(timeout=custom_timeout, limits=limits) as http_client:
            print(f"[HTTP Client] Created with connection pooling (max={limits.max_connections})")
            self._init_openai_client(http_client)

            # Create direct MCP connection for batch
            print(f"[MCP] Attempting to connect to: {self.mcp_server_url}")
            mcp_context = MCPServerSse(
                name="Knowledge Flow Server (Batch)",
                params={
                    "url": self.mcp_server_url,
                    "timeout": 60.0,
                    "sse_read_timeout": 600,
                },
            )
            try:
                # Establish connection
                try:
                    print(f"[MCP] Connecting to MCP server...")
                    await mcp_context.connect()
                    self.mcp_server = mcp_context
                    print(f"[MCP] Successfully connected to MCP server at {self.mcp_server_url}")
                except httpx.ConnectTimeout as e:
                    print(f"[❌ MCP CONNECTION ERROR] Cannot connect to MCP server at {self.mcp_server_url}")
                    print(f"[ERROR] {e}")
                    print("[INFO] Please ensure the MCP server is running (e.g., via start_server.sh)")
                    raise
                except httpcore.ReadError as e:
                    print(f"[❌ MCP CONNECTION ERROR] Read error from MCP server at {self.mcp_server_url}")
                    print(f"[ERROR] {e}")
                    print("[INFO] The MCP server may have closed the connection or is not responding correctly.")
                    print("[INFO] Please check:")
                    print("      1. MCP server is running and healthy (check server logs)")
                    print("      2. The /sse endpoint is accessible")
                    print("      3. No firewall blocking the connection")
                    raise
                except httpx.HTTPError as e:
                    print(f"[❌ MCP CONNECTION ERROR] HTTP error connecting to MCP server: {e}")
                    raise
                except Exception as e:
                    print(f"[❌ MCP CONNECTION ERROR] Failed to connect to MCP server: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    raise

                # Use asyncio.Semaphore for concurrency control
                concurrency = getattr(self.args, 'multiprocess_oai', 10)
                if concurrency <= 0:
                    concurrency = 1
                print(f"Running with concurrency level: {concurrency}")
                semaphore = asyncio.Semaphore(concurrency)

                # Stagger delay for smoother API call distribution
                # This spreads out task launches to prevent bursts
                stagger_delay = getattr(self.args, 'stagger_delay', 0.5)  # seconds between task launches
                post_acquire_delay = getattr(self.args, 'post_acquire_delay', 0.2)  # delay after acquiring semaphore
                print(f"[Rate Limiting] Stagger delay: {stagger_delay}s, Post-acquire delay: {post_acquire_delay}s")

                # tqdm provides the progress bar
                with tqdm(total=len(prompts), desc="Processing prompts (asyncio)", unit="problem") as pbar:

                    # Define the task wrapper that manages concurrency
                    async def _spawn_task(prompt, idx, launch_delay):
                        """Task wrapper to run one problem and manage semaphore."""
                        # Stagger task launches to prevent all tasks from hitting semaphore at once
                        if launch_delay > 0:
                            await asyncio.sleep(launch_delay)

                        task_start_time = datetime.now()
                        print(f"[Task {idx}] Starting execution at {task_start_time.strftime('%H:%M:%S')}")

                        try:
                            # Acquire semaphore before running the task
                            async with semaphore:
                                print(f"[Task {idx}] Acquired semaphore")
                                # Small delay after acquiring semaphore to smooth out API calls
                                # This prevents burst when multiple tasks acquire semaphore simultaneously
                                if post_acquire_delay > 0:
                                    await asyncio.sleep(post_acquire_delay + random.uniform(0, 0.1))
                                result = await self._run_single_async(prompt, idx)
                                if isinstance(result, dict) and result.get("critical_failure"):
                                    raise ClosedResourceError("Critical failure in task: ClosedResourceError detected")
                                results_storage.append(result)
                                print(f"[Task {idx}] Released semaphore")

                            # Print task completion time
                            task_duration = (datetime.now() - task_start_time).total_seconds()
                            print(f"[Task {idx}] ✓ Completed in {task_duration:.1f}s ({task_duration/60:.1f} minutes)")

                        except Exception as e:
                            # Re-raise critical errors to stop the batch
                            if isinstance(e, ClosedResourceError):
                                raise

                            # Catch all other exceptions and create error result
                            # This prevents one failed task from crashing the entire batch
                            error_type = type(e).__name__
                            print(f"[Task {idx}] ✗ Failed with {error_type}: {e}")
                            error_result = {
                                "problem_idx": idx,
                                "output": [""],
                                "messages": [],
                                "error_type": error_type,
                            }
                            results_storage.append(error_result)

                        finally:
                            # Always update the progress bar
                            pbar.update(1)
                            # Force garbage collection to clean up any agent instances
                            # This helps prevent resource exhaustion in long batch runs
                            import gc
                            gc.collect()

                    # Use asyncio.gather to run all tasks concurrently
                    # Tasks are staggered to prevent API bursts
                    print(f"[asyncio] Spawning {len(prompts)} tasks with staggered launches...")
                    # Create explicit Tasks so we can cancel them if needed
                    tasks = [
                        asyncio.create_task(_spawn_task(prompt, idx, idx * stagger_delay))
                        for idx, prompt in enumerate(prompts)
                    ]
                    
                    try:
                        await asyncio.gather(*tasks)
                    except ClosedResourceError:
                        print(f"\n[🚨 CRITICAL] Batch execution aborted due to ClosedResourceError.")
                        print(f"[🚨 CRITICAL] Cancelling {len(tasks)} remaining tasks...")
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        
                        # Wait for tasks to cleanup
                        print(f"[asyncio] Waiting for tasks to cleanup...")
                        try:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        except Exception:
                            pass
                        print(f"[asyncio] Cleanup complete.")
                        
                        # Re-raise to ensure we exit with failure
                        raise

                    # When this completes, all tasks are guaranteed to be complete.
                    print("[asyncio] All tasks completed.")

            finally:
                # Clean up the MCP connection
                try:
                    await mcp_context.cleanup()
                    print("[MCP] MCP connection closed.")
                except Exception as e:
                    print(f"[WARNING] MCP cleanup error: {e}")
                finally:
                    self.mcp_server = None

        # HTTP client automatically closed by context manager
        print("[HTTP Client] Closed.")

        print(f"[Batch] Processing {len(results_storage)} results...")

        # Collect outputs, messages, reference_solution_is_wrong flags, and error types from results
        ref_flags = [False] * len(prompts)
        error_counts: Dict[str, int] = {}  # Aggregate error counts by type
        for result in results_storage:
            idx = result["problem_idx"]
            outputs[idx] = result["output"]
            all_messages[idx] = result["messages"]
            ref_flags[idx] = result.get("reference_solution_is_wrong", False)
            all_rollout_solutions[idx] = result.get("rollout_solutions", [])

            # Track error types
            err_type = result.get("error_type")
            if err_type:
                error_counts[err_type] = error_counts.get(err_type, 0) + 1

        # Fill in any Nones for tasks that might have failed catastrophically
        for i in range(len(outputs)):
            if outputs[i] is None:
                outputs[i] = [""]
            if all_messages[i] is None:
                all_messages[i] = []
            if all_rollout_solutions[i] is None:
                all_rollout_solutions[i] = []

        # Log error summary if any errors occurred
        if error_counts:
            print(f"[Batch] Error summary: {error_counts}")

        # Return format for trt.py
        return {
            "outputs": outputs,
            "messages": all_messages,
            "reference_solution_is_wrong": ref_flags,
            "rollout_solutions": all_rollout_solutions,  # All rollout solutions for per-rollout evaluation
            "error_counts": error_counts,
        }

    def run_batch_async(self, prompts: List[List[Dict[str, str]]]) -> Dict[str, List]:
        """
        Synchronous entry point for running a batch of problems.

        This function is the main entry point for `lcb_runner` when in
        batch mode. It spins up the asyncio event loop to run
        the entire batch.

        Args:
            prompts: A list of prompts to run in batch.

        Returns:
            A dictionary of outputs and messages.
        """
        if self.args.use_cache:
            raise ValueError("Cache is not supported for OAIAgentRunner")

        # asyncio.run() starts the event loop and runs the given async function
        return asyncio.run(self._run_batch_async_inner(prompts))

    def run_batch(self, prompts: List[List[Dict[str, str]]]) -> Dict[str, List]:
        """
        Main batch execution method, called by the framework.
        
        This just an alias for `run_batch_async` which handles
        spinning up the `trio` event loop.

        Args:
            prompts: List of prompt lists (OpenAI message format).

        Returns:
            Dict with:
                - "outputs": List of output strings
                - "messages": List of conversation histories
        """
        return self.run_batch_async(prompts)
