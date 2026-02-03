"""
Helper functions for the OpenAI Agent Runner.

This module contains utility functions for:
- Error logging
- Solution validation and sanitization
- Code extraction from markdown
- Strategy and knowledge parsing
- Tool argument formatting
"""

import os
import re
import json
import unicodedata
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


def log_critical_error_to_file(
    error_type: str,
    error_message: str,
    problem_idx: int,
    phase: str,
    additional_context: Dict[str, Any] = None,
    output_dir: str = None,
):
    """
    Log critical errors to a file for forensic analysis.

    This is called for errors like ClosedResourceError that indicate
    systemic failures that need investigation.
    """
    timestamp = datetime.now().isoformat()

    # Determine log directory
    if output_dir is None:
        output_dir = os.getenv("OUTPUT_DIR", "output")
    log_dir = Path(output_dir) / "critical_errors"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "error_type": error_type,
        "error_message": str(error_message)[:2000],  # Truncate long messages
        "problem_idx": problem_idx,
        "phase": phase,
        "full_traceback": traceback.format_exc(),
        "additional_context": additional_context or {},
        "environment": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "mcp_server_url": os.getenv("MCP_SERVER_URL", "not set"),
        }
    }

    # Write to log file (append mode)
    log_file = log_dir / f"critical_errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"[CRITICAL ERROR LOGGED] Details written to: {log_file}")
    except Exception as e:
        print(f"[WARNING] Failed to write error log: {e}")

    # Also print detailed info to console
    print(f"\n{'=' * 80}")
    print(f"[CRITICAL ERROR] {error_type}")
    print(f"{'=' * 80}")
    print(f"Timestamp: {timestamp}")
    print(f"Problem: {problem_idx}")
    print(f"Phase: {phase}")
    print(f"Message: {str(error_message)[:500]}")
    if additional_context:
        print(f"Context: {json.dumps(additional_context, indent=2)}")
    print(f"{'=' * 80}\n")


def get_timestamp() -> str:
    """Get a formatted timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def is_valid_solution(solution: str, min_length: int = 50) -> bool:
    """
    Validates if a solution string contains actual code.

    Args:
        solution: The solution string to validate.
        min_length: Minimum length for the solution to be considered valid.

    Returns:
        True if the solution appears to be valid Python code.
    """
    if not solution or not solution.strip():
        return False

    trimmed = solution.strip()
    if len(trimmed) < min_length:
        return False

    # Check for Python code indicators
    code_indicators = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "print("]
    has_code_indicator = any(indicator in trimmed for indicator in code_indicators)

    return has_code_indicator


def sanitize_solution(solution: str) -> str:
    """
    Sanitizes a solution string by removing corrupted characters.
    Preserves markdown ```python ``` format if present.

    Uses:
    1. Unicode normalization (NFKC)
    2. Control character removal
    3. Common LLM corruption fixes

    Args:
        solution: The raw solution string from LLM output.

    Returns:
        Sanitized solution string with markdown format preserved.
    """
    if not solution:
        return solution

    # Step 1: Unicode normalization (NFKC converts compatibility characters)
    sanitized = unicodedata.normalize('NFKC', solution)

    # Step 2: Remove null characters and other problematic control chars
    # Keep: \n (10), \r (13), \t (9), space (32) and above
    sanitized = ''.join(
        char for char in sanitized
        if ord(char) >= 32 or char in '\n\r\t'
    )

    # Step 3: Normalize line endings to Unix style
    sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')

    # Step 4: Fix common LLM corruptions
    # - 'nn' at line starts (corrupted newlines)
    sanitized = re.sub(r'\nnn(\s+)', r'\n\1', sanitized)
    if sanitized.startswith('nn'):
        sanitized = sanitized[2:]

    # Step 5: Replace problematic Unicode characters
    replacements = {
        '\u00a0': ' ',      # Non-breaking space -> regular space
        '\u2018': "'",      # Left single quote
        '\u2019': "'",      # Right single quote
        '\u201c': '"',      # Left double quote
        '\u201d': '"',      # Right double quote
        '\u2013': '-',      # En dash
        '\u2014': '-',      # Em dash
        '\u2026': '...',    # Ellipsis
        '\ufeff': '',       # BOM / zero-width no-break space
        '\u200b': '',       # Zero-width space
        '\u200c': '',       # Zero-width non-joiner
        '\u200d': '',       # Zero-width joiner
    }
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)

    # Step 6: Ensure solution is wrapped in ```python ``` if it contains code
    # but doesn't have the markdown format
    if sanitized.strip() and '```python' not in sanitized:
        # Check if it looks like Python code (has code indicators)
        code_indicators = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "return ", "print("]
        has_code = any(indicator in sanitized for indicator in code_indicators)
        if has_code:
            # Wrap in markdown format
            sanitized = f"```python\n{sanitized.strip()}\n```"

    return sanitized


def extract_code_from_text(text: str) -> Optional[str]:
    """
    Extracts Python code block from markdown-formatted text.

    Args:
        text: Text that may contain ```python code blocks.

    Returns:
        Extracted code string or None if no valid code found.
    """
    if not text:
        return None

    # Sanitize the text first to remove corrupted characters
    text = sanitize_solution(text)

    # Try to find Python code block
    code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if is_valid_solution(code):
            return code

    # Try generic code block
    code_match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if is_valid_solution(code):
            return code

    return None


def extract_code_for_evaluation(solution: str) -> str:
    """
    Extracts raw Python code from a solution that may be wrapped in markdown.
    Used when code needs to be executed/evaluated.

    Args:
        solution: Solution string, possibly wrapped in ```python ```

    Returns:
        Raw Python code without markdown fences.
    """
    if not solution:
        return solution

    # Try to extract from ```python block
    code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Try generic code block
    code_match = re.search(r'```\n(.*?)```', solution, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # No markdown, return as-is (might already be raw code)
    return solution.strip()


def format_tool_arguments(arguments: Any) -> str:
    """
    Formats tool call arguments for pretty-printing in logs.

    Truncates large argument lists to keep logs readable.

    Args:
        arguments: The arguments passed to the tool, can be a
                   JSON string or a dict.

    Returns:
        A formatted string representing the arguments.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        args_str = json.dumps(args, indent=2)
    except json.JSONDecodeError:
        args_str = str(arguments)

    lines = args_str.split('\n')

    # Truncate if too long
    if len(lines) > 10 or len(args_str) > 300:
        formatted_lines = [f"│   {line}" for line in lines[:8]]
        formatted_lines.append(f"│   ... ({len(lines) - 8} more lines)")
        return '\n'.join(formatted_lines)
    else:
        return '\n'.join(f"│   {line}" for line in lines)


def extract_output_from_messages(messages: List[Any]) -> str:
    """
    Fallback function to extract the final output text from the
    message history if streaming fails to capture it.

    Args:
        messages: The list of message objects from the agent result.

    Returns:
        The extracted assistant output string, or "" if not found.
    """
    if not messages:
        return ""

    for msg in reversed(messages):
        # Try multiple ways to identify assistant messages
        is_assistant = False
        if hasattr(msg, "role"):
            role = msg.role
            # Handle both string and enum roles
            is_assistant = (role == "assistant" or str(role).lower() == "assistant"
                           or "assistant" in str(role).lower())

        # Also check for message type attribute (some SDK versions)
        if not is_assistant and hasattr(msg, "type"):
            msg_type = str(msg.type).lower()
            is_assistant = "assistant" in msg_type or "output" in msg_type or "response" in msg_type

        if not is_assistant:
            continue

        # Try to extract content from various possible structures
        content = None

        # Structure 1: msg.content as string
        if hasattr(msg, "content"):
            if isinstance(msg.content, str) and msg.content.strip():
                content = msg.content
            elif isinstance(msg.content, list):
                # Structure 2: msg.content as list of content blocks
                for content_item in msg.content:
                    if hasattr(content_item, "text") and content_item.text:
                        content = content_item.text
                        break
                    elif isinstance(content_item, str) and content_item.strip():
                        content = content_item
                        break

        # Structure 3: msg.text directly
        if not content and hasattr(msg, "text") and msg.text:
            content = msg.text

        # Structure 4: msg.output
        if not content and hasattr(msg, "output") and msg.output:
            content = str(msg.output)

        if content:
            print(f"[INFO] Extracted {len(content)} chars from messages")
            return content

    # Don't print warning - final_output fallback usually works
    return ""


def parse_strategies_from_knowledge(knowledge_context: str) -> List[Dict[str, str]]:
    """Parse strategies from MCP check_knowledge response.

    Args:
        knowledge_context: The knowledge context string from check_knowledge.

    Returns:
        List of strategy dicts with 'id', 'key_technique', and 'what_to_try' keys.
    """
    strategies = []
    if not knowledge_context or "## Strategies for This Round" not in knowledge_context:
        return strategies

    # Pattern to match strategy blocks
    pattern = r"### Strategy (\d+)\s*\n\*\*Key Technique\*\*:\s*(.+?)\n\*\*What to Try\*\*:\s*(.+?)(?=\n###|\n---|\Z)"
    for match in re.findall(pattern, knowledge_context, re.DOTALL):
        strat_id, key_technique, what_to_try = match
        strategies.append({
            "id": int(strat_id),
            "key_technique": key_technique.strip(),
            "what_to_try": what_to_try.strip(),
        })

    if strategies:
        print(f"[INFO] Parsed {len(strategies)} strategies from knowledge context")
    return strategies


def format_strategy_performance(strategy_performance: List[Dict]) -> str:
    """Format strategy performance for KM prompt.

    Args:
        strategy_performance: List of strategy performance dicts with keys:
            - rollout_id: The rollout number (1-indexed)
            - key_technique: The strategy name
            - what_to_try: The strategy description
            - pass_rate: Optional test pass rate (0.0-1.0)
            - passed: Optional number of tests passed
            - failed: Optional number of tests failed

    Returns:
        Formatted markdown string showing strategy performance with details.
    """
    if not strategy_performance:
        return ""

    lines = ["## This Round's Strategy Performance"]
    lines.append("")
    lines.append("Below shows what strategy each rollout used and how it performed:")
    lines.append("")

    for perf in strategy_performance:
        rollout = perf.get("rollout_id", "?")
        technique = perf.get("key_technique", "Unknown")
        what_to_try = perf.get("what_to_try", "")
        pass_rate = perf.get("pass_rate")
        passed = perf.get("passed")
        failed = perf.get("failed")

        if pass_rate is not None:
            rate_str = f"{pass_rate:.0%}"
            if pass_rate >= 1.0:
                result = "PASSED ALL"
            elif pass_rate >= 0.8:
                result = "MOSTLY PASSED"
            elif pass_rate >= 0.5:
                result = "PARTIAL"
            else:
                result = "FAILED"
        else:
            rate_str = "N/A"
            result = "Untested"

        lines.append(f"### Rollout {rollout} → Solution {rollout + 1}")
        lines.append(f"- **Strategy**: {technique}")
        if what_to_try:
            lines.append(f"- **Approach**: {what_to_try}")
        lines.append(f"- **Result**: {result} ({rate_str})")
        if passed is not None and failed is not None:
            lines.append(f"- **Tests**: {passed} passed, {failed} failed")
        lines.append("")

    lines.append("**Use this performance data to generate BETTER strategies for the next round.**")
    lines.append("- Strategies that FAILED should be avoided or significantly modified")
    lines.append("- Strategies that PASSED can be refined or used as a base for variations")
    lines.append("")
    return "\n".join(lines)


def parse_strategy_history_from_knowledge(knowledge_context: str) -> str:
    """Parse strategy history section from knowledge context.

    The knowledge context from check_knowledge contains a section like:
    ## Strategy History (Previous Rounds)
    *What was tried before and how it performed:*

    **Round 1:**
    - Dynamic Programming: 100% (SELECTED)
    - Greedy: 40% (failed)

    Returns:
        The strategy history text if found, empty string otherwise.
    """
    if not knowledge_context or "## Strategy History" not in knowledge_context:
        return ""

    # Extract the strategy history section
    start_marker = "## Strategy History (Previous Rounds)"
    if start_marker not in knowledge_context:
        return ""

    start_idx = knowledge_context.index(start_marker)
    # Find the end - next "---" or "## Reference Solution" or end of text
    end_markers = ["---", "## Reference Solution", "## Strategies for This Round"]
    end_idx = len(knowledge_context)
    for marker in end_markers:
        marker_idx = knowledge_context.find(marker, start_idx + len(start_marker))
        if marker_idx != -1 and marker_idx < end_idx:
            end_idx = marker_idx

    history_text = knowledge_context[start_idx:end_idx].strip()
    return history_text


def parse_reference_solution_from_knowledge(knowledge_context: str) -> tuple:
    """
    Parse and extract the reference solution from knowledge context.

    The MCP server returns the reference solution in a clearly marked section:
        ---
        ## Reference Solution
        ```python
        <code>
        ```

    Args:
        knowledge_context: The full knowledge context string from check_knowledge.

    Returns:
        Tuple of (reference_solution_code, knowledge_without_reference).
        Empty string for reference solution if not found.
    """
    if "## Reference Solution" not in knowledge_context:
        return "", knowledge_context

    # Split at the reference solution section
    parts = knowledge_context.split("## Reference Solution", 1)

    # Fix: Use proper suffix removal instead of rstrip("---") which removes individual chars
    knowledge_without_ref = parts[0].rstrip()
    if knowledge_without_ref.endswith("---"):
        knowledge_without_ref = knowledge_without_ref[:-3].rstrip()

    if len(parts) < 2:
        return "", knowledge_without_ref

    ref_section = parts[1]

    # Extract code from ```python ... ``` block
    # Handle both single and double-wrapped markdown (in case content already has fences)
    code_match = re.search(r"```python\n(.*?)```", ref_section, re.DOTALL)
    if code_match:
        reference_code = code_match.group(1).strip()

        # Check if extracted code is itself wrapped in markdown (double-wrap case)
        # This happens when the stored content already has ```python ... ```
        if reference_code.startswith("```python"):
            inner_match = re.search(r"```python\n(.*?)```", reference_code, re.DOTALL)
            if inner_match:
                reference_code = inner_match.group(1).strip()

        if reference_code:
            return reference_code, knowledge_without_ref

    # Fallback: try generic code block
    code_match = re.search(r"```\n(.*?)```", ref_section, re.DOTALL)
    if code_match:
        reference_code = code_match.group(1).strip()
        if reference_code:
            return reference_code, knowledge_without_ref

    return "", knowledge_without_ref


def reorder_test_results(test_results: str, reverse_mapping: Dict[int, int]) -> str:
    """
    Reorder test results to match shuffled solution order.

    Phase 1.5 Test Generator produces results with solution_id matching original order:
    - solution_id 1 = reference (original_id 0)
    - solution_id 2 = rollout1 (original_id 1)
    - etc.

    This method reorders those results to match the shuffled order used in KM.

    Args:
        test_results: JSON string from execute_generated_tests tool
        reverse_mapping: Dict mapping original_id -> shuffled_idx

    Returns:
        Reordered test results as formatted string for KM template
    """
    try:
        results_data = json.loads(test_results)
    except json.JSONDecodeError:
        # If not valid JSON, return as-is
        return test_results

    if "results" not in results_data:
        return test_results

    original_results = results_data["results"]

    # Create mapping from original solution_id (1-indexed) to result
    # solution_id in test results is 1-indexed, original_id is 0-indexed
    result_by_original_id = {}
    for result in original_results:
        original_id = result["solution_id"] - 1  # Convert 1-indexed to 0-indexed
        result_by_original_id[original_id] = result

    # Reorder results according to shuffled order
    reordered_results = []
    for original_id in sorted(reverse_mapping.keys()):
        shuffled_idx = reverse_mapping[original_id]
        if original_id in result_by_original_id:
            result = result_by_original_id[original_id].copy()
            result["solution_id"] = shuffled_idx + 1  # New 1-indexed ID
            reordered_results.append((shuffled_idx, result))

    # Sort by shuffled_idx
    reordered_results.sort(key=lambda x: x[0])
    reordered_results = [r[1] for r in reordered_results]

    # Update best_solution_id if present
    best_id = results_data.get("best_solution_id")
    if best_id is not None:
        best_original_id = best_id - 1  # Convert to 0-indexed
        if best_original_id in reverse_mapping:
            new_best_id = reverse_mapping[best_original_id] + 1  # Convert back to 1-indexed
            results_data["best_solution_id"] = new_best_id

    results_data["results"] = reordered_results

    # Format as readable string for template
    lines = ["```json", json.dumps(results_data, indent=2), "```"]
    return "\n".join(lines)


def extract_fn_name_from_problem(problem_description: str) -> Optional[str]:
    """
    Extract the function name from problem description for call-based execution.

    Looks for patterns like:
    - "def solution(" or "def functionName("
    - "class Solution:" followed by "def methodName("
    - Explicit mention of "Function: functionName"

    Args:
        problem_description: The problem description text.

    Returns:
        The function name if found, None otherwise.
    """
    if not problem_description:
        return None

    # Pattern 1: Look for explicit function definition in starter code
    # def solution(nums: List[int]) -> int:
    fn_pattern = re.search(r'def\s+(\w+)\s*\(', problem_description)
    if fn_pattern:
        fn_name = fn_pattern.group(1)
        # Skip common non-entry-point names
        if fn_name not in ('__init__', '__str__', '__repr__', 'helper', 'dfs', 'bfs'):
            return fn_name

    # Pattern 2: Look for class Solution with method
    class_pattern = re.search(
        r'class\s+Solution.*?def\s+(\w+)\s*\(',
        problem_description,
        re.DOTALL
    )
    if class_pattern:
        fn_name = class_pattern.group(1)
        if fn_name != '__init__':
            return fn_name

    # Pattern 3: Look for explicit function name mention
    # "Function: twoSum" or "function name: twoSum"
    explicit_pattern = re.search(
        r'[Ff]unction(?:\s+name)?:\s*(\w+)',
        problem_description
    )
    if explicit_pattern:
        return explicit_pattern.group(1)

    # Default: assume stdin-based if no function name found
    return None


def extract_tool_call_arguments(result: Any, tool_name: str) -> Optional[Dict]:
    """
    Extract arguments from a specific tool call in agent result.

    This is a helper function to consolidate the logic for extracting
    tool call arguments from various SDK result structures.

    Args:
        result: The agent result object.
        tool_name: The name of the tool to find arguments for.

    Returns:
        The arguments dict if found, None otherwise.
    """
    def _parse_args(args):
        """Parse arguments from string or dict."""
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return None
        return args if isinstance(args, dict) else None

    # Method 1: Check result.messages for tool_calls
    if hasattr(result, 'messages'):
        for msg in result.messages:
            # Structure 1: msg.tool_calls[].function.name/arguments
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tc_name = None
                    tc_args = None

                    if hasattr(tool_call, 'function'):
                        tc_name = getattr(tool_call.function, 'name', None)
                        tc_args = getattr(tool_call.function, 'arguments', None)
                    elif hasattr(tool_call, 'name'):
                        tc_name = tool_call.name
                        tc_args = getattr(tool_call, 'arguments', None)

                    if tc_name == tool_name and tc_args:
                        parsed = _parse_args(tc_args)
                        if parsed:
                            return parsed

            # Structure 2: msg.raw_item
            if hasattr(msg, 'raw_item'):
                raw_item = msg.raw_item
                if hasattr(raw_item, 'name') and raw_item.name == tool_name:
                    if hasattr(raw_item, 'arguments'):
                        parsed = _parse_args(raw_item.arguments)
                        if parsed:
                            return parsed

            # Structure 3: msg itself for tool call properties
            if hasattr(msg, 'type') and msg.type == "tool_call_item":
                if hasattr(msg, 'name') and msg.name == tool_name:
                    if hasattr(msg, 'arguments'):
                        parsed = _parse_args(msg.arguments)
                        if parsed:
                            return parsed

    # Method 2: Check result.new_items
    if hasattr(result, 'new_items'):
        for item in result.new_items:
            if hasattr(item, 'type') and item.type == "tool_call_item":
                if hasattr(item, 'name') and item.name == tool_name:
                    if hasattr(item, 'arguments'):
                        parsed = _parse_args(item.arguments)
                        if parsed:
                            return parsed

            if hasattr(item, 'raw_item'):
                raw_item = item.raw_item
                if hasattr(raw_item, 'name') and raw_item.name == tool_name:
                    if hasattr(raw_item, 'arguments'):
                        parsed = _parse_args(raw_item.arguments)
                        if parsed:
                            return parsed

    return None


def handle_raw_response_event(
    event: Any,
    output_text: str,
    in_output_mode: bool,
    response_count: int
) -> tuple:
    """
    Handles raw token-by-token streaming events from the agent.

    Args:
        event: The event object from the stream.
        output_text: The cumulative output text so far.
        in_output_mode: A flag to track if we are currently in an output block.
        response_count: The number of responses seen so far.

    Returns:
        A tuple of (updated_output_text, updated_in_output_mode, updated_response_count).
    """
    if not (hasattr(event, 'data') and hasattr(event.data, 'type')):
        return output_text, in_output_mode, response_count

    event_type = event.data.type

    if event_type == "response.output_text.delta":
        # This is a token delta for the final output
        if hasattr(event.data, 'delta') and event.data.delta:
            if not in_output_mode:
                print("\n┌─ [📝 Output Stream] ─────────────────────────────")
                in_output_mode = True
            print(event.data.delta, end='', flush=True)
            output_text += event.data.delta

    elif event_type == "response.created":
        # A new agent response (thought/tool call/output) is starting
        response_count += 1
        print(f"┌─ [🤖 Response #{response_count}] Generation started │ {get_timestamp()}")

    elif event_type == "response.done":
        # The agent response is complete
        if in_output_mode:
            print("\n└─ [✓ Output Complete] ────────────────────────────")
            in_output_mode = False
        else:
            print(f"└─ [✓ Response Complete] │ {get_timestamp()}")

    return output_text, in_output_mode, response_count


def handle_run_item_event(
    item: Any,
    tool_calls_count: int,
    tool_call_history: List[str],
    check_knowledge_calls: int
) -> tuple:
    """
    Handles functional run item events (like tool calls).

    Args:
        item: The run item from the stream event.
        tool_calls_count: Cumulative tool calls.
        tool_call_history: List of tool names called (modified in-place).
        check_knowledge_calls: Cumulative check_knowledge calls.

    Returns:
        A tuple of (updated_tool_calls_count, updated_check_knowledge_calls).
    """
    if not hasattr(item, 'type'):
        return tool_calls_count, check_knowledge_calls

    # Tool call started
    if item.type == "tool_call_item":
        tool_calls_count += 1
        if hasattr(item, 'name'):
            tool_name = item.name
            tool_call_history.append(tool_name)

            # This check is specific to the Knowledge Flow architecture
            if tool_name == "check_knowledge":
                check_knowledge_calls += 1

            print(f"\n┌─ [🔧 Tool Call #{tool_calls_count}] {tool_name} │ {get_timestamp()}")
            if hasattr(item, 'arguments'):
                print(format_tool_arguments(item.arguments))

    # Tool call output received
    elif item.type == "tool_call_output_item":
        if hasattr(item, 'output'):
            output_str = str(item.output)
            lines = output_str.split('\n')
            print("└─ [✓ Tool Result] ")
            # Truncate long tool outputs
            for line in lines[:5]:
                if len(line) > 100:
                    line = line[:100] + "..."
                print(f"   {line}")
            if len(lines) > 5:
                print(f"   ... ({len(lines) - 5} more lines)")

    # Message output completed
    elif item.type == "message_output_item":
        print(f"\n└─ [✓ Message Generated] │ {get_timestamp()}")

    return tool_calls_count, check_knowledge_calls


async def log_test_failure_to_disk(
    log_dir: Path,
    problem_idx: int,
    test_results_json: str,
    solutions: List[str],
    problem_description: str,
    phase: str = "test_generator",
    additional_context: Optional[Dict] = None,
) -> None:
    """
    Log test failure details to disk for debugging.

    Parses test results from MCP server and logs failures with full context.

    Args:
        log_dir: Directory to write log files to
        problem_idx: Problem index
        test_results_json: JSON string of test results from MCP server
        solutions: List of solution code strings that were tested
        problem_description: The problem description
        phase: Which phase generated this (test_generator, knowledge_manager)
        additional_context: Any additional context to include
    """
    try:
        results_data = json.loads(test_results_json)
    except json.JSONDecodeError:
        # If we can't parse, log the raw string
        results_data = {"raw_response": test_results_json}

    timestamp = datetime.utcnow().isoformat()
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")

    # Check if there are any failures
    results = results_data.get("results", [])
    has_failures = any(r.get("pass_rate", 1.0) < 1.0 for r in results)

    if not has_failures:
        return  # No failures to log

    # Build log entry
    log_entry = {
        "timestamp": timestamp,
        "problem_idx": problem_idx,
        "phase": phase,
        "test_count": results_data.get("test_count", 0),
        "best_solution_id": results_data.get("best_solution_id"),
        "summary": results_data.get("summary", ""),
        "results": [],
        "solutions": {},
        "problem_description_preview": problem_description[:500] + "..." if len(problem_description) > 500 else problem_description,
        "additional_context": additional_context,
    }

    # Process each solution's results
    for result in results:
        sol_id = result.get("solution_id", 0)
        sol_idx = sol_id - 1  # Convert to 0-indexed

        result_entry = {
            "solution_id": sol_id,
            "passed": result.get("passed", 0),
            "failed": result.get("failed", 0),
            "total": result.get("total", 0),
            "pass_rate": result.get("pass_rate", 0),
            "errors": result.get("errors", []),
        }

        # Determine failure type
        errors = result.get("errors", [])
        failure_types = set()
        for err in errors:
            err_lower = err.lower()
            if "wrong answer" in err_lower:
                failure_types.add("wrong_answer")
            elif "time limit" in err_lower:
                failure_types.add("timeout")
            elif "runtime error" in err_lower:
                failure_types.add("runtime_error")
            elif "compilation" in err_lower:
                failure_types.add("compilation")
            elif "empty" in err_lower:
                failure_types.add("empty_solution")

        result_entry["failure_types"] = list(failure_types)
        log_entry["results"].append(result_entry)

        # Include solution code if available
        if 0 <= sol_idx < len(solutions):
            # Truncate very long solutions
            sol_code = solutions[sol_idx]
            if len(sol_code) > 5000:
                sol_code = sol_code[:5000] + "\n... (truncated)"
            log_entry["solutions"][f"solution_{sol_id}"] = sol_code

    # Generate log filename
    log_filename = f"problem_{problem_idx}__{phase}__{safe_timestamp}.json"
    log_path = log_dir / log_filename

    # Write to disk
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        print(f"[Runner] Test failure logged: {log_filename}")
    except Exception as e:
        print(f"[Runner] Warning: Failed to log test failure: {e}")


def extract_update_knowledge_args_from_result(result: Any) -> tuple:
    """
    Extract reference_solution_is_wrong and selected_solution_id from agent result.

    This helper consolidates the logic for extracting tool call arguments from
    various SDK result structures when using the update_knowledge tool.

    Args:
        result: The agent result object.

    Returns:
        Tuple of (reference_solution_is_wrong, selected_solution_id).
        Either value can be None if not found.
    """
    captured_ref_flag = None
    captured_solution_id = None

    def _parse_args(args):
        """Parse arguments from string or dict."""
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return None
        return args if isinstance(args, dict) else None

    def _extract_from_tool_args(tool_args):
        """Extract values from tool arguments dict."""
        nonlocal captured_ref_flag, captured_solution_id

        parsed = _parse_args(tool_args)
        if parsed is None:
            return False

        if "reference_solution_is_wrong" in parsed and captured_ref_flag is None:
            captured_ref_flag = parsed["reference_solution_is_wrong"]
            print(f"[Knowledge Manager] Captured reference_solution_is_wrong: {captured_ref_flag}")
        if "selected_solution_id" in parsed and captured_solution_id is None:
            captured_solution_id = parsed["selected_solution_id"]
            print(f"[Knowledge Manager] Captured selected_solution_id: {captured_solution_id}")
        return True

    # Method 1: Check result.messages for tool_calls
    if hasattr(result, 'messages'):
        for msg in result.messages:
            # Structure 1: msg.tool_calls[].function.name/arguments
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tc_name = None
                    tc_args = None

                    if hasattr(tool_call, 'function'):
                        tc_name = getattr(tool_call.function, 'name', None)
                        tc_args = getattr(tool_call.function, 'arguments', None)
                    elif hasattr(tool_call, 'name'):
                        tc_name = tool_call.name
                        tc_args = getattr(tool_call, 'arguments', None)

                    if tc_name == "update_knowledge" and tc_args:
                        _extract_from_tool_args(tc_args)

            # Structure 2: msg.raw_item (similar to streaming)
            if hasattr(msg, 'raw_item'):
                raw_item = msg.raw_item
                if hasattr(raw_item, 'name') and raw_item.name == "update_knowledge":
                    if hasattr(raw_item, 'arguments'):
                        _extract_from_tool_args(raw_item.arguments)

            # Structure 3: Check msg itself for tool call properties
            if hasattr(msg, 'type') and msg.type == "tool_call_item":
                if hasattr(msg, 'name') and msg.name == "update_knowledge":
                    if hasattr(msg, 'arguments'):
                        _extract_from_tool_args(msg.arguments)

    # Method 2: Check result.new_items (some SDK versions use this)
    if hasattr(result, 'new_items') and captured_solution_id is None:
        for item in result.new_items:
            if hasattr(item, 'type') and item.type == "tool_call_item":
                if hasattr(item, 'name') and item.name == "update_knowledge":
                    if hasattr(item, 'arguments'):
                        _extract_from_tool_args(item.arguments)

            if hasattr(item, 'raw_item'):
                raw_item = item.raw_item
                if hasattr(raw_item, 'name') and raw_item.name == "update_knowledge":
                    if hasattr(raw_item, 'arguments'):
                        _extract_from_tool_args(raw_item.arguments)

    return captured_ref_flag, captured_solution_id


def extract_content_from_tool_result(tool_result: Any) -> str:
    """
    Extract string content from MCP CallToolResult.

    This helper consolidates the logic for extracting string content from
    MCP tool results, which may have various structures.

    Args:
        tool_result: The tool result from MCP call_tool.

    Returns:
        Extracted content as a string.
    """
    if hasattr(tool_result, 'content'):
        # Content might be a list of content blocks
        if isinstance(tool_result.content, list):
            return "\n".join(
                str(block.text) if hasattr(block, 'text') else str(block)
                for block in tool_result.content
            )
        else:
            return str(tool_result.content)
    else:
        return str(tool_result)


def is_closed_resource_error(exception: Exception) -> bool:
    """
    Check if an exception is a ClosedResourceError or wraps one.

    This is used to detect when the MCP server connection has died,
    which requires special handling (abort rather than retry).

    Args:
        exception: The exception to check.

    Returns:
        True if this is a ClosedResourceError or wraps one.
    """
    error_type = type(exception).__name__
    error_str = str(exception)
    return "ClosedResourceError" in error_type or "ClosedResourceError" in error_str
