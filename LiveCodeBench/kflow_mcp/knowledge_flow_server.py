"""
Knowledge Flow MCP Server - Solution Evolution Tracking

Workflow:
1. check_knowledge() → Inspect prior attempts
2. Solve the problem
3. update_knowledge() → Persist the latest solution and reflection

Design Notes:
- One JSON file per problem id (problem_{id}.json)
- Only the most recent solution is stored
- Knowledge entries carry an active status flag
- File writes are protected by per-problem async locks
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
import re
import asyncio
import aiofiles
import aiofiles.os
import json
import sys
import traceback

# Use orjson for fast, safe JSON serialization with strict validation
# Falls back to stdlib json if orjson is not available
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    print("[MCP Server] Warning: orjson not available, using stdlib json (slower, less strict)")

# Add LiveCodeBench to path for test execution utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from lcb_runner.evaluation.testing_util import (
        grade_call_based, grade_stdio, compile_code, get_function,
        import_string, clean_if_name, make_function, call_method,
        Capturing, truncatefn
    )
    import time
    import multiprocessing
    from multiprocessing import Process, Queue
    TEST_EXECUTION_AVAILABLE = True
    print("[MCP Server] Test execution utilities loaded successfully")
except ImportError as e:
    TEST_EXECUTION_AVAILABLE = False
    print(f"[MCP Server] Warning: Test execution utilities not available: {e}")

# Test result formatting configuration
TEST_RESULT_CONFIG = {
    "max_input_chars": 1000,           # Was 200
    "max_expected_chars": 1000,        # Was 200
    "max_actual_chars": 1000,          # Was 200
    "max_detailed_failed_tests": 10,   # Was 5
    "max_total_failed_tests": 30,      # Was 5
    "max_error_messages": 20,          # Was 10
    "format_as_markdown": True,
}


def _smart_truncate(s: str, max_chars: int) -> str:
    """Truncate string intelligently, preserving structure."""
    if max_chars <= 0:
        return s
    s = str(s)
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return f"{s[:half]}...({len(s) - max_chars} chars omitted)...{s[-half:]}"


def _format_test_results_for_km(results: List[Dict[str, Any]], test_count: int) -> str:
    """
    Format test results as readable markdown for the Knowledge Manager.

    Args:
        results: List of solution results from _evaluate_candidate_solution()
        test_count: Total number of test cases

    Returns:
        Formatted markdown string with per-solution results
    """
    config = TEST_RESULT_CONFIG
    lines = []

    for result in results:
        solution_id = result.get("solution_id", "?")
        passed = result.get("passed", 0)
        total = result.get("total", test_count)
        pass_rate = result.get("pass_rate", 0.0)

        # Solution header
        status = "PASS" if passed == total else "FAIL"
        lines.append(f"### Solution {solution_id}: {status} ({passed}/{total} tests passed, {pass_rate:.1%})")
        lines.append("")

        if passed == total:
            lines.append("All tests passed.")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        # Group failures by type
        failed_tests = result.get("failed_tests", [])
        if not failed_tests:
            # Only errors available, no detailed test info
            errors = result.get("errors", [])
            if errors:
                lines.append("**Errors:**")
                for err in errors[:config["max_error_messages"]]:
                    lines.append(f"- {err}")
                lines.append("")
            lines.append("---")
            lines.append("")
            continue

        # Group by error type
        by_type = {}
        for ft in failed_tests[:config["max_detailed_failed_tests"]]:
            error_type = ft.get("error", "Unknown Error")
            if error_type not in by_type:
                by_type[error_type] = []
            by_type[error_type].append(ft)

        for error_type, tests in by_type.items():
            lines.append(f"**{error_type}** ({len(tests)} test{'s' if len(tests) > 1 else ''}):")
            lines.append("")

            for ft in tests:
                test_num = ft.get("test_num", "?")
                lines.append(f"- **Test {test_num}**:")
                lines.append(f"  - Input: `{ft.get('input', 'N/A')}`")
                lines.append(f"  - Expected: `{ft.get('expected', 'N/A')}`")
                if error_type != "Time Limit Exceeded":
                    lines.append(f"  - Actual: `{ft.get('actual', 'N/A')}`")
                lines.append("")

        # Note if there are more failures not shown
        total_failures = result.get("failed", 0)
        shown_failures = len(failed_tests)
        if total_failures > shown_failures:
            lines.append(f"*... and {total_failures - shown_failures} more failed tests not shown*")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def reset_test_executor() -> None:
    """
    Legacy function kept for API compatibility.

    With the new multiprocessing.Process approach, each test runs in its own
    process that is explicitly terminated on timeout. No cleanup is needed
    as there's no persistent pool.
    """
    # No-op: Each test now runs in its own Process that is killed on timeout
    pass


# ============================================================================
# All-Tests Grading Functions (No Early Exit)
# ============================================================================

class TimeoutException(Exception):
    """Exception raised when a test case times out."""
    pass


def _process_wrapper(result_queue: Queue, func, args):
    """
    Wrapper function that runs in subprocess and puts result in queue.

    This runs in a separate process and communicates results via Queue.
    The parent process can terminate() or kill() this process on timeout.
    """
    try:
        result = func(*args)
        result_queue.put(("success", result))
    except Exception as e:
        # Serialize exception info since some exceptions can't be pickled
        result_queue.put(("error", (type(e).__name__, str(e))))


def _run_with_timeout(func, args, timeout_seconds):
    """
    Run a function with a hard timeout using multiprocessing.Process.
    Returns (success, result_or_exception).

    This approach actually terminates runaway processes on timeout:
    1. Creates a new Process for each execution
    2. Uses terminate() (SIGTERM) then kill() (SIGKILL) on timeout
    3. Guarantees no zombie processes accumulate
    4. Each execution is fully isolated
    """
    result_queue = multiprocessing.Queue()

    p = Process(target=_process_wrapper, args=(result_queue, func, args))
    p.start()
    p.join(timeout=timeout_seconds)

    if p.is_alive():
        # Process timed out - terminate it forcefully
        p.terminate()  # SIGTERM
        p.join(timeout=1)  # Give it 1 second to clean up

        if p.is_alive():
            # Still alive after SIGTERM - use SIGKILL
            p.kill()  # SIGKILL - guaranteed to stop
            p.join()

        # Clean up the queue
        try:
            while not result_queue.empty():
                result_queue.get_nowait()
        except Exception:
            pass

        return False, TimeoutException(f"Execution exceeded {timeout_seconds}s timeout")

    # Process completed within timeout
    if result_queue.empty():
        return False, RuntimeError("Process died unexpectedly without returning a result")

    try:
        status, result = result_queue.get_nowait()
    except Exception as e:
        return False, RuntimeError(f"Failed to retrieve result from subprocess: {e}")

    if status == "success":
        return True, result
    else:
        # result is (exception_type, exception_message)
        exc_type, exc_msg = result
        return False, RuntimeError(f"{exc_type}: {exc_msg}")


def _run_single_call_test(method, gt_inp: list, gt_out, timeout: int) -> tuple:
    """Run a single call-based test case and return (result, metadata)."""
    start = time.time()

    success, result = _run_with_timeout(method, gt_inp, timeout)
    elapsed = time.time() - start

    if not success:
        # result is an exception
        e = result
        if isinstance(e, TimeoutException):
            return -3, {
                "error": repr(e),
                "error_code": -3,
                "error_message": "Time Limit Exceeded",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }
        else:
            return -4, {
                "error": repr(e),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }

    prediction = result

    # Convert tuples to lists for comparison
    if isinstance(prediction, tuple):
        prediction = list(prediction)

    if prediction == gt_out:
        return True, {"execution_time": elapsed}
    else:
        return -2, {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
            "error_message": "Wrong Answer",
        }


def grade_call_based_all(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    """
    Run ALL call-based test cases without early exit on failure.
    Returns (all_results, metadata_list) where metadata_list contains info for each test.
    """
    code = import_string + "\n\n" + code
    try:
        compiled_sol = compile_code(code, timeout)
    except Exception as e:
        return None, {"error": f"Compilation failed: {e}"}

    if compiled_sol is None:
        return None, {"error": "Compilation returned None"}

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return None, {"error": f"Function '{fn_name}' not found"}

    # Parse inputs/outputs
    try:
        parsed_inputs = [
            [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
        ]
        parsed_outputs = [json.loads(output) for output in all_outputs]
    except Exception as e:
        return None, {"error": f"Failed to parse inputs/outputs: {e}"}

    all_results = []
    all_metadata = []
    total_execution = 0

    for idx, (gt_inp, gt_out) in enumerate(zip(parsed_inputs, parsed_outputs)):
        result, metadata = _run_single_call_test(method, gt_inp, gt_out, timeout)
        all_results.append(result)
        all_metadata.append(metadata)
        if result is True and "execution_time" in metadata:
            total_execution += metadata["execution_time"]

    return all_results, {"execution_time": total_execution, "test_metadata": all_metadata}


def _stdio_capture_wrapper(method, gt_inp: str):
    """
    Execute a stdio-based method with input capture.

    This is a top-level function (not a closure) so it can be pickled for multiprocessing.
    """
    with Capturing() as captured_output:
        call_method(method, gt_inp)
    return captured_output[0] if captured_output else ""


def _run_single_stdio_test(method, gt_inp: str, gt_out: str, timeout: int) -> tuple:
    """Run a single stdio-based test case and return (result, metadata)."""
    from lcb_runner.evaluation.testing_util import get_stripped_lines, convert_line_to_decimals

    start = time.time()

    # Use multiprocessing.Process with hard timeout via terminate()/kill()
    success, result = _run_with_timeout(_stdio_capture_wrapper, (method, gt_inp), timeout)
    elapsed = time.time() - start

    if not success:
        # result is an exception
        e = result
        if isinstance(e, TimeoutException):
            return -3, {
                "error": str(e),
                "error_code": -3,
                "error_message": "Time Limit Exceeded",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }
        else:
            return -4, {
                "error": repr(e),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }

    prediction = result or ""
    stripped_prediction_lines = get_stripped_lines(prediction)
    stripped_gt_out_lines = get_stripped_lines(gt_out)

    WA_metadata = {
        "output": truncatefn(prediction),
        "inputs": truncatefn(gt_inp),
        "expected": truncatefn(gt_out),
        "error_code": -2,
    }

    if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
        WA_metadata["error_message"] = "Wrong answer: mismatched output length"
        return -2, WA_metadata

    for output_line_idx, (pred_line, gt_line) in enumerate(
        zip(stripped_prediction_lines, stripped_gt_out_lines)
    ):
        # Exact match
        if pred_line == gt_line:
            continue

        # Try decimal comparison
        success_pred, decimal_pred = convert_line_to_decimals(pred_line)
        success_gt, decimal_gt = convert_line_to_decimals(gt_line)

        if success_pred and success_gt and decimal_pred == decimal_gt:
            continue

        WA_metadata["error_message"] = f"Wrong answer at line {output_line_idx}: {truncatefn(pred_line)} != {truncatefn(gt_line)}"
        return -2, WA_metadata

    return True, {"execution_time": elapsed}


def grade_stdio_all(
    code: str, all_inputs: list, all_outputs: list, timeout: int
):
    """
    Run ALL stdio-based test cases without early exit on failure.
    Returns (all_results, metadata_list) where metadata_list contains info for each test.
    """
    code = clean_if_name(code)
    code = make_function(code)

    try:
        compiled_sol = compile_code(code, timeout)
    except Exception as e:
        return None, {"error": f"Compilation failed: {e}"}

    if compiled_sol is None:
        return None, {"error": "Compilation returned None"}

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return None, {"error": "wrapped_function not found after code transformation"}

    all_results = []
    all_metadata = []
    total_execution = 0

    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        result, metadata = _run_single_stdio_test(method, gt_inp, gt_out, timeout)
        all_results.append(result)
        all_metadata.append(metadata)
        if result is True and "execution_time" in metadata:
            total_execution += metadata["execution_time"]

    return all_results, {"execution_time": total_execution, "test_metadata": all_metadata}


def _detect_solution_mode(solution: str, fn_name_hint: Optional[str] = None) -> tuple:
    """
    Detect if a solution is call-based or stdin-based based on code analysis.

    Different rollouts may generate different solution formats (class-based vs stdin-based).
    This function analyzes the solution code to determine the appropriate execution mode.

    Args:
        solution: The solution code to analyze.
        fn_name_hint: Optional function name hint from problem description.

    Returns:
        Tuple of (is_call_based: bool, detected_fn_name: Optional[str])
        - is_call_based=True means use grade_call_based_all with detected_fn_name
        - is_call_based=False means use grade_stdio_all
    """
    if not solution or not solution.strip():
        return (False, None)

    # Check for stdin-based indicators first (more specific)
    has_stdin = 'sys.stdin' in solution or 'input()' in solution
    has_main = 'def main' in solution or 'if __name__' in solution

    # If solution reads from stdin and has main-like structure, it's stdin-based
    if has_stdin and has_main:
        return (False, None)

    # Check for class Solution pattern (common in LeetCode-style problems)
    class_match = re.search(r'class\s+Solution.*?def\s+(\w+)\s*\(', solution, re.DOTALL)
    if class_match:
        fn_name = class_match.group(1)
        if fn_name not in ('__init__', '__str__', '__repr__'):
            return (True, fn_name)

    # Check for function definition matching the hint
    if fn_name_hint:
        fn_pattern = rf'def\s+{re.escape(fn_name_hint)}\s*\('
        if re.search(fn_pattern, solution):
            return (True, fn_name_hint)

    # Check for any standalone function definition (not in a class)
    # Look for function definitions at the start of a line (not indented inside a class)
    standalone_fn = re.search(r'^def\s+(\w+)\s*\(', solution, re.MULTILINE)
    if standalone_fn:
        fn_name = standalone_fn.group(1)
        # Skip common non-entry-point names
        if fn_name not in ('main', 'helper', 'dfs', 'bfs', '__init__', 'solve', 'wrapped_function'):
            # If there's no stdin reading, assume it's call-based
            if not has_stdin:
                return (True, fn_name)

    # Default: if has stdin indicators, treat as stdin-based; otherwise try call-based with hint
    if has_stdin:
        return (False, None)

    # If we have a hint and no strong indicators either way, use the hint
    if fn_name_hint:
        return (True, fn_name_hint)

    # Default to stdin-based if uncertain (safer fallback)
    return (False, None)


async def _grade_solution_async(
    solution_id: int,
    solution: str,
    all_inputs: list,
    all_outputs: list,
    fn_name: Optional[str],
    timeout: int,
    test_count: int,
) -> dict:
    """Grade a single solution asynchronously (runs in thread pool for signal handling)."""
    if not solution or not solution.strip():
        return {
            "solution_id": solution_id,
            "passed": 0,
            "failed": test_count,
            "total": test_count,
            "tests_run": 0,
            "pass_rate": 0.0,
            "errors": ["Empty solution"]
        }

    loop = asyncio.get_event_loop()

    def _do_grading():
        if fn_name:
            return grade_call_based_all(
                code=solution,
                all_inputs=all_inputs,
                all_outputs=all_outputs,
                fn_name=fn_name,
                timeout=timeout
            )
        else:
            return grade_stdio_all(
                code=solution,
                all_inputs=all_inputs,
                all_outputs=all_outputs,
                timeout=timeout
            )

    try:
        # Run grading in process pool for clean timeout cancellation
        result = await loop.run_in_executor(None, _do_grading)

        if result is None or result[0] is None:
            error_msg = result[1].get("error", "Unknown error") if result else "Compilation failed"
            return {
                "solution_id": solution_id,
                "passed": 0,
                "failed": test_count,
                "total": test_count,
                "tests_run": 0,
                "pass_rate": 0.0,
                "errors": [error_msg]
            }

        test_results, metadata = result
        passed = sum(1 for r in test_results if r is True)
        tests_run = len(test_results)
        failed = test_count - passed

        errors = []
        for i, r in enumerate(test_results):
            if r is not True:
                error_desc = {
                    -2: "Wrong Answer",
                    -3: "Time Limit Exceeded",
                    -4: "Runtime Error"
                }.get(r, f"Error code {r}")
                errors.append(f"Test {i+1}: {error_desc}")

        pass_rate = round(passed / test_count, 3) if test_count > 0 else 0

        return {
            "solution_id": solution_id,
            "passed": passed,
            "failed": failed,
            "total": test_count,
            "tests_run": tests_run,
            "pass_rate": pass_rate,
            "errors": errors[:5],  # Limit to first 5 errors
            "_test_results": test_results,  # For logging
            "_metadata": metadata,
        }

    except Exception as e:
        error_msg = str(e)[:200]
        return {
            "solution_id": solution_id,
            "passed": 0,
            "failed": test_count,
            "total": test_count,
            "tests_run": 0,
            "pass_rate": 0.0,
            "errors": [f"Execution error: {error_msg}"]
        }


# ============================================================================
# Configuration
# ============================================================================

import os

# Use timestamp from environment variable if provided, otherwise use current time
KNOWLEDGE_BASE_TIMESTAMP = os.getenv("KNOWLEDGE_BASE_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

# Create time-stamped knowledge base directory
KNOWLEDGE_BASE_ROOT = Path("./knowledge_base")
KNOWLEDGE_BASE_DIR = KNOWLEDGE_BASE_ROOT / KNOWLEDGE_BASE_TIMESTAMP
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[MCP Server] Knowledge base directory: {KNOWLEDGE_BASE_DIR}")

# Control whether shared knowledge is enabled (set by runner via environment variable)
USE_SHARED_KNOWLEDGE = os.getenv("USE_SHARED_KNOWLEDGE", "0") == "1"
print(f"[MCP Server] Shared knowledge enabled: {USE_SHARED_KNOWLEDGE}")

MAX_KNOWLEDGE_ITEMS = 5
MAX_SHARED_KNOWLEDGE_ITEMS = 10
SHARED_KNOWLEDGE_KEY = "__shared__"
SHARED_KNOWLEDGE_FILE = KNOWLEDGE_BASE_DIR / "shared_knowledge.json"

# Strategy statistics configuration
STRATEGY_STATS_FILE = KNOWLEDGE_BASE_DIR / "strategy_stats.json"
# Global strategy statistics (persist across problems)
# Key: strategy_key_technique, Value: {"wins": int, "tries": int}
_strategy_stats: Dict[str, Dict[str, int]] = {}

# Test failure logs directory
TEST_FAILURE_LOGS_DIR = KNOWLEDGE_BASE_DIR / "test_failure_logs"
TEST_FAILURE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[MCP Server] Test failure logs directory: {TEST_FAILURE_LOGS_DIR}")

# ============================================================================
# File Locking for Async Safety
# ============================================================================

_file_locks: Dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()

# Cache for candidate solutions (registered before KM runs, used during update_knowledge)
# Key: problem_id, Value: List[str] where solutions are in shuffled order
_candidate_solutions: Dict[str, List[str]] = {}

# Cache for ID mappings (shuffled_idx -> original_id)
# Key: problem_id, Value: Dict[int, int] mapping shuffled index to original ID
# original_id: 0 = reference solution, 1+ = rollout solutions
_id_mappings: Dict[str, Dict[int, int]] = {}

# Cache for strategies (generated by KM, used in next round)
# Key: problem_id (sanitized), Value: List of strategy dicts with key_technique and what_to_try
_strategies: Dict[str, List[Dict[str, str]]] = {}


def clear_problem_caches(problem_id: str = None) -> dict:
    """
    Clear in-memory caches to free memory.

    This function helps prevent unbounded memory growth during long-running evaluations
    by clearing cached solutions, ID mappings, and strategies.

    Args:
        problem_id: If provided, clear caches only for this problem.
                   If None, clear all caches.

    Returns:
        Dict with counts of cleared items: {"solutions": n, "mappings": n, "strategies": n}
    """
    global _candidate_solutions, _id_mappings, _strategies

    cleared = {"solutions": 0, "mappings": 0, "strategies": 0}

    if problem_id:
        sanitized_id = sanitize_problem_id(problem_id)
        # Also try the file stem version
        file_stem = f"problem_{sanitized_id}" if not sanitized_id.startswith("problem_") else sanitized_id

        for key in [sanitized_id, file_stem, problem_id]:
            if key in _candidate_solutions:
                del _candidate_solutions[key]
                cleared["solutions"] += 1
            if key in _id_mappings:
                del _id_mappings[key]
                cleared["mappings"] += 1
            if key in _strategies:
                del _strategies[key]
                cleared["strategies"] += 1
    else:
        # Clear all caches
        cleared["solutions"] = len(_candidate_solutions)
        cleared["mappings"] = len(_id_mappings)
        cleared["strategies"] = len(_strategies)

        _candidate_solutions.clear()
        _id_mappings.clear()
        _strategies.clear()
        # Note: Keep _file_locks as they're lightweight and reusable

    if any(cleared.values()):
        total = sum(cleared.values())
        print(f"[MCP] Cleared caches: {cleared} (total: {total} entries)")

    return cleared


async def get_file_lock(file_key: str) -> asyncio.Lock:
    """Get or create a lock for a specific problem."""
    async with _locks_lock:
        if file_key not in _file_locks:
            _file_locks[file_key] = asyncio.Lock()
        return _file_locks[file_key]


def sanitize_problem_id(problem_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", problem_id.strip())
    safe_id = safe_id.strip("_")
    return safe_id or "unknown"


def problem_file_stem(problem_id: str) -> str:
    safe_id = sanitize_problem_id(problem_id)
    if safe_id.startswith("problem_"):
        return safe_id
    return f"problem_{safe_id}"


def build_file_path(problem_id: str) -> Path:
    return KNOWLEDGE_BASE_DIR / f"{problem_file_stem(problem_id)}.json"

# ============================================================================
# Safe JSON Utilities (using orjson with fallback to stdlib json)
# ============================================================================

def safe_json_loads(data: bytes | str, default: Any = None) -> Any:
    """
    Safely load JSON data with orjson (fast, strict validation).

    orjson provides:
    - Strict UTF-8 validation (rejects invalid UTF-8)
    - Strict integer handling (rejects integers > 64-bit)
    - Fast parsing (6x faster than stdlib)

    Falls back to stdlib json if orjson unavailable.
    Returns default value on parse errors.
    """
    if not data:
        return default if default is not None else {}

    try:
        if ORJSON_AVAILABLE:
            # orjson.loads accepts both bytes and str
            if isinstance(data, str):
                data = data.encode('utf-8')
            return orjson.loads(data)
        else:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return json.loads(data)
    except (orjson.JSONDecodeError if ORJSON_AVAILABLE else json.JSONDecodeError) as e:
        print(f"[JSON] Parse error: {e}")
        return default if default is not None else {}
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        print(f"[JSON] Encoding error: {e}")
        return default if default is not None else {}
    except Exception as e:
        print(f"[JSON] Unexpected error loading JSON: {e}")
        return default if default is not None else {}


def safe_json_dumps(data: Any, pretty: bool = True) -> str:
    """
    Safely serialize data to JSON string with orjson.

    orjson provides:
    - Native datetime/date serialization
    - Native UUID serialization
    - Strict output (valid JSON guaranteed)
    - Fast serialization (6x faster than stdlib)

    Falls back to stdlib json if orjson unavailable.
    Returns empty object '{}' on serialization errors.
    """
    try:
        if ORJSON_AVAILABLE:
            # orjson options for pretty printing and native type support
            options = orjson.OPT_NON_STR_KEYS
            if pretty:
                options |= orjson.OPT_INDENT_2
            # orjson.dumps returns bytes, decode to str
            return orjson.dumps(data, option=options).decode('utf-8')
        else:
            if pretty:
                return json.dumps(data, indent=2, ensure_ascii=False, default=str)
            return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        # Use simple string to avoid recursive exception formatting
        print(f"[JSON] Serialization error: {type(e).__name__}")
        return "{}"
    except Exception as e:
        # Avoid formatting exception object which can cause recursion
        print(f"[JSON] Unexpected error dumping JSON: {type(e).__name__}")
        return "{}"


async def safe_json_load_file(file_path: Path, default: Any = None) -> Any:
    """
    Safely load JSON from file with atomic read.

    Returns default value if file doesn't exist or on parse errors.
    """
    if not file_path.exists():
        return default if default is not None else {}

    try:
        async with aiofiles.open(file_path, "rb") as f:
            raw = await f.read()
        return safe_json_loads(raw, default)
    except IOError as e:
        print(f"[JSON] IO error reading {file_path}: {e}")
        return default if default is not None else {}
    except Exception as e:
        print(f"[JSON] Unexpected error reading {file_path}: {e}")
        return default if default is not None else {}


async def safe_json_save_file(file_path: Path, data: Any, pretty: bool = True) -> bool:
    """
    Safely save JSON to file with atomic write pattern.

    Uses write-to-temp-then-rename pattern for crash safety:
    1. Write to temporary file in same directory
    2. Flush and sync to disk
    3. Atomically rename to target path

    Returns True on success, False on failure.
    """
    try:
        # Serialize first to catch errors before touching filesystem
        json_str = safe_json_dumps(data, pretty)
        if json_str == "{}":
            # Check if serialization actually failed vs empty data
            if data and data != {}:
                print(f"[JSON] Serialization failed for {file_path}")
                return False

        json_bytes = json_str.encode('utf-8')

        # Create temp file in same directory for atomic rename
        temp_path = file_path.with_suffix('.json.tmp')

        # Write to temp file
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(json_bytes)
            await f.flush()
            # Sync to disk (important for crash safety)
            os.fsync(f.fileno())

        # Atomic rename (same filesystem guarantees atomicity)
        await aiofiles.os.rename(temp_path, file_path)

        return True

    except IOError as e:
        print(f"[JSON] IO error writing {file_path}: {e}")
        # Clean up temp file if it exists
        temp_path = file_path.with_suffix('.json.tmp')
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        return False
    except Exception as e:
        print(f"[JSON] Unexpected error writing {file_path}: {e}")
        return False


# ============================================================================
# Test Failure Logging
# ============================================================================

async def log_test_failure(
    problem_id: str,
    solution_id: int,
    solution_code: str,
    test_cases: List[Dict[str, str]],
    test_results: List[Any],
    failure_type: str,
    error_details: List[str],
    fn_name: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Log test case evaluation failure to disk for debugging.

    Creates a detailed JSON log file containing:
    - Problem and solution identifiers
    - Full solution code
    - Test cases (inputs and expected outputs)
    - Actual test results
    - Error details and failure type

    Args:
        problem_id: Problem identifier
        solution_id: Solution ID (1-indexed)
        solution_code: The full solution code that was tested
        test_cases: List of test cases with input/expected_output
        test_results: Raw results from grade_call_based/grade_stdio
        failure_type: Type of failure (compilation, wrong_answer, timeout, runtime, exception)
        error_details: List of error messages
        fn_name: Function name for call-based tests
        metadata: Additional metadata from test execution
    """
    timestamp = datetime.utcnow().isoformat()
    sanitized_id = sanitize_problem_id(problem_id)

    # Limit test cases to prevent huge log entries that could cause serialization issues
    MAX_TEST_CASES_TO_LOG = 50
    logged_test_cases = test_cases[:MAX_TEST_CASES_TO_LOG] if len(test_cases) > MAX_TEST_CASES_TO_LOG else test_cases

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "problem_id": problem_id,
        "sanitized_id": sanitized_id,
        "solution_id": solution_id,
        "failure_type": failure_type,
        "fn_name": fn_name,
        "error_details": error_details,
        "test_summary": {
            "total_tests": len(test_cases),  # Keep original count for accuracy
            "tests_run": len(test_results) if test_results else 0,
            "passed": sum(1 for r in test_results if r is True) if test_results else 0,
            "test_cases_truncated": len(test_cases) > MAX_TEST_CASES_TO_LOG,
        },
        "solution_code": solution_code,
        "test_cases": logged_test_cases,
        "test_results": [
            {"test_idx": i, "result": r, "status": _result_to_status(r)}
            for i, r in enumerate(test_results)
        ] if test_results else [],
        "metadata": metadata,
    }

    # Generate log filename: problem_id__solution_id__timestamp.json
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    log_filename = f"{sanitized_id}__sol{solution_id}__{safe_timestamp}.json"
    log_path = TEST_FAILURE_LOGS_DIR / log_filename

    # Save to disk with extra safety for recursion errors
    try:
        success = await safe_json_save_file(log_path, log_entry)
        if success:
            print(f"[MCP] Test failure logged: {log_filename}")
        else:
            print(f"[MCP] Warning: Failed to log test failure to {log_filename}")
    except RecursionError:
        print(f"[MCP] Warning: RecursionError while logging test failure")
    except Exception:
        # Silently fail rather than propagate errors from logging
        print(f"[MCP] Warning: Exception while logging test failure")


def _result_to_status(result: Any) -> str:
    """Convert test result code to human-readable status."""
    if result is True:
        return "PASS"
    status_map = {
        -2: "WRONG_ANSWER",
        -3: "TIME_LIMIT_EXCEEDED",
        -4: "RUNTIME_ERROR",
        -1: "UNKNOWN_ERROR",
    }
    if isinstance(result, int):
        return status_map.get(result, f"ERROR_CODE_{result}")
    return str(result)


# ============================================================================
# Knowledge File Class (JSON-based)
# ============================================================================

class KnowledgeFile:
    """Represents a knowledge file with solution history stored as JSON."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data: Dict[str, Any] = {
            "metadata": {},
            "current_solution": None,
            "knowledge_items": [],
            "strategies": [],
            "strategy_history": [],  # Historical strategies with performance
        }

    @property
    def metadata(self) -> Dict[str, str]:
        return self.data.get("metadata", {})

    @metadata.setter
    def metadata(self, value: Dict[str, str]) -> None:
        self.data["metadata"] = value

    @property
    def current_solution(self) -> Optional[Dict[str, str]]:
        return self.data.get("current_solution")

    @current_solution.setter
    def current_solution(self, value: Optional[Dict[str, str]]) -> None:
        self.data["current_solution"] = value

    @property
    def knowledge_items(self) -> List[Dict[str, Any]]:
        return self.data.get("knowledge_items", [])

    @knowledge_items.setter
    def knowledge_items(self, value: List[Dict[str, Any]]) -> None:
        self.data["knowledge_items"] = value

    @property
    def strategies(self) -> List[Dict[str, str]]:
        return self.data.get("strategies", [])

    @strategies.setter
    def strategies(self, value: List[Dict[str, str]]) -> None:
        self.data["strategies"] = value

    @property
    def strategy_history(self) -> List[Dict[str, Any]]:
        return self.data.get("strategy_history", [])

    @strategy_history.setter
    def strategy_history(self, value: List[Dict[str, Any]]) -> None:
        self.data["strategy_history"] = value

    async def load(self) -> None:
        """Load knowledge from JSON file using safe utilities."""
        if not self.file_path.exists():
            return

        loaded = await safe_json_load_file(self.file_path, default=None)
        if loaded is not None:
            self.data = loaded
        # On error, keep default empty structure

    def set_strategies(self, strategies: List[Dict[str, str]]) -> None:
        """Set strategies for next round."""
        self.data["strategies"] = strategies

    def add_strategy_history(self, strategy_performance: List[Dict[str, Any]], round_num: int, timestamp: str) -> None:
        """Add strategy performance from current round to history.

        Args:
            strategy_performance: List of strategy performance dicts with keys:
                - key_technique: Strategy name
                - what_to_try: Strategy description
                - pass_rate: Test pass rate (0.0-1.0), optional
                - was_selected: Whether this strategy's solution was chosen
            round_num: The round number
            timestamp: ISO timestamp
        """
        history = self.data.get("strategy_history", [])

        # Add round entry
        round_entry = {
            "round": round_num,
            "timestamp": timestamp,
            "strategies": strategy_performance,
        }
        history.append(round_entry)

        # Keep last 10 rounds of history to prevent unbounded growth
        MAX_STRATEGY_HISTORY_ROUNDS = 10
        if len(history) > MAX_STRATEGY_HISTORY_ROUNDS:
            history = history[-MAX_STRATEGY_HISTORY_ROUNDS:]

        self.data["strategy_history"] = history

    def set_solution(self, content: str, timestamp: str) -> None:
        """Set the current solution."""
        self.data["current_solution"] = {
            "timestamp": timestamp,
            "content": content,
        }

    def add_knowledge(self, content: str, timestamp: str) -> None:
        """Add a new knowledge item."""
        items = self.data.get("knowledge_items", [])
        items.append({
            "timestamp": timestamp,
            "content": content,
            "status": True,
        })
        # Keep only the last MAX_KNOWLEDGE_ITEMS entries
        if len(items) > MAX_KNOWLEDGE_ITEMS:
            items = items[-MAX_KNOWLEDGE_ITEMS:]
        self.data["knowledge_items"] = items

    def disable_active_entry(self, active_index: int) -> bool:
        """Disable a knowledge entry by its 1-based active index."""
        if active_index < 1:
            return False

        items = self.data.get("knowledge_items", [])
        active_positions: List[int] = []
        for idx, item in enumerate(items):
            item_status = item.get("status")
            if item_status is None or item_status:
                active_positions.append(idx)

        if active_index > len(active_positions):
            return False

        target_idx = active_positions[active_index - 1]
        items[target_idx]["status"] = False
        return True

    async def save(self) -> None:
        """Save knowledge to JSON file using atomic write."""
        success = await safe_json_save_file(self.file_path, self.data)
        if not success:
            print(f"[KnowledgeFile] Warning: Failed to save {self.file_path}")

    def format_for_prompt(self) -> str:
        """Format knowledge items for prompt (excludes current solution)."""
        metadata = self.data.get("metadata", {})
        title = metadata.get("name") or f"Problem {metadata.get('problem_id', '')}".strip()
        lines: List[str] = [f"# {title}"]
        description = metadata.get("description")
        if description:
            lines.append(f"**Problem**: {description}")

        # Note: Current solution is now returned separately via format_reference_solution()
        # and included in the "To be Ranked Solutions" section

        items = self.data.get("knowledge_items", [])
        active_items = [item for item in items if item.get("status") is None or item.get("status")]
        if active_items:
            lines.append("")
            lines.append("## Knowledge Items")
            for idx, item in enumerate(active_items, 1):
                detail = item.get("content", "")
                lines.append(f"{idx}. {detail}")

        solution = self.data.get("current_solution")
        if not solution and not active_items:
            return "No previous attempts for this problem."

        return "\n".join(lines)

    def format_reference_solution(self) -> str:
        """Format the current solution as a reference solution for ranking."""
        solution = self.data.get("current_solution")
        if not solution:
            return ""

        content = solution.get("content", "")
        if not content:
            return ""

        # Strip existing markdown code fences if present to avoid double-wrapping
        # The content field in JSON may already be wrapped in ```python ... ```
        content = content.strip()
        if content.startswith("```python"):
            content = content[len("```python"):].lstrip('\n')
        elif content.startswith("```"):
            content = content[3:].lstrip('\n')

        if content.endswith("```"):
            content = content[:-3].rstrip('\n')

        return content


# ============================================================================
# Shared Knowledge File Class (JSON-based)
# ============================================================================

class SharedKnowledgeFile:
    """Represents the shared knowledge file with general insights across all problems."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data: Dict[str, Any] = {
            "metadata": {},
            "knowledge_items": [],
        }

    @property
    def metadata(self) -> Dict[str, str]:
        return self.data.get("metadata", {})

    @metadata.setter
    def metadata(self, value: Dict[str, str]) -> None:
        self.data["metadata"] = value

    @property
    def knowledge_items(self) -> List[Dict[str, str]]:
        return self.data.get("knowledge_items", [])

    @knowledge_items.setter
    def knowledge_items(self, value: List[Dict[str, str]]) -> None:
        self.data["knowledge_items"] = value

    async def load(self) -> None:
        """Load shared knowledge from JSON file using safe utilities."""
        if not self.file_path.exists():
            return

        loaded = await safe_json_load_file(self.file_path, default=None)
        if loaded is not None:
            self.data = loaded

    def add_knowledge(self, content: str, timestamp: str) -> None:
        """Add a new knowledge item."""
        items = self.data.get("knowledge_items", [])
        items.append({
            "timestamp": timestamp,
            "content": content,
        })
        # Keep only the last MAX_SHARED_KNOWLEDGE_ITEMS entries
        if len(items) > MAX_SHARED_KNOWLEDGE_ITEMS:
            items = items[-MAX_SHARED_KNOWLEDGE_ITEMS:]
        self.data["knowledge_items"] = items

    async def save(self) -> None:
        """Save shared knowledge to JSON file using atomic write."""
        success = await safe_json_save_file(self.file_path, self.data)
        if not success:
            print(f"[SharedKnowledgeFile] Warning: Failed to save {self.file_path}")

    def format_for_prompt(self) -> str:
        """Format shared knowledge for prompt."""
        items = self.data.get("knowledge_items", [])
        if not items:
            return ""

        lines: List[str] = ["# General Knowledge Base"]
        lines.append(f"{len(items)} general insights available.")
        lines.append("")
        for idx, item in enumerate(items, 1):
            detail = item.get("content", "")
            lines.append(f"{idx}. {detail}")

        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

async def load_existing_file(problem_id: str) -> Optional[KnowledgeFile]:
    """Load existing knowledge file if it exists."""
    file_path = build_file_path(problem_id)
    if not file_path.exists():
        return None
    kf = KnowledgeFile(file_path)
    await kf.load()
    return kf


async def ensure_file(problem_id: str, problem_description: str, timestamp: str) -> KnowledgeFile:
    """Load existing or create new knowledge file."""
    file_path = build_file_path(problem_id)
    kf = KnowledgeFile(file_path)
    if file_path.exists():
        await kf.load()
    else:
        kf.data["metadata"] = {
            "name": f"Problem {problem_id}",
            "description": problem_description,
            "problem_id": problem_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    return kf


async def load_shared_knowledge() -> Optional[SharedKnowledgeFile]:
    """Load the shared knowledge file if it exists."""
    if not SHARED_KNOWLEDGE_FILE.exists():
        return None
    skf = SharedKnowledgeFile(SHARED_KNOWLEDGE_FILE)
    await skf.load()
    return skf


async def ensure_shared_knowledge(timestamp: str) -> SharedKnowledgeFile:
    """Load or create the shared knowledge file."""
    skf = SharedKnowledgeFile(SHARED_KNOWLEDGE_FILE)
    if SHARED_KNOWLEDGE_FILE.exists():
        await skf.load()
    else:
        skf.data["metadata"] = {
            "name": "Shared Knowledge Base",
            "description": "General insights and failure modes applicable across all problems",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    return skf

# ============================================================================
# MCP Server
# ============================================================================

SERVER_INSTRUCTIONS = """
# Knowledge Flow MCP Server

Track solution attempts and learn from past mistakes.

## Tools

### check_knowledge(problem_id, problem_description)
Get previous solution and insights for a problem.

### update_knowledge(problem_id, problem_description, selected_solution_id, ...)
Save selected solution and optional reflection.
- `selected_solution_id`: 0 = reference solution, 1+ = rollout solutions
- `reflection`: What was wrong with previous attempts (problem-specific)
- `general_reflection`: Insights applicable across all problems
- `disable_knowledge_id`: Remove obsolete knowledge entry by index
"""

mcp = FastMCP("knowledge_flow", instructions=SERVER_INSTRUCTIONS)

@mcp.tool()
async def check_knowledge(
    problem_id: str,
    problem_description: str,
) -> str:
    """
    Check if this problem has been attempted before.

    Retrieves the latest solution and active knowledge items for a problem.
    Also displays shared general knowledge applicable across all problems.
    Files are addressed directly by sanitized problem_id values.

    Args:
        problem_id: Unique identifier (e.g., "problem_0")
        problem_description: Description of the problem

    Returns:
        Previous solution history with reflections on what was wrong, or
        a message indicating this is a new problem with no previous attempts.
        General knowledge is shown first, followed by problem-specific knowledge.
    """
    print(f"[MCP] check_knowledge called for: {problem_id}")

    # Load shared knowledge first (only if enabled)
    skf = None
    shared_knowledge_text = ""
    if USE_SHARED_KNOWLEDGE:
        skf = await load_shared_knowledge()
        if skf and skf.knowledge_items:
            shared_knowledge_text = skf.format_for_prompt()
            print(f"[MCP] Loaded {len(skf.knowledge_items)} general knowledge items")
    else:
        print(f"[MCP] Shared knowledge disabled, skipping")

    file_stem = problem_file_stem(problem_id)
    kf = await load_existing_file(problem_id)

    # Build output with shared knowledge first, then problem-specific
    output_parts = []

    # Add shared knowledge section if available
    if shared_knowledge_text:
        output_parts.append(shared_knowledge_text)
        output_parts.append("")
        output_parts.append("---")
        output_parts.append("")

    if kf:
        has_solution = bool(kf.current_solution)
        active_knowledge = [item for item in kf.knowledge_items if item.get("status") is None or item.get("status")]
        knowledge_count = len(active_knowledge)

        print(f"[MCP] Found existing file: {file_stem} (solution={int(has_solution)}, active knowledge={knowledge_count})")

        output_parts.append("# Previous Attempts Found")
        output_parts.append("")
        output_parts.append(f"Latest solution recorded: {'Yes' if has_solution else 'No'}.")
        output_parts.append(f"Active knowledge items: {knowledge_count}.")
        output_parts.append("")
        output_parts.append("---")
        output_parts.append("")
        body = kf.format_for_prompt()
        if body:
            output_parts.append(body)

        # Add strategies section if available for this problem
        # Check in-memory cache first, then fall back to file
        sanitized_id = problem_file_stem(problem_id)
        strategies = None
        if sanitized_id in _strategies and _strategies[sanitized_id]:
            strategies = _strategies[sanitized_id]
        elif kf.strategies:
            strategies = kf.strategies
            # Also cache in memory for future lookups
            _strategies[sanitized_id] = strategies

        if strategies:
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")
            output_parts.append("## Strategies for This Round")
            for idx, strat in enumerate(strategies, 1):
                strat_id = strat.get("id", idx)
                output_parts.append(f"### Strategy {strat_id}")
                output_parts.append(f"**Key Technique**: {strat.get('key_technique', '')}")
                output_parts.append(f"**What to Try**: {strat.get('what_to_try', '')}")
                output_parts.append("")
            print(f"[MCP] Included {len(strategies)} strategies for {problem_id}")

        # Add strategy history section if available
        strategy_history = kf.strategy_history
        if strategy_history:
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")
            output_parts.append("## Strategy History (Previous Rounds)")
            output_parts.append("*What was tried before and how it performed:*")
            output_parts.append("")

            for round_entry in strategy_history:
                round_num = round_entry.get("round", "?")
                round_strategies = round_entry.get("strategies", [])
                if not round_strategies:
                    continue

                output_parts.append(f"**Round {round_num}:**")
                for strat in round_strategies:
                    technique = strat.get("key_technique", "Unknown")
                    pass_rate = strat.get("pass_rate")
                    was_selected = strat.get("was_selected", False)

                    if pass_rate is not None:
                        rate_str = f"{pass_rate:.0%}"
                    else:
                        rate_str = "N/A"

                    if was_selected:
                        result = "SELECTED"
                    elif pass_rate is not None and pass_rate >= 0.8:
                        result = "good"
                    elif pass_rate is not None and pass_rate >= 0.5:
                        result = "partial"
                    elif pass_rate is not None:
                        result = "failed"
                    else:
                        result = "untested"

                    output_parts.append(f"- {technique}: {rate_str} ({result})")
                output_parts.append("")

            print(f"[MCP] Included {len(strategy_history)} rounds of strategy history")

        # Add reference solution as a separate clearly-marked section for parsing
        reference_solution = kf.format_reference_solution()
        if reference_solution:
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")
            output_parts.append("## Reference Solution")
            output_parts.append("```python")
            output_parts.append(reference_solution)
            output_parts.append("```")

        return "\n".join(output_parts)

    print(f"[MCP] No previous attempts found for {problem_id}")

    # Even if no problem-specific knowledge, show shared knowledge if available
    if shared_knowledge_text:
        output_parts.append("# No Previous Attempts Found")
        output_parts.append("")
        output_parts.append("This is the first attempt at this problem.")
        output_parts.append("No previous solution or knowledge is stored yet.")
        return "\n".join(output_parts)

    return (
        "# No Previous Attempts Found\n\n"
        "This is the first attempt at this problem.\n"
        "No previous solution or knowledge is stored yet."
    )


@mcp.tool()
async def register_candidate_solutions(
    problem_id: str,
    solutions_json: str,
    id_mapping_json: str = "",  # Optional: shuffled_idx -> original_id mapping
) -> str:
    """
    Register candidate solutions for a problem before the Knowledge Manager runs.

    The runner calls this to store solutions that can be referenced by ID
    in the subsequent update_knowledge call.

    Args:
        problem_id: Unique identifier (e.g., "problem_0")
        solutions_json: JSON array of solution strings in shuffled order.
                        Solutions are presented to the model anonymously.
        id_mapping_json: Optional JSON object mapping shuffled index (0-based) to original ID.
                        original_id: 0 = reference solution, 1+ = rollout solutions.
                        Example: {"0": 2, "1": 0, "2": 1} means:
                        - Solution 1 (shuffled_idx=0) is actually rollout 2 (original_id=2)
                        - Solution 2 (shuffled_idx=1) is actually the reference (original_id=0)
                        - Solution 3 (shuffled_idx=2) is actually rollout 1 (original_id=1)

    Returns:
        Confirmation message
    """
    print(f"[MCP] register_candidate_solutions called for: {problem_id}")

    try:
        solutions = json.loads(solutions_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON in solutions_json: {e}"

    if not isinstance(solutions, list):
        return "ERROR: solutions_json must be a JSON array"

    sanitized_id = sanitize_problem_id(problem_id)
    _candidate_solutions[sanitized_id] = solutions

    # Parse and store ID mapping if provided
    if id_mapping_json:
        try:
            id_mapping_raw = json.loads(id_mapping_json)
            # Convert string keys to int if needed (JSON keys are always strings)
            id_mapping = {int(k): v for k, v in id_mapping_raw.items()}
            _id_mappings[sanitized_id] = id_mapping
            print(f"[MCP] Stored ID mapping for {sanitized_id}: {id_mapping}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[MCP] WARNING: Could not parse id_mapping_json: {e}")
            _id_mappings[sanitized_id] = {}
    else:
        # No mapping provided - use identity mapping (shuffled_idx == original_id)
        _id_mappings[sanitized_id] = {}

    print(f"[MCP] Registered {len(solutions)} candidate solutions for {sanitized_id} (shuffled order)")
    for i, sol in enumerate(solutions):
        original_id = _id_mappings.get(sanitized_id, {}).get(i, i)
        origin_label = "reference" if original_id == 0 else f"rollout {original_id}"
        print(f"[MCP]   - Solution {i + 1} (shuffled_idx={i}, original={origin_label}): {len(sol) if sol else 0} chars")

    return f"Registered {len(solutions)} candidate solutions for {sanitized_id}"


@mcp.tool()
async def update_knowledge(
    problem_id: str,
    problem_description: str,
    selected_solution_id: Optional[int],  # MANDATORY: 0=reference, 1+=rollouts
    reflection: Optional[str] = "",
    current_solution: Optional[str] = "",  # Optional when selected_solution_id is set
    disable_knowledge_id: Optional[int] = None,
    general_reflection: Optional[str] = "",
    strategies: Optional[List[Dict[str, str]]] = None,  # List of strategy dicts for next round
    strategy_performance: Optional[List[Dict[str, Any]]] = None,  # Performance of this round's strategies
    # NOTE: reference_solution_is_wrong is automatically derived from selected_solution_id
) -> str:
    """
    Save a solution and optional reflection to the knowledge base.

    Persists the latest solution snapshot and optionally appends a knowledge
    entry explaining what changed. Existing knowledge can be disabled by
    referencing its 1-based identifier.

    Args:
        problem_id: Unique identifier matching check_knowledge call
        problem_description: Problem description matching check_knowledge call
        selected_solution_id: REQUIRED. ID of the selected solution.
            ID mapping: 0=reference solution, 1+=rollout solutions.
            The runner looks up the actual solution from registered candidates.
            NOTE: reference_solution_is_wrong is automatically set to True when
            selected_solution_id != 0 (i.e., a rollout was chosen over reference).
        reflection: Optional explanation of what was wrong with previous attempts.
            IMPORTANT: Make reflections STANDALONE and ACTIONABLE.
            Do NOT reference solution IDs (e.g., "Solution 1 failed...").
            Instead, describe the bug pattern generically.
            Examples:
            - "Off-by-one error in loop boundary when handling single-element arrays"
            - "Missing edge case for negative numbers; fixed by adding abs() check"
            - "Used O(n^2) nested loops instead of O(n) hash map for duplicate detection"

    Optional Args:
        current_solution: Optional solution code. Not required when selected_solution_id
            is set, as the runner looks up the actual solution from registered candidates.
            Only provide this for backwards compatibility or manual testing.
        disable_knowledge_id: 1-based index of the knowledge item to disable.
            The index corresponds to the numbering shown in the Knowledge Items
            section returned by check_knowledge. If omitted, no item is disabled.
        general_reflection: Optional general insight applicable across all problems.
            Use this for broadly applicable failure modes and best practices.
            Requires minimum 100 characters (stricter than regular reflection).
            Examples:
            - "When approaching token limit, always output partial solution with clear
              continuation points rather than stopping mid-thought. Include what's done
              and what remains."
        strategies: List of strategy dicts for next round. Each dict should have:
            - key_technique: The algorithm/technique name (e.g., "Dynamic Programming")
            - what_to_try: Specific approach to implement
            Example: [{"key_technique": "DP", "what_to_try": "Use memoization"}]

    Returns:
        Confirmation with statistics about saved knowledge
    """
    # Normalize inputs to handle None values
    current_solution = current_solution or ""
    reflection = reflection or ""
    general_reflection = general_reflection or ""
    selected_solution_id = selected_solution_id if selected_solution_id is not None else -1

    # Convert 1-indexed selected_solution_id (from model) to 0-indexed shuffled_idx
    # Model sees "Solution 1, Solution 2, etc." so it returns 1-indexed IDs
    shuffled_idx = selected_solution_id - 1 if selected_solution_id >= 1 else 0

    sanitized_id = problem_file_stem(problem_id)

    # Map shuffled index back to original ID using the stored mapping
    # original_id: 0 = reference solution, 1+ = rollout solutions
    id_mapping = _id_mappings.get(sanitized_id, {})
    original_id = id_mapping.get(shuffled_idx, shuffled_idx)  # Fallback to identity if no mapping

    # Derive reference_solution_is_wrong from original_id (not shuffled)
    # If original_id != 0, it means a rollout was chosen over reference
    reference_solution_is_wrong = (original_id != 0)

    print(f"[MCP] update_knowledge called for: {problem_id}")
    print(f"[MCP]   - Selected solution ID (1-indexed): {selected_solution_id}")
    print(f"[MCP]   - Shuffled index (0-indexed): {shuffled_idx}")
    print(f"[MCP]   - Original ID: {original_id} ({'reference' if original_id == 0 else f'rollout {original_id}'})")
    print(f"[MCP]   - Reference solution is wrong: {reference_solution_is_wrong}")
    print(f"[MCP]   - Solution length: {len(current_solution)} chars")
    print(f"[MCP]   - Reflection provided: {bool(reflection.strip())}")
    print(f"[MCP]   - General reflection provided: {bool(general_reflection.strip())}")
    print(f"[MCP]   - Disable knowledge ID: {disable_knowledge_id}")

    # Trim inputs
    trimmed_reflection = reflection.strip()
    trimmed_general_reflection = general_reflection.strip()

    file_lock = await get_file_lock(sanitized_id)

    async with file_lock:
        timestamp = datetime.utcnow().isoformat()
        kf = await ensure_file(problem_id, problem_description, timestamp)

        # Update metadata
        metadata = kf.data.get("metadata", {})
        metadata.setdefault("problem_id", problem_id)
        metadata.setdefault("name", f"Problem {problem_id}")
        metadata["description"] = problem_description
        metadata.setdefault("created_at", timestamp)
        metadata["updated_at"] = timestamp
        metadata["reference_solution_is_wrong"] = reference_solution_is_wrong
        kf.data["metadata"] = metadata

        # Look up actual solution from registered candidates using shuffled index
        # Solutions in _candidate_solutions are stored in shuffled order
        if sanitized_id in _candidate_solutions:
            candidates = _candidate_solutions[sanitized_id]
            if 0 <= shuffled_idx < len(candidates):
                actual_solution = candidates[shuffled_idx]
                if actual_solution:
                    kf.set_solution(actual_solution, timestamp)
                    print(f"[MCP] Saved solution from shuffled_idx={shuffled_idx} (original_id={original_id}, {len(actual_solution)} chars)")
                else:
                    print(f"[MCP] Solution at shuffled_idx={shuffled_idx} is empty, keeping previous solution")
            else:
                print(f"[MCP] Invalid shuffled_idx {shuffled_idx}, candidates have {len(candidates)} entries")
        elif current_solution.strip():
            # Fallback: use current_solution if provided and no candidates registered
            kf.set_solution(current_solution.strip(), timestamp)
            print(f"[MCP] No candidates registered, using current_solution fallback ({len(current_solution.strip())} chars)")
        else:
            print(f"[MCP] No candidates registered for {sanitized_id} and no current_solution provided")

        # Handle disable_knowledge_id - single integer only
        knowledge_disabled = False
        disable_message = ""
        disable_target: Optional[int] = None

        if disable_knowledge_id is not None:
            if isinstance(disable_knowledge_id, int):
                disable_target = disable_knowledge_id
            elif isinstance(disable_knowledge_id, float):
                if disable_knowledge_id.is_integer():
                    disable_target = int(disable_knowledge_id)
            elif isinstance(disable_knowledge_id, str):
                candidate = disable_knowledge_id.strip()
                if candidate.isdigit():
                    disable_target = int(candidate)

            if disable_target is not None and disable_target > 0:
                knowledge_disabled = kf.disable_active_entry(disable_target)
                if knowledge_disabled:
                    print(f"[MCP] Disabled knowledge item {disable_target}")
                else:
                    disable_message = f"No active knowledge item found for id {disable_target}."
            elif disable_target is not None and disable_target <= 0:
                disable_message = "disable_knowledge_id must be positive."
            else:
                disable_message = "Invalid disable_knowledge_id provided."

            if disable_message:
                print(f"[MCP] {disable_message}")

        # Add knowledge entry if reflection provided
        knowledge_added = False
        if trimmed_reflection:
            kf.add_knowledge(trimmed_reflection, timestamp)
            knowledge_added = True
            print("[MCP] Added knowledge item")

        # Store strategies for next round if provided (persist to file)
        strategies_stored = 0
        if strategies:
            # Handle both list and string inputs for robustness
            strat_list = strategies
            if isinstance(strategies, str):
                try:
                    strat_list = json.loads(strategies)
                except json.JSONDecodeError as e:
                    print(f"[MCP] Failed to parse strategies string: {e}")
                    strat_list = []

            if isinstance(strat_list, list) and len(strat_list) > 0:
                # Add IDs to strategies if not present
                for idx, strat in enumerate(strat_list, 1):
                    if isinstance(strat, dict) and "id" not in strat:
                        strat["id"] = idx
                kf.set_strategies(strat_list)
                _strategies[sanitized_id] = strat_list  # Also keep in memory
                strategies_stored = len(strat_list)
                print(f"[MCP] Stored {strategies_stored} strategies for {sanitized_id}")

        # Store strategy performance history if provided
        strategy_history_added = 0
        if strategy_performance:
            # Handle both list and string inputs
            perf_list = strategy_performance
            if isinstance(strategy_performance, str):
                try:
                    perf_list = json.loads(strategy_performance)
                except json.JSONDecodeError as e:
                    print(f"[MCP] Failed to parse strategy_performance string: {e}")
                    perf_list = []

            if isinstance(perf_list, list) and len(perf_list) > 0:
                # Get current round number from history length
                current_round = len(kf.strategy_history) + 1
                # Mark the selected strategy as winner
                for perf in perf_list:
                    if isinstance(perf, dict):
                        rollout_id = perf.get("rollout_id", 0)
                        # A rollout r_idx has original_id = r_idx + 1, so check if this rollout won
                        perf["was_selected"] = (original_id == rollout_id)
                kf.add_strategy_history(perf_list, current_round, timestamp)
                strategy_history_added = len(perf_list)
                print(f"[MCP] Added {strategy_history_added} strategies to history (round {current_round})")

        await kf.save()

    # Handle general reflection (shared knowledge) with separate lock
    # Only save if shared knowledge is enabled
    general_knowledge_added = False
    general_knowledge_total = 0
    if trimmed_general_reflection and USE_SHARED_KNOWLEDGE:
        shared_lock = await get_file_lock(SHARED_KNOWLEDGE_KEY)
        async with shared_lock:
            timestamp = datetime.utcnow().isoformat()
            skf = await ensure_shared_knowledge(timestamp)

            skf.data["metadata"]["updated_at"] = timestamp
            skf.add_knowledge(trimmed_general_reflection, timestamp)
            general_knowledge_added = True
            general_knowledge_total = len(skf.knowledge_items)

            await skf.save()
            print(f"[MCP] Added general knowledge item (total: {general_knowledge_total})")
    elif trimmed_general_reflection and not USE_SHARED_KNOWLEDGE:
        print(f"[MCP] Shared knowledge disabled, ignoring general_reflection ({len(trimmed_general_reflection)} chars)")

    active_total = sum(1 for item in kf.knowledge_items if item.get("status") is None or item.get("status"))
    response: List[str] = [f"Updated knowledge for {sanitized_id}"]
    origin_label = "reference" if original_id == 0 else f"rollout {original_id}"
    response.append(f"Solution saved: Yes (Solution {selected_solution_id} -> original: {origin_label})")
    response.append(f"Reference solution flagged as wrong: {reference_solution_is_wrong}")
    response.append(f"Knowledge added: {knowledge_added}")
    response.append(f"Knowledge disabled: {knowledge_disabled}")
    response.append(f"Active knowledge total: {active_total}")
    if general_knowledge_added:
        response.append(f"General knowledge added: Yes (total general insights: {general_knowledge_total})")
    if strategies_stored > 0:
        response.append(f"Strategies stored for next round: {strategies_stored}")
    if strategy_history_added > 0:
        response.append(f"Strategy history updated: {strategy_history_added} strategies added")
    if disable_message:
        response.append(disable_message)

    # Clear caches for this problem after persisting to free memory
    # Solutions and ID mappings are no longer needed after knowledge is updated
    cleared = clear_problem_caches(problem_id)
    if any(cleared.values()):
        response.append(f"Caches cleared: {cleared}")

    return "\n".join(response)


# ============================================================================
# Strategy Statistics Persistence
# ============================================================================

async def load_strategy_stats():
    """Load strategy statistics from file using safe utilities."""
    global _strategy_stats
    if STRATEGY_STATS_FILE.exists():
        loaded = await safe_json_load_file(STRATEGY_STATS_FILE, default={})
        _strategy_stats = loaded
        print(f"[MCP] Loaded {len(_strategy_stats)} strategy stats from file")


async def save_strategy_stats():
    """Save strategy statistics to file using atomic write."""
    success = await safe_json_save_file(STRATEGY_STATS_FILE, _strategy_stats)
    if not success:
        print(f"[MCP] Warning: Could not save strategy stats")


@mcp.tool()
async def store_strategy_performance(
    problem_id: str,
    strategy_performance: List[Dict[str, Any]],
    selected_original_id: int,
) -> str:
    """
    Store strategy performance history for a problem.

    Called by the runner after Phase 2 to record which strategies were tried
    and how they performed. This is separate from update_knowledge because
    the runner has the test results data, not the KM agent.

    Args:
        problem_id: The problem identifier (e.g., "problem_0")
        strategy_performance: List of strategy performance dicts with:
            - rollout_id: The rollout number (1-indexed)
            - key_technique: Strategy name
            - what_to_try: Strategy description
            - pass_rate: Test pass rate (0.0-1.0), optional
        selected_original_id: The original_id of the selected solution
            (0 = reference, 1+ = rollout index)

    Returns:
        Confirmation message
    """
    print(f"[MCP] store_strategy_performance: {problem_id}, {len(strategy_performance)} strategies")

    sanitized_id = problem_file_stem(problem_id)
    file_lock = await get_file_lock(sanitized_id)

    async with file_lock:
        timestamp = datetime.utcnow().isoformat()
        kf = await load_existing_file(problem_id)

        if not kf:
            return f"No knowledge file found for {problem_id}"

        # Handle both list and string inputs
        perf_list = strategy_performance
        if isinstance(strategy_performance, str):
            try:
                perf_list = json.loads(strategy_performance)
            except json.JSONDecodeError as e:
                return f"Failed to parse strategy_performance: {e}"

        if not isinstance(perf_list, list) or len(perf_list) == 0:
            return "No strategy performance data provided"

        # Mark the selected strategy as winner
        for perf in perf_list:
            if isinstance(perf, dict):
                rollout_id = perf.get("rollout_id", 0)
                # A rollout r_idx has original_id = r_idx, so check if this rollout won
                perf["was_selected"] = (selected_original_id == rollout_id)

        # Get current round number from history length
        current_round = len(kf.strategy_history) + 1
        kf.add_strategy_history(perf_list, current_round, timestamp)

        await kf.save()

    return f"Stored {len(perf_list)} strategy performance entries for {sanitized_id} (round {current_round})"


@mcp.tool()
async def clear_caches(
    problem_id: Optional[str] = None,
) -> str:
    """
    Clear in-memory caches to free memory during long-running evaluations.

    Clears cached candidate solutions, ID mappings, and strategies.
    This helps prevent unbounded memory growth when processing many problems.

    Args:
        problem_id: If provided, clear caches only for this specific problem.
                   If omitted or None, clear all caches.

    Returns:
        Confirmation message with counts of cleared items.
    """
    print(f"[MCP] clear_caches called for: {problem_id or 'all problems'}")

    cleared = clear_problem_caches(problem_id)

    if problem_id:
        return f"Cleared caches for {problem_id}: {cleared}"
    else:
        return f"Cleared all caches: {cleared}"


# ============================================================================
# Test Execution Tool
# ============================================================================

async def _evaluate_candidate_solution(
    problem_id: str,
    sol_idx: int,
    solution: str,
    test_cases: List[Dict[str, Any]],
    fn_name: Optional[str],
    timeout_per_test: float,
    all_inputs: List[Any],
    all_outputs: List[Any],
) -> Dict[str, Any]:
    """
    Evaluate a single candidate solution asynchronously.
    """
    solution_id = sol_idx + 1  # 1-indexed for KM display
    total_tests = len(test_cases)
    
    if not solution or not solution.strip():
        print(f"[MCP]   Solution {solution_id}: Empty - skipped")
        return {
            "solution_id": solution_id,
            "passed": 0,
            "failed": total_tests,
            "total": total_tests,
            "tests_run": 0,
            "pass_rate": 0.0,
            "errors": ["Empty solution"]
        }

    # Initialize actual_fn_name to hint value
    actual_fn_name = fn_name
    
    try:
        # Detect execution mode (CPU bound, fast enough to run in executor or directly)
        is_call_based, detected_fn_name = _detect_solution_mode(solution, fn_name)
        
        # Define the blocking grading task
        def _run_grading():
            if is_call_based and detected_fn_name:
                return grade_call_based_all(
                    code=solution,
                    all_inputs=all_inputs,
                    all_outputs=all_outputs,
                    fn_name=detected_fn_name,
                    timeout=int(timeout_per_test)
                ), detected_fn_name
            else:
                return grade_stdio_all(
                    code=solution,
                    all_inputs=all_inputs,
                    all_outputs=all_outputs,
                    timeout=int(timeout_per_test)
                ), None

        # Run grading in executor to avoid blocking the async event loop.
        # With the new multiprocessing.Process approach (instead of ProcessPoolExecutor),
        # each test runs in its own process that is explicitly terminated on timeout.
        # This avoids the previous thread starvation issue since processes are independent.
        loop = asyncio.get_event_loop()
        result_tuple, used_fn_name = await loop.run_in_executor(None, _run_grading)
        
        if used_fn_name:
            actual_fn_name = used_fn_name
            print(f"[MCP]   Solution {solution_id}: Detected as call-based (fn={actual_fn_name})")
        else:
            print(f"[MCP]   Solution {solution_id}: Detected as stdin-based")
            
        result = result_tuple

        # Handle compilation failure
        if result is None or result[0] is None:
            error_msg = result[1].get("error", "Compilation failed") if result else "Compilation failed"
            error_list = [error_msg]
            print(f"[MCP]   Solution {solution_id}: {error_msg}")
            
            # Log failure
            await log_test_failure(
                problem_id=problem_id,
                solution_id=solution_id,
                solution_code=solution,
                test_cases=test_cases,
                test_results=[],
                failure_type="compilation",
                error_details=error_list,
                fn_name=actual_fn_name,
            )
            
            return {
                "solution_id": solution_id,
                "passed": 0,
                "failed": total_tests,
                "total": total_tests,
                "tests_run": 0,
                "pass_rate": 0.0,
                "errors": error_list
            }

        # Process results
        test_results, metadata = result
        test_metadata_list = metadata.get("test_metadata", [])
        
        passed = sum(1 for r in test_results if r is True)
        tests_run = len(test_results)
        failed = tests_run - passed
        pass_rate = round(passed / total_tests, 3) if total_tests > 0 else 0
        
        # Build errors list
        errors = []
        failed_test_details = []
        for i, r in enumerate(test_results):
            if r is not True:
                error_desc = {
                    -2: "Wrong Answer",
                    -3: "Time Limit Exceeded",
                    -4: "Runtime Error"
                }.get(r, f"Error code {r}")
                
                # Get test case details safely
                tc = test_cases[i] if i < len(test_cases) else {}
                tc_input = tc.get("input", "N/A")
                tc_expected = tc.get("expected_output", "N/A")

                # Use config-based smart truncation
                config = TEST_RESULT_CONFIG
                tc_input_str = _smart_truncate(str(tc_input), config["max_input_chars"])
                tc_expected_str = _smart_truncate(str(tc_expected), config["max_expected_chars"])

                actual_output = "N/A"
                if i < len(test_metadata_list):
                    tm = test_metadata_list[i]
                    if "output" in tm:
                        actual_output = _smart_truncate(str(tm.get("output", "N/A")), config["max_actual_chars"])
                            
                errors.append(f"Test {i+1}: {error_desc}")
                failed_test_details.append({
                    "test_num": i + 1,
                    "error": error_desc,
                    "input": tc_input_str,
                    "expected": tc_expected_str,
                    "actual": actual_output,
                })

        print(f"[MCP]   Solution {solution_id}: {passed}/{total_tests} passed ({pass_rate:.1%})")

        # Log failures if any
        if failed > 0:
            failure_types = set()
            for r in test_results:
                if r == -2: failure_types.add("wrong_answer")
                elif r == -3: failure_types.add("timeout")
                elif r == -4: failure_types.add("runtime_error")
                elif r is not True: failure_types.add("unknown")
            failure_type = "_".join(sorted(failure_types)) if failure_types else "test_failure"

            await log_test_failure(
                problem_id=problem_id,
                solution_id=solution_id,
                solution_code=solution,
                test_cases=test_cases,
                test_results=test_results,
                failure_type=failure_type,
                error_details=errors,
                fn_name=actual_fn_name,
                metadata=metadata,
            )
            
        return {
            "solution_id": solution_id,
            "passed": passed,
            "failed": failed,
            "total": total_tests,
            "tests_run": tests_run,
            "pass_rate": pass_rate,
            "errors": errors[:TEST_RESULT_CONFIG["max_error_messages"]],
            "failed_tests": failed_test_details[:TEST_RESULT_CONFIG["max_total_failed_tests"]],
        }

    except Exception as e:
        error_msg = str(e)[:200]
        full_traceback = traceback.format_exc()
        truncated_traceback = full_traceback[:2000] if len(full_traceback) > 2000 else full_traceback
        error_list = [f"Execution error: {error_msg}"]
        print(f"[MCP]   Solution {solution_id}: Execution error - {error_msg[:50]}...")
        
        await log_test_failure(
            problem_id=problem_id,
            solution_id=solution_id,
            solution_code=solution,
            test_cases=test_cases,
            test_results=[],
            failure_type="exception",
            error_details=error_list + [truncated_traceback],
            fn_name=actual_fn_name,
        )
        
        return {
            "solution_id": solution_id,
            "passed": 0,
            "failed": total_tests,
            "total": total_tests,
            "tests_run": 0,
            "pass_rate": 0.0,
            "errors": error_list
        }


@mcp.tool()
async def execute_generated_tests(
    problem_id: str,
    test_cases_json: str,
    fn_name: Optional[str] = None,
    timeout_per_test: float = 60.0,
) -> str:
    """
    Execute candidate solutions against KM-generated test cases.

    **Runs ALL test cases** - continues evaluating even after failures to get
    complete pass/fail statistics for each solution.

    The Knowledge Manager calls this tool after generating test cases to
    objectively rank solutions based on actual execution results.

    Args:
        problem_id: Problem identifier (solutions looked up from registered candidates)
        test_cases_json: JSON array of test cases, each with:
            - "input": Test input (JSON-serialized args for call-based, or stdin string)
            - "expected_output": Expected output (JSON-serialized for call-based, or stdout string)
        fn_name: Function name for call-based execution (None for stdin-based)
        timeout_per_test: Timeout in seconds per test case (default: 5.0)

    Returns:
        JSON with execution results per solution:
        {
            "results": [
                {"solution_id": 1, "passed": 3, "failed": 2, "pass_rate": 0.6, "errors": [...]},
                ...
            ],
            "test_count": 5,
            "best_solution_id": 2
        }
    """
    print(f"[MCP] execute_generated_tests called for: {problem_id}")

    if not TEST_EXECUTION_AVAILABLE:
        return json.dumps({
            "error": "Test execution utilities not available. Check LCB path.",
            "results": [],
            "test_count": 0
        })

    sanitized_id = sanitize_problem_id(problem_id)

    # Get registered solutions
    if sanitized_id not in _candidate_solutions:
        return json.dumps({
            "error": f"No solutions registered for {problem_id}. Call register_candidate_solutions first.",
            "results": [],
            "test_count": 0
        })

    solutions = _candidate_solutions[sanitized_id]
    print(f"[MCP] Found {len(solutions)} registered solutions for {sanitized_id}")

    # Parse test cases
    try:
        test_cases = json.loads(test_cases_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid test_cases_json: {e}",
            "results": [],
            "test_count": 0
        })

    if not test_cases or not isinstance(test_cases, list):
        return json.dumps({
            "error": "No valid test cases provided. Expected non-empty JSON array.",
            "results": [],
            "test_count": 0
        })

    print(f"[MCP] Executing ALL {len(test_cases)} test cases against {len(solutions)} solutions (parallel)")

    # Prepare inputs/outputs in LCB format
    all_inputs = [tc.get("input", "") for tc in test_cases]
    all_outputs = [tc.get("expected_output", "") for tc in test_cases]
    total_tests = len(test_cases)

    # Create tasks for all solutions
    tasks = []
    for sol_idx, solution in enumerate(solutions):
        task = _evaluate_candidate_solution(
            problem_id=problem_id,
            sol_idx=sol_idx,
            solution=solution,
            test_cases=test_cases,
            fn_name=fn_name,
            timeout_per_test=timeout_per_test,
            all_inputs=all_inputs,
            all_outputs=all_outputs,
        )
        tasks.append(task)
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Sort results by solution_id to maintain deterministic order
    results.sort(key=lambda x: x["solution_id"])

    # Find best solution by pass rate
    best_solution_id = None
    if results:
        best_result = max(results, key=lambda x: (x["pass_rate"], -x["solution_id"]))
        best_solution_id = best_result["solution_id"]

    response = {
        "results": results,
        "test_count": total_tests,
        "best_solution_id": best_solution_id,
        "summary": f"Tested {len(solutions)} solutions against ALL {total_tests} test cases (parallel execution)",
        "disclaimer": (
            "NOTE: These test cases were generated and may contain errors. "
            "A solution failing a test case does not necessarily mean the solution is wrong - "
            "the test case itself could have incorrect expected output. "
            "Consider the test case logic carefully when evaluating results."
        ),
    }

    # Add formatted markdown for Knowledge Manager if enabled
    if TEST_RESULT_CONFIG["format_as_markdown"]:
        response["formatted_results"] = _format_test_results_for_km(results, total_tests)

    print(f"[MCP] Test execution complete. Best solution: {best_solution_id}")

    # Reset executor to clear any zombie threads from timeout'd tests
    reset_test_executor()

    return json.dumps(response, indent=2)


# ============================================================================
# Strategy Learning Tools
# ============================================================================

@mcp.tool()
async def update_strategy_stats(
    strategy_key_technique: str,
    success: bool,
) -> str:
    """
    Update statistics for a strategy based on whether it produced a passing solution.

    Called after test execution to track which strategies work well.

    Args:
        strategy_key_technique: The strategy name (e.g., "Dynamic Programming", "Binary Search")
        success: True if the strategy produced a solution that passed all tests

    Returns:
        Confirmation with updated statistics
    """
    print(f"[MCP] update_strategy_stats: {strategy_key_technique} -> {'success' if success else 'failure'}")

    # Load existing stats if not already loaded
    if not _strategy_stats:
        await load_strategy_stats()

    # Initialize if new strategy
    if strategy_key_technique not in _strategy_stats:
        _strategy_stats[strategy_key_technique] = {"wins": 0, "tries": 0}

    # Update counts
    _strategy_stats[strategy_key_technique]["tries"] += 1
    if success:
        _strategy_stats[strategy_key_technique]["wins"] += 1

    # Save to file
    await save_strategy_stats()

    stats = _strategy_stats[strategy_key_technique]
    win_rate = stats["wins"] / stats["tries"] if stats["tries"] > 0 else 0

    return f"Updated {strategy_key_technique}: {stats['wins']}/{stats['tries']} wins ({win_rate:.1%})"


@mcp.tool()
async def get_strategy_recommendations(
    num_strategies: int = 3,
) -> str:
    """
    Get strategy recommendations for the next round.

    Uses simple win rate sorting: strategies with higher success rates are ranked higher.
    Untried strategies are given priority to ensure exploration.

    Args:
        num_strategies: Number of strategies to recommend (default: 3)

    Returns:
        JSON with ranked strategy recommendations:
        {
            "strategies": [
                {"technique": "DP", "win_rate": 0.7, "tries": 10, "wins": 7},
                ...
            ],
            "total_strategies": 5
        }
    """
    print(f"[MCP] get_strategy_recommendations: requesting {num_strategies} strategies")

    # Load existing stats if not already loaded
    if not _strategy_stats:
        await load_strategy_stats()

    if not _strategy_stats:
        return json.dumps({
            "strategies": [],
            "message": "No strategy data yet. Strategies will be learned over time.",
            "total_strategies": 0
        })

    total_tries = sum(s["tries"] for s in _strategy_stats.values())
    if total_tries == 0:
        # Return all strategies with no preference
        strategies = [
            {"technique": tech, "win_rate": 0, "tries": 0, "wins": 0}
            for tech in list(_strategy_stats.keys())[:num_strategies]
        ]
        return json.dumps({
            "strategies": strategies,
            "message": "All strategies untried - explore freely",
            "total_strategies": len(_strategy_stats)
        })

    # Simple scoring: win rate with priority for untried strategies
    scored_strategies = []
    for technique, stats in _strategy_stats.items():
        wins = stats["wins"]
        tries = stats["tries"]

        if tries == 0:
            win_rate = 0
            # Give untried strategies high priority (sort them first)
            sort_key = (1, 0)  # (untried=1, win_rate=0)
        else:
            win_rate = wins / tries
            sort_key = (0, -win_rate)  # (untried=0, negative win_rate for descending)

        scored_strategies.append({
            "technique": technique,
            "win_rate": round(win_rate, 3),
            "tries": tries,
            "wins": wins,
            "_sort_key": sort_key,
        })

    # Sort: untried first, then by win rate (highest first)
    scored_strategies.sort(key=lambda x: x["_sort_key"])

    # Remove internal sort key before returning
    for s in scored_strategies:
        del s["_sort_key"]

    response = {
        "strategies": scored_strategies[:num_strategies],
        "total_strategies": len(scored_strategies),
        "total_tries": total_tries
    }

    print(f"[MCP] Returning {len(response['strategies'])} strategy recommendations")
    return json.dumps(response, indent=2)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KFLOW_PORT", "8000"))
    host = os.environ.get("KFLOW_HOST", "127.0.0.1")
    transport = os.environ.get("KFLOW_TRANSPORT", "sse")

    print(f"[KFlow MCP] Starting server on {host}:{port} with {transport} transport")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        # Use uvicorn to run the SSE app with custom port
        app = mcp.sse_app()
        uvicorn.run(app, host=host, port=port)
    elif transport == "streamable-http":
        app = mcp.streamable_http_app()
        uvicorn.run(app, host=host, port=port)
    else:
        print(f"Unknown transport: {transport}")
        mcp.run(transport="stdio")
