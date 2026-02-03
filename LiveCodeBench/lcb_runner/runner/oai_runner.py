import os
from time import sleep  # (unused, but kept if BaseRunner expects it)
import asyncio
import httpx
from datetime import datetime

import openai
from openai import AsyncAzureOpenAI  # using the async Azure client

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner

from tqdm import tqdm
from typing import Optional, Any, List

# Module-level shared HTTP client management
_http_client: Optional[httpx.AsyncClient] = None
_http_client_timeout: Optional[int] = None


def shared_http_client(timeout_seconds: int = 3600) -> httpx.AsyncClient:
    """
    Creates or retrieves a shared httpx.AsyncClient with a custom read timeout.
    """
    global _http_client, _http_client_timeout

    # Recreate client if timeout has changed
    if _http_client is not None and _http_client_timeout != timeout_seconds:
        print(f"[HTTP Client] Timeout changed from {_http_client_timeout}s to {timeout_seconds}s, recreating client")
        # Note: Ideally close the old client in an async context.
        _http_client = None

    if _http_client is None:
        custom_timeout = httpx.Timeout(
            connect=60.0,
            read=float(timeout_seconds),  # important for long-running streams
            write=60.0,
            pool=10.0,
        )
        _http_client = httpx.AsyncClient(timeout=custom_timeout)
        _http_client_timeout = timeout_seconds
        print(f"[HTTP Client] Created with read timeout: {timeout_seconds}s")

    return _http_client


class OpenAIRunner(BaseRunner):
    # Azure endpoint from environment variable (required)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    subscription_key = os.getenv("OPENAI_API_KEY")
    api_version = "2025-04-01-preview"

    def __init__(self, args, model):
        super().__init__(args, model)

        # Async client (uses anyio under the hood, works with asyncio)
        self.async_client = AsyncAzureOpenAI(
            api_version=OpenAIRunner.api_version,
            azure_endpoint=OpenAIRunner.endpoint,
            api_key=OpenAIRunner.subscription_key,
            http_client=shared_http_client(timeout_seconds=args.openai_timeout),
        )

        assert "__" in args.model, (
            f"Model {args.model} is not a valid OpenAI Reasoning model; "
            f"expected a '__' separating model and reasoning effort."
        )
        model_name, reasoning_effort = args.model.split("__", 1)

        # Request kwargs for Responses API
        self.client_kwargs: dict[str, Any] = {
            "model": model_name,
            "max_output_tokens": args.max_tokens,
            "reasoning": {"effort": reasoning_effort},
            "stream": True,         # stream events
            "background": False,     # allow background generation (Azure supports on some deployments)
        }

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _process_stream_response(self, response, problem_idx: Optional[int] = None) -> str:
        """
        Synchronous streaming processor (kept for completeness; currently unused).
        """
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"
        output_text = ""
        in_output_mode = False
        chunk_count = 0

        if stream_debug and problem_idx is not None:
            print(f"\n{'='*80}")
            print(f"┌─ [Streaming Response] Problem {problem_idx} │ {self._get_timestamp()}")
            print(f"{'='*80}\n")

        try:
            for event in response:
                chunk_count += 1

                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    if hasattr(event, "delta") and event.delta:
                        text = event.delta
                        output_text += text
                        if stream_debug:
                            if not in_output_mode:
                                print("┌─ [📝 Output Stream] ─────────────────────────────")
                                in_output_mode = True
                            print(text, end="", flush=True)

            if stream_debug:
                if in_output_mode:
                    print("\n└─ [✓ Stream Complete] ────────────────────────────")
                if problem_idx is not None:
                    print(f"\n{'='*80}")
                    print(f"└─ Problem {problem_idx} │ Stream Complete │ {self._get_timestamp()}")
                    print("    📊 Summary:")
                    print(f"        • Chunks received: {chunk_count}")
                    print(f"        • Output length: {len(output_text)} characters")
                    print(f"{'='*80}\n")

        except Exception as e:
            if stream_debug:
                print(f"\n[❌ Stream Error] {type(e).__name__}: {str(e)}")
            raise

        return output_text

    async def _process_stream_response_async(self, response, problem_idx: Optional[int] = None, completion_idx: Optional[int] = None) -> str:
        """
        Async streaming processor for Responses API.
        """
        stream_debug = os.getenv("STREAM_DEBUG", "false").lower() == "true"
        output_text = ""
        in_output_mode = False
        chunk_count = 0

        if stream_debug and problem_idx is not None:
            print(f"\n{'='*80}")
            if completion_idx is not None:
                n = getattr(self.args, 'n', 1)
                print(f"┌─ [Streaming Response] Problem {problem_idx} | Completion {completion_idx+1}/{n} │ {self._get_timestamp()}")
            else:
                print(f"┌─ [Streaming Response] Problem {problem_idx} │ {self._get_timestamp()}")
            print(f"{'='*80}\n")

        try:
            async for event in response:
                chunk_count += 1

                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    if hasattr(event, "delta") and event.delta:
                        text = event.delta
                        output_text += text
                        if stream_debug:
                            if not in_output_mode:
                                print("┌─ [📝 Output Stream] ─────────────────────────────")
                                in_output_mode = True
                            print(text, end="", flush=True)

            if stream_debug:
                if in_output_mode:
                    print("\n└─ [✓ Stream Complete] ────────────────────────────")
                if problem_idx is not None:
                    print(f"\n{'='*80}")
                    if completion_idx is not None:
                        n = getattr(self.args, 'n', 1)
                        print(f"└─ Problem {problem_idx} | Completion {completion_idx+1}/{n} │ Stream Complete │ {self._get_timestamp()}")
                    else:
                        print(f"└─ Problem {problem_idx} │ Stream Complete │ {self._get_timestamp()}")
                    print("    📊 Summary:")
                    print(f"        • Chunks received: {chunk_count}")
                    print(f"        • Output length: {len(output_text)} characters")
                    print(f"{'='*80}\n")

        except Exception as e:
            if stream_debug:
                print(f"\n[❌ Stream Error] {type(e).__name__}: {str(e)}")
            raise

        return output_text

    async def _run_single_async(
        self,
        prompt: List[dict[str, str]],
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
        problem_idx: Optional[int] = None,
    ) -> List[str]:
        """
        Async version of a single run with retry & concurrency control.
        Generates self.args.n completions for the given prompt.
        """

        assert isinstance(prompt, list)

        n = getattr(self.args, 'n', 1)

        async def _run_one_completion(completion_idx: int) -> str:
            """
            Generate a single completion with retry logic.
            """
            last_error: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        # Optional: visible logging for retry attempts
                        print(f"[RETRY] Problem {problem_idx}, Completion {completion_idx+1}/{n}: attempt {attempt+1}/{max_retries}")

                    # Acquire semaphore for the entire request duration (including stream processing)
                    async with semaphore:
                        # Create response (streaming when self.client_kwargs['stream'] == True)
                        response = await self.async_client.responses.create(
                            input=prompt,
                            **self.client_kwargs,
                        )

                        is_streaming = bool(self.client_kwargs.get("stream", False))
                        if is_streaming:
                            content = await self._process_stream_response_async(
                                response, problem_idx=problem_idx, completion_idx=completion_idx
                            )
                        else:
                            # Non-stream path: extract output_texts
                            output_texts: List[str] = []
                            for output in getattr(response, "output", []) or []:
                                if hasattr(output, "content") and output.content:
                                    for c in output.content:
                                        if getattr(c, "type", None) == "output_text":
                                            output_texts.append(getattr(c, "text", ""))
                                elif getattr(output, "type", None) == "output_text":
                                    output_texts.append(getattr(output, "text", ""))
                            content = "\n\n".join(output_texts) if output_texts else ""

                    # Heuristic: too-short responses -> retry with backoff
                    if content and len(content) < 10 and attempt < max_retries - 1:
                        print(f"\n[WARNING] Problem {problem_idx}, Completion {completion_idx+1}/{n}: Short response detected (len={len(content)}). Retrying...")
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

                    return content  # success -> exit retry loop

                except (
                    openai.APIError,
                    openai.RateLimitError,
                    openai.APIStatusError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ) as e:
                    last_error = e
                    print(f"\n[ERROR] Problem {problem_idx}, Completion {completion_idx+1}/{n}, Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[RETRY] Waiting {wait_time}s before retry... (consider lowering concurrency)")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[ERROR] Problem {problem_idx}, Completion {completion_idx+1}/{n}: Max retries reached for API error.")
                        # Return empty string on final failure
                        return ""

                except Exception as e:
                    last_error = e
                    print(f"\n[ERROR] Problem {problem_idx}, Completion {completion_idx+1}/{n}, Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[RETRY] Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[ERROR] Problem {problem_idx}, Completion {completion_idx+1}/{n}: Max retries reached after unexpected error.")
                        # Return empty string on final failure
                        return ""

            # If we get here, all retries failed
            print(f"\n[FINAL ERROR] Problem {problem_idx}, Completion {completion_idx+1}/{n}: All retries exhausted. Final error: {type(last_error).__name__}: {str(last_error)}")
            return ""

        # Create tasks for n completions and run them in parallel
        tasks = [_run_one_completion(i) for i in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results: convert exceptions to empty strings
        processed_results: List[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"\n[ERROR] Problem {problem_idx}, Completion {i+1}/{n}: Unhandled exception: {type(result).__name__}: {str(result)}")
                processed_results.append("")
            else:
                processed_results.append(result)

        return processed_results

    def run_batch(self, prompts: List[List[dict[str, str]]]) -> List[List[str]]:
        """
        Run a batch of prompts concurrently using asyncio, limited by a semaphore.
        """

        async def _run_batch_async():
            concurrency = getattr(self.args, "multiprocess_oai", 10)
            print(f"Running with concurrency level: {concurrency}")
            semaphore = asyncio.Semaphore(concurrency)
            results: List[Optional[List[str]]] = [None] * len(prompts)

            async def _run_and_store(task_idx: int, prompt: List[dict[str, str]], pbar):
                try:
                    results[task_idx] = await self._run_single_async(
                        prompt, semaphore, problem_idx=task_idx
                    )
                finally:
                    pbar.update(1)

            with tqdm(total=len(prompts), desc="Processing batch") as pbar:
                tasks = [_run_and_store(i, prompt, pbar) for i, prompt in enumerate(prompts)]
                await asyncio.gather(*tasks)

            responses: List[List[str]] = [
                r if isinstance(r, list) else [] for r in results
            ]

            suspicious_count = 0
            for i, resp in enumerate(responses):
                first = resp[0] if (resp and len(resp) > 0) else ""
                if first and len(first) < 10:
                    suspicious_count += 1
                    print(f"\n[DEBUG] Response {i} is suspicious: {repr(first)}")

            if suspicious_count > 0:
                print(f"\n[WARNING] Found {suspicious_count} suspicious short responses out of {len(responses)}")
                print("Inspect the logs or enable STREAM_DEBUG=true for more detail.")

            return responses

        return asyncio.run(_run_batch_async())
