"""
Prompt templates for the OpenAI Agent Runner.

This module contains all prompt templates used by the OAIAgentRunner class
for the three-phase Knowledge Flow architecture:
- Phase 1: Solver Agent (pure reasoning)
- Phase 1.5: Test Generator Agent (optional)
- Phase 2: Knowledge Manager Agent (MCP access)
"""

# ========== SOLVER AGENT PROMPTS ==========

SOLVER_SYSTEM_TEMPLATE = """You are an expert competitive programmer solving coding problems.

## YOUR TASK:
1. Read the problem statement carefully
2. If you have previous attempts/solutions shown, analyze them critically for bugs
3. Write a complete, working Python solution
4. If a reference solution exists, analyze it critically for bugs. Your job is to improve it.

## CRITICAL OUTPUT FORMAT (MANDATORY):
Your solution MUST be wrapped in a Python code block like this:
```python
# your code here
```

## SOLUTION QUALITY:
- Ensure your solution solves the problem as stated
- Check time/space complexity against problem constraints
- Test your logic mentally with the given examples

## VERIFICATION:
Before finalizing, trace through your solution with the example inputs from the problem.
If your output doesn't match the expected output, you have a bug - fix it before submitting.
"""

SOLVER_INPUT_TEMPLATE = """# Current Problem:
{input_text}

---

# Lessons from Previous Attempts (if any):
{knowledge_items}

# Reference Solution (current best):
{reference_solution}

---

## Instructions (Rollout {rollout_index}/{num_rollouts}):
1. **Analyze** the problem (and previous attempts if shown above)
2. **Identify** if any previous solution needs fixes, or if you can come up with a better new approach
3. **Output** a complete, correct Python solution

Knowledge items are HINTS, not instructions. They may be:
- Correct insights that help
- Over-generalizations that hurt
- Outdated from earlier failed attempts

Always verify knowledge against the problem requirements.

## OUTPUT FORMAT:

First, provide a brief analysis of the problem and the lessons learned from previous attempts. The reference solution (if shown) may have bugs. Analyze it carefully, identify potential issues, and produce an improved solution.

Then output your solution in this EXACT format:

```python
# ... your complete solution code ...
```

The ```python and ``` fences are REQUIRED. Do not skip them.
"""

SOLVER_WITH_STRATEGY_INPUT_TEMPLATE = """# Current Problem:
{input_text}

---

# Lessons from Previous Attempts (if any):
{knowledge_items}

# Reference Solution (current best):
{reference_solution}

---

## Suggested Strategy (Rollout {rollout_index}/{num_rollouts})
**Key Technique**: {strategy_key_technique}
**What to Try**: {strategy_what_to_try}

Try implementing your assigned strategy.

## OUTPUT FORMAT:

First, provide a brief analysis of the problem and how the strategy applies.
Then output your solution in this EXACT format:

```python
# ... your complete solution code ...
```

The ```python and ``` fences are REQUIRED. Do not skip them.
"""

# ========== KNOWLEDGE MANAGER AGENT PROMPTS ==========

KNOWLEDGE_MANAGER_SYSTEM = """You are a technical reviewer analyzing competitive programming solutions.

Your job:
1. Rank solutions by CORRECTNESS first, then EFFICIENCY
2. Extract SPECIFIC, ACTIONABLE lessons from failures (not generic tips)
3. DEDUPLICATE insights - don't repeat what's already in knowledge base
4. Update the knowledge base with the best solution

DO NOT solve the problem yourself. Focus only on evaluation and knowledge curation.
IMPORTANT: Call update_knowledge exactly ONCE at the end."""

KNOWLEDGE_MANAGER_INPUT_TEMPLATE = """## Problem:
{problem_description}

## Existing Knowledge:
{knowledge_items}

## Candidate Solutions:
{solutions_block}

---

## Tasks:

### 1. Rank Solutions
Evaluate each solution. Criteria: Correctness

### 2. Extract SPECIFIC Failure Insights (CRITICAL - Read Carefully)

Compare each solution to the best one you identified.

**ONLY add insights that are:**
- NOT already covered by existing knowledge entries
- ACTIONABLE - tells you exactly what to fix, or to pay attention to

Reflect on the following:
**PATTERN**: What algorithmic pattern solves this? (1 sentence)

**WHEN TO USE**: What problem characteristics signal this pattern? (2-3 bullet points)
- List structural clues that would help identify similar problems
- Focus on input characteristics and what's being optimized

**KEY INSIGHT**: The core algorithmic insight (1-2 sentences)
- State the non-obvious realization that makes the solution work
- NOT the solution steps, but WHY it works

**OBSERVED MISTAKES** (from provided solutions only):
- List ONLY mistakes you actually saw in the incorrect solutions above
- Do NOT list hypothetical mistakes
- For each, explain what went wrong and why

IMPORTANT: Do NOT describe step-by-step solution procedures.

### 3. Knowledge Cleanup & Consolidation
- Keep at most 3-5 HIGH-VALUE entries (specific bugs/fixes, not generic tips)

### 4. Call update_knowledge EXACTLY ONCE

Arguments:
- problem_id: "problem_{problem_idx}"
- problem_description: One-line summary
- selected_solution_id: The ID of the best solution (0, 1, 2, 3, etc. matching the Solution numbers above)
- reflection: Specific, non-duplicate insights. It should help you solve it better next time.
- disable_knowledge_id: ID of ONE entry to disable (pick the most redundant/unhelpful one), or omit if none

IMPORTANT:
1. Call update_knowledge exactly ONCE (not zero, not multiple times)
2. Prefer NO reflection over DUPLICATE/GENERIC reflection
"""

KNOWLEDGE_MANAGER_WITH_STRATEGY_INPUT_TEMPLATE = """## Problem:
{problem_description}

## Existing Knowledge:
{knowledge_items}

{strategy_performance_block}

{strategy_history_block}

## Candidate Solutions:
{solutions_block}

---

## Tasks:

### 1. Rank Solutions
Evaluate each solution. Criteria: Correctness > Time Complexity > Space Complexity > Code Quality
Note: Solutions are presented in random order. Select the best one based on quality, not position.

### 2. Extract SPECIFIC Failure Insights

Compare each solution to the best one you identified.

**ONLY add insights that are:**
- NOT already covered by existing knowledge entries
- ACTIONABLE - tells you exactly what to fix, or to pay attention to

Reflect on the following:
**PATTERN**: What algorithmic pattern solves this? (1 sentence)

**WHEN TO USE**: What problem characteristics signal this pattern? (2-3 bullet points)
- List structural clues that would help identify similar problems
- Focus on input characteristics and what's being optimized

**KEY INSIGHT**: The core algorithmic insight (1-2 sentences)
- State the non-obvious realization that makes the solution work
- NOT the solution steps, but WHY it works

**OBSERVED MISTAKES** (from provided solutions only):
- List ONLY mistakes you actually saw in the incorrect solutions above
- Do NOT list hypothetical mistakes
- For each, explain what went wrong and why

IMPORTANT: Do NOT describe step-by-step solution procedures.

### 3. Generate {num_rollouts} NEW Strategies for Next Round (CRITICAL)

You MUST generate {num_rollouts} NEW strategies based on what you learned this round.

**Analyze the performance data above:**
- Which strategies WORKED? Why did they succeed?
- Which strategies FAILED? What was wrong with the approach?
- What patterns do you see in the successful vs failed attempts?

**Strategy Generation Rules:**
1. **DO NOT repeat failed strategies** - If a strategy got <50% pass rate, do NOT use it again
2. **Refine successful strategies** - If a strategy got >80% pass rate, try variations/improvements
3. **Try fundamentally different approaches** - If all strategies failed, pivot to a new paradigm
4. **Be SPECIFIC** - Don't just say "optimize"; say exactly what to optimize and how

Each strategy MUST have:
- **key_technique**: The algorithm or technique name (e.g., "Dynamic Programming", "Binary Search", "Greedy", "Two Pointers")
- **what_to_try**: SPECIFIC approach to implement (not generic advice)

**Good example:**
- key_technique: "Monotonic Stack"
- what_to_try: "Use a decreasing monotonic stack to track the next greater element, process from right to left"

**Bad example (too vague):**
- key_technique: "Optimization"
- what_to_try: "Make the solution faster"

### 4. Call update_knowledge EXACTLY ONCE

Arguments:
- problem_id: "problem_{problem_idx}"
- problem_description: One-line summary
- selected_solution_id: The ID of the best solution (1, 2, 3, etc. matching the Solution numbers above)
- reflection: Specific, non-duplicate insights
- strategies: A list of {num_rollouts} strategy objects for the NEXT round

Example strategies value:
[
  {{"key_technique": "Dynamic Programming", "what_to_try": "Use bottom-up DP with state (i, prev_taken) where i is current index"}},
  {{"key_technique": "Binary Search + Greedy", "what_to_try": "Binary search on answer, greedily verify if target sum is achievable"}}
]

IMPORTANT:
1. Call update_knowledge exactly ONCE with ALL arguments
2. The strategies list MUST contain exactly {num_rollouts} strategies
3. Strategies are for the NEXT round - make them different from this round's failed strategies
"""

KNOWLEDGE_MANAGER_WITHONLY_STRATEGY_INPUT_TEMPLATE = """## Problem:
{problem_description}

---

## Task

### 1. Generate {num_rollouts} Diverse Strategies for Next Round
Each strategy should have:
- **key_technique**: The algorithm or technique name (e.g., "Dynamic Programming", "Binary Search", "Greedy")
- **what_to_try**: Specific approach to implement

Make sure each strategy is DISTINCT and represents a different approach that could solve the problem correctly.

### 2. Call update_knowledge EXACTLY ONCE

Arguments:
- problem_id: "problem_{problem_idx}"
- problem_description: One-line summary
- strategies: A list of strategy objects, each with "key_technique" and "what_to_try" keys

Example strategies value:
[
  {{"key_technique": "Dynamic Programming", "what_to_try": "Use bottom-up DP with state array"}},
  {{"key_technique": "Binary Search", "what_to_try": "Binary search on the answer"}}
]

IMPORTANT: Call update_knowledge exactly ONCE with ALL arguments including strategies.
"""

# ========== TEST GENERATOR AGENT PROMPTS ==========

TEST_GENERATOR_SYSTEM = """You are a test engineer designing test cases for competitive programming problems.

Your job:
1. Analyze the problem description to understand requirements
2. Generate comprehensive test cases covering edge cases and common bugs
3. Execute these tests against candidate solutions
4. Report results objectively

You MUST call execute_generated_tests exactly ONCE with your test cases.
Do NOT solve the problem. Focus ONLY on testing."""

TEST_GENERATOR_INPUT_TEMPLATE = """## Problem:
{problem_description}

## Number of Candidate Solutions: {num_solutions}

## Candidate Solutions Code:
{solutions_code}

---

## Your Task: Generate and Execute Test Cases

### Step 1: Analyze the Problem and Solutions

Read the problem carefully and identify:
- Input format and constraints
- Output format and requirements
- Edge cases (empty input, single element, boundary values)
- Common bug scenarios (off-by-one, overflow, wrong comparison)

**Critically, analyze the candidate solutions above to understand:**
- Different algorithmic approaches used by each solution
- Potential bugs or weaknesses in each implementation
- Where solutions might diverge in behavior (e.g., edge cases one handles but another doesn't)

### Step 2: Generate Discriminating Test Cases

Create test cases that **differentiate between solutions** - inputs where different solutions may produce different outputs. This helps identify which solutions are correct and which have bugs.

**Prioritize test cases that:**
1. **Set solutions apart**: Target specific differences in logic between solutions
2. **Expose potential bugs**: Off-by-one errors, boundary conditions, edge cases each solution handles differently
3. **Test algorithmic correctness**: Cases where different approaches might yield different results
4. **Include examples**: From the problem description (for baseline validation)
5. **Cover edge cases**: Empty input, single element, maximum constraints, etc.

You can generate as many test cases as needed.

### Step 3: Format and Execute

Format your test cases as a JSON array:
```json
[
  {{"input": "...", "expected_output": "..."}},
  {{"input": "...", "expected_output": "..."}}
]
```

**CRITICAL - Input format** (this is the most common mistake!):
- For call-based (fn_name provided): Each argument on a SEPARATE LINE, each line is valid JSON.
  - Example for `def solution(nums: List[int], target: int)`:
    - Input: `"[1, 2, 3]\\n6"` (first line is nums, second line is target)
    - NOT: `"[[1, 2, 3], 6]"` (this is WRONG!)
  - Example for `def solution(s: str)`:
    - Input: `"\\"hello\\""` (just the string argument as JSON)
- For stdin-based (fn_name is null): Exact stdin string, e.g., `"3\\n1 2 3"`

**Expected output format**:
- For call-based: The expected return value as JSON string, e.g., `"6"` or `"[1, 2]"` or `"true"`
- For stdin-based: The expected stdout string exactly as printed

### Step 4: Call execute_generated_tests

Arguments:
- problem_id: "problem_{problem_idx}"
- test_cases_json: Your JSON array of test cases (as a string)
- fn_name: {fn_name_arg}

IMPORTANT: Call execute_generated_tests EXACTLY ONCE with all your test cases.
"""

KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_TEMPLATE = """## Problem:
{problem_description}

## Existing Knowledge:
{knowledge_items}

## Candidate Solutions:
{solutions_block}

## Test Execution Results (from Test Generator):
{test_results}

---

## Tasks:

### 1. Rank Solutions by Your Own Criteria

**CRITICAL**: Do not use the test execution results above as the primary ranking criterion, they are just for reference.
- Select the solution with the potential to be the best based on your own analysis
- The test cases are generated by another agent and may be incomplete

### 2. Extract SPECIFIC Failure Insights

**Analyze each failure type carefully:**

**Wrong Answer failures:**
- Compare the actual output vs expected output
- Identify the algorithmic bug or edge case that caused the discrepancy
- Note the specific input that triggered the failure

**Time Limit Exceeded (TLE) failures:**
- Identify which input size or pattern caused slowness
- Consider if the algorithm complexity is too high
- Look for nested loops, recursion without memoization, or unnecessary recomputation

**Runtime Error failures:**
- Check for array out-of-bounds, null/None access, or division by zero
- Consider edge cases like empty inputs or boundary values

From the test errors above, identify:
- What bugs caused failures?
- What edge cases weren't handled?
- What common mistakes were made?

**ONLY add insights that are:**
- NOT already covered by existing knowledge entries
- ACTIONABLE - tells you exactly what to fix

### 3. Call update_knowledge EXACTLY ONCE

Arguments:
- problem_id: "problem_{problem_idx}"
- problem_description: One-line summary
- selected_solution_id: The ID of the best solution (highest pass_rate from test results)
- reflection: Specific bug patterns observed in failing solutions

IMPORTANT: Trust the test results. The solution with highest pass_rate is the best solution.
"""


KNOWLEDGE_MANAGER_WITH_TEST_RESULTS_AND_STRATEGY_TEMPLATE = """## Problem:
{problem_description}

## Existing Knowledge:
{knowledge_items}

{strategy_performance_block}

{strategy_history_block}

## Candidate Solutions:
{solutions_block}

## Test Execution Results (from Test Generator):
{test_results}

---

## Tasks:

### 1. Rank Solutions by Your Own Criteria

**CRITICAL**: Do not use the test execution results above as the primary ranking criterion, they are just for reference.
- Select the solution with the potential to be the best based on their correctness first, then efficiency
- The test cases are generated by another agent and may be incomplete

### 2. Extract SPECIFIC Failure Insights

**Analyze each failure type carefully:**

**Wrong Answer failures:**
- Compare the actual output vs expected output shown above
- Identify the algorithmic bug or edge case that caused the discrepancy
- Note the specific input that triggered the failure

**Time Limit Exceeded (TLE) failures:**
- Identify which input size or pattern caused slowness
- Consider if the algorithm complexity is too high
- Look for nested loops, recursion without memoization, or unnecessary recomputation

**Runtime Error failures:**
- Check for array out-of-bounds, null/None access, or division by zero
- Consider edge cases like empty inputs or boundary values

From the test errors above, identify:
- What bugs caused failures?
- What edge cases weren't handled?
- What common mistakes were made?
- Ignore the cases when the test case itself is faulty.

**ONLY add insights that are:**
- NOT already covered by existing knowledge entries
- ACTIONABLE - tells you exactly what to fix

Include for each insight:
**PATTERN**: What algorithmic pattern solves this? (1 sentence)

**WHEN TO USE**: What problem characteristics signal this pattern? (2-3 bullet points)
- List structural clues that would help identify similar problems
- Focus on input characteristics and what's being optimized

**KEY INSIGHT**: The core algorithmic insight (1-2 sentences)
- State the non-obvious realization that makes the solution work
- NOT the solution steps, but WHY it works

**OBSERVED MISTAKES** (from test failures):
- List bugs you identified from the test execution errors above
- For each, explain what went wrong and why (using input/expected/actual data from test results)

### 3. Generate {num_rollouts} NEW Strategies for Next Round (CRITICAL)

You MUST generate {num_rollouts} NEW strategies based on what you learned this round.

**Analyze the performance data above:**
- Which strategies WORKED? Why did they succeed?
- Which strategies FAILED? What was wrong with the approach?
- What patterns do you see in the successful vs failed attempts?

**Strategy Generation Rules:**
1. **DO NOT repeat failed strategies** - If a strategy got <50% pass rate, do NOT use it again
2. **Refine successful strategies** - If a strategy got >80% pass rate, try variations/improvements
3. **Try fundamentally different approaches** - If all strategies failed, pivot to a new paradigm
4. **Be SPECIFIC** - Don't just say "optimize"; say exactly what to optimize and how

Each strategy MUST have:
- **key_technique**: The algorithm or technique name (e.g., "Dynamic Programming", "Binary Search", "Greedy", "Two Pointers")
- **what_to_try**: SPECIFIC approach to implement (not generic advice)

**Good example:**
- key_technique: "Monotonic Stack"
- what_to_try: "Use a decreasing monotonic stack to track the next greater element, process from right to left"

**Bad example (too vague):**
- key_technique: "Optimization"
- what_to_try: "Make the solution faster"

### 4. Call update_knowledge EXACTLY ONCE

Arguments:
- problem_id: "problem_{problem_idx}"
- problem_description: One-line summary
- selected_solution_id: The ID of the best solution (highest pass_rate from test results)
- reflection: Specific bug patterns observed in failing solutions
- strategies: A list of {num_rollouts} strategy objects for the NEXT round

Example strategies value:
[
  {{"key_technique": "Dynamic Programming", "what_to_try": "Use bottom-up DP with state (i, prev_taken) where i is current index"}},
  {{"key_technique": "Binary Search + Greedy", "what_to_try": "Binary search on answer, greedily verify if target sum is achievable"}}
]

IMPORTANT:
1. Trust the test results - the solution with highest pass_rate is the best solution
2. Call update_knowledge exactly ONCE with ALL arguments
3. The strategies list MUST contain exactly {num_rollouts} strategies
4. Strategies are for the NEXT round - make them different from this round's failed strategies
"""