import argparse
import json
import re
import time

from datasets import load_dataset
from typing import List
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_code_blocks(response: str) -> str:
    """Extract code blocks from markdown-formatted response.

    Looks for code blocks in the format:
    ```python / ```python3 / ```py / ```
    code here
    ```

    Args:
        response: String containing markdown code blocks

    Returns:
        List of dicts with 'code' key containing the extracted code
    """
    code_blocks = []

    # Pattern to match Python code block prefixes or no prefix
    # Handles: ```python, ```python3, ```py, or plain ```
    pattern = r'```(?:python3?|py)?\s*\n(.*?)```'

    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
        code = match.strip()
        if code:
            code_blocks.append(code)

    return "\n".join(code_blocks)

def build_prompt(original_code: str) -> str:
    """Build an optimization prompt using a simple on-disk template with safe token replacement.

    The template must contain tokens {{LANGUAGE_NAME}}, {{SOURCE_FILENAME}}, {{BASELINE_SECONDS}},
    and markers :::HEADER::: and :::CODE::: which will be replaced verbatim.
    Falls back to an inline template if the file is missing.
    """

    return (
        f"You are an expert python performance engineer.\n\n"
        f"Task: Provide an optimized implementation of the following function.\n"
        f"- Do not change the function signature expected by the harness.\n"
        f"- Return only the code in a single fenced block.\n\n"
        f"--- function definition ---\n{original_code}\n"
    )

def build_code_opt_prompt(original_code: str) -> str:
    """Build an optimization prompt using a simple on-disk template with safe token replacement.

    The template must contain tokens {{LANGUAGE_NAME}}, {{SOURCE_FILENAME}}, {{BASELINE_SECONDS}},
    and markers :::HEADER::: and :::CODE::: which will be replaced verbatim.
    Falls back to an inline template if the file is missing.
    """

    return (
        f"You are an expert python performance engineer.\n\n"
        f"Task: Optimize the provided for speed while preserving exact behavior and I/O.\n"
        f"- Do not change the function signature expected by the harness.\n"
        f"- Provide a full replacement for this code.\n"
        f"- Return only the code in a single fenced block.\n\n"
        f"--- current source ---\n{original_code}\n"
    )

def llm_generate(client, model: str, prompt: str) -> str:
    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            temperature=0.6,
            top_p=0.95,
        )

    text = completion.choices[0].message.content
    return text

def generate_solutions(problems: List[dict], model: str, num_completions: int, baseline_codes: list[str] = []) -> List[List[str]]:
    """Generate solutions for a list of problems.

    Args:
        problems: List of problem dictionaries with 'prompt' field
        model: Model name to use for generation
        num_completions: Number of solutions to generate per problem

    Returns:
        List[List[str]]: List of lists where result[i] contains solutions for problem i
    """
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="ryan123",
    )

    # Initialize result list with empty lists for each problem
    generations = [[] for _ in range(len(problems))]
    num_none = 0

    start = time.time()

    def process_problem(problem: dict, num_completions: int, idx: int, baseline_codes: list[str]) -> tuple[list[str], int]:
        func_defn = problem["prompt"]
        if baseline_codes:
            prompt = build_code_opt_prompt(baseline_codes[idx])
        else:
            prompt = build_prompt(func_defn)
        responses = []
        for _ in range(num_completions):
            response = llm_generate(client, model, prompt)
            code = extract_code_blocks(response)
            if code.strip():
                responses.append(code)
            else:
                print(f"Problem {idx}: No code blocks found in response")
        return responses, idx

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_problem, problem, num_completions, idx, baseline_codes)
                   for idx, problem in enumerate(problems)]
        for future in as_completed(futures):
            solutions, idx = future.result()
            generations[idx] = solutions
            if not solutions:
                num_none += 1

    end = time.time()

    print(f"Problems with no solutions: {num_none}/{len(problems)}, time: {end - start:.2f}s")
    return generations

def main():
    """Parse command line arguments and generate solutions for enamel benchmark."""
    parser = argparse.ArgumentParser(description="Generate solutions for enamel benchmark")
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output JSON file path for generations (will be List[List[str]] format)"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name for generation"
    )
    parser.add_argument(
        "--num-completions",
        type=int,
        default=1,
        help="Number of solutions to generate per problem (default: 1)"
    )
    args = parser.parse_args()

    # Load enamel dataset from CSV
    dataset = load_dataset("q-rz/enamel", split="ENAMEL_HumanEval")
    problems = []
    baseline_codes = []
    with open("samples/humaneval-canonical.json") as f:
        d = json.load(f)
    
    for idx, obj in enumerate(dataset):
        problems.append(obj)
        baseline_codes.append(d[idx][0])

    # Generate solutions: returns List[List[str]]
    generations = generate_solutions(problems, args.model, args.num_completions)

    # Save as JSON in List[List[str]] format
    with open(args.output, "w") as f:
        json.dump(generations, f, indent=2)

    print(f"Saved {len(generations)} problems to {args.output}")

if __name__ == "__main__":
    main()
