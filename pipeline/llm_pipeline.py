"""responses = completion(
  # model="hosted_vllm/CobraMamba/Qwen3-32B-AWQ",
  model="hosted_vllm/openai/gpt-oss-120b",
  api_base="http://h01:8333/v1",
  # model="ollama/gpt-oss:20b",
  # api_base="http://h02:8222",
  messages = [{"role": "user", "content": "Who are you?"}]
)"""
#!/usr/bin/env python3
from vllm import LLM, SamplingParams

def main():
    # === Load model ===
    model_name = "openai/gpt-oss-20b"

    # Generation parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=300,
        stop=None,
    )

    # Initialize vLLM engine
    llm = LLM(
        model=model_name,
        dtype="bfloat16",   # or "float16" if bfloat16 unsupported
        tensor_parallel_size=1,  # increase if you have multiple GPUs
        gpu_memory_utilization=0.9,
    )

    # === Example data ===
    claim = "The Czech Republic has the highest beer consumption per capita in the world."
    evidence = """According to the World Beer Index 2021, the Czech Republic leads the world 
    in beer consumption per capita, averaging 468 beers per person per year."""

    prompt = f"""
### Instruction:
You are a fact-checking assistant. You receive a claim and several evidence passages.
Decide whether the claim is Supported, Refuted, or Not enough info based on the evidence.
Then, explain your reasoning step by step.

### Claim:
{claim}

### Evidence:
{evidence}

### Output format:
Label: [Supported / Refuted / Not enough info]
Explanation:
"""

    # === Generate ===
    outputs = llm.generate([prompt], sampling_params)

    print("=== Model Output ===")
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
