import argparse
import os

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple Mini-SGLang offline example.")
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Model path or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--attention-backend",
        default="fi",
        choices=["fi", "fa", "trtllm", "auto"],
        help="Attention backend to use. Defaults to FlashInfer to avoid TRTLLM architecture limits.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for each prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, attention_backend=args.attention_backend)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
