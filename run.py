import argparse
import sys
import torch
import json
from medusa.model.medusa_model import MedusaModel
from fastchat.serve.cli import SimpleChatIO

def main(args):
    chatio = SimpleChatIO(args.multiline)
    model = MedusaModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = model.get_tokenizer()

    if args.input is not None:
        prompt = args.input
    elif args.input_file is not None:
        with open(args.input_file, 'r') as pf:
                prompt = pf.read()
    else:
         print("please provide your prompt to LLM with either --input or as file path with --input_file")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
            model.base_model.device
        )
    input_ids_len = input_ids.size(1)

    outputs = chatio.stream_output(model.medusa_generate(
            input_ids,
            temperature=args.temperature,
            max_steps=args.max_steps,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--input_file",
        type=str
    )
    parser.add_argument(
        "--input",
        type=str
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)