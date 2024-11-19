# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m medusa.inference.cli --model <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import os
import re
import sys
import torch
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from medusa.model.medusa_model import MedusaModel
import time
import yaml
# from needle_in_a_haystack.prompt import Prompter

# from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name or path.")
parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--input_file", type=str)

args = parser.parse_args()

# load model and tokenizer
model = MedusaModel.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    load_in_8bit=args.load_in_8bit,
    load_in_4bit=args.load_in_4bit,
)

tokenizer = model.get_tokenizer()

if args.input_file is not None:
    # load question from a file
    with open(args.input_file, 'r') as pf:
            question = pf.read()
else:
    # self-defined question
    question = "hiiiiiiiiiiii, how are you"

# start a new conversation
conv = get_conversation_template(args.model)
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# encode prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        model.base_model.device
)
input_len = input_ids.shape[1]

# start generating with medusa with initialization
# actual generation start from medusa_model.py line 319
outputs = model.medusa_generate(
                input_ids,
                temperature=args.temperature,
                max_steps=512,
)


# decode and output
response = ""
for output in outputs:
    response = output['text']
    time.sleep(0.01)
print("response:", response.strip())