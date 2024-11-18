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
from needle_in_a_haystack.prompt import Prompter

INT_MAX = torch.iinfo(torch.int64).max

from torch.profiler import profile, record_function, ProfilerActivity

activities = [ProfilerActivity.CUDA]
prof_file = "/home/seliny2/Medusa/profiling/pytorch/amd/medusa/model-inference-decode.csv"
    
def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        with profile(activities=activities, with_stack=True, with_flops=True, with_modules=True, profile_memory=True, record_shapes=True) as prof1:
            with record_function("model_load"):
        
        # torch.cuda.memory._record_memory_history(
        #     max_entries=INT_MAX
        # )
       
                model = MedusaModel.from_pretrained(
                    args.model,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                )

                tokenizer = model.get_tokenizer()

        # torch.cuda.memory._record_memory_history(enabled=None)
        # try:
        #     torch.cuda.memory._dump_snapshot(f"{snapshot}-nvidia.pickle")
        # except Exception as e:
        #     print(f"Failed to capture memory snapshot {e}")

        conv = None

        def new_chat():
            return get_conversation_template(args.model)

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        #while True:
        if not conv:
            conv = new_chat()
        #while True:
        if not conv:
            conv = new_chat()

        try:
            print("starting my prompter...")
            # with profile(activities=activities, with_stack=True, with_flops=True, with_modules=True, profile_memory=True, record_shapes=True) as prof2:
            #     with record_function("prompter"):
            # torch.cuda.memory._record_memory_history(
            #     max_entries=INT_MAX
            # )
            prompter = Prompter(
                tokenizer
            )
            context_len=1000
            context = prompter.generate_context(context_len, 50)
            inp = prompter.generate_prompt(context, context_len, 50)

            # torch.cuda.memory._record_memory_history(enabled=None)
            # try:
            #     torch.cuda.memory._dump_snapshot(f"{snapshot}-prompter.pickle")
            # except Exception as e:
            #     logger.error(f"Failed to capture memory snapshot {e}")

            print(inp)

        except EOFError:
            inp = ""
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!load <filename>")
                # continue
            else:
                filename = args[1]
            if len(args) != 2:
                print("usage: !!load <filename>")
                # continue
            else:
                filename = args[1]

            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    # continue
            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    # continue

            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)
            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)

            conv = get_conv_template(new_conv["template_name"])
            conv.set_system_message(new_conv["system_message"])
            conv.messages = new_conv["messages"]
            reload_conv(conv)
            # continue
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        try:
            chatio.prompt_for_output(conv.roles[1])
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                model.base_model.device
            )
            
            with profile(activities=activities, with_stack=True, with_flops=True, with_modules=True, profile_memory=True, record_shapes=True) as prof3:
                with record_function("inference"):

            # torch.cuda.memory._record_memory_history(
            #     max_entries=INT_MAX
            # )
                    start_time = time.time()  # Record the start time
                    outputs = chatio.stream_output(
                        model.medusa_generate(
                            input_ids,
                            temperature=args.temperature,
                            max_steps=32,
                            context_len=0,
                        )
                    )
                    end_time = time.time()  # Record the end time
            # Stop recording memory snapshot history.
            

            # torch.cuda.memory._record_memory_history(enabled=None)
            # try:
            #     torch.cuda.memory._dump_snapshot(f"{snapshot}-nvidia.pickle")
            # except Exception as e:
            #     print(f"Failed to capture memory snapshot {e}")
            
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Elapsed time: {elapsed_time:.3f} seconds")
            conv.update_last_message(outputs.strip())

            with open(prof_file, 'w') as pf:
                pf.write(prof1.key_averages().table())
                # pf.write(prof2.key_averages().table())
                pf.write(prof3.key_averages().table())

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
                reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":

    print(INT_MAX)
    torch.cuda.empty_cache()

    # torch.cuda.memory._record_memory_history(
    #     max_entries=INT_MAX
    # )
    
    # with profile(activities=activities, with_stack=True, with_flops=True, with_modules=True, profile_memory=True, record_shapes=True) as prof3:
    #     with record_function("inference"):

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
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)

    # with open(prof_file, 'a') as pf:
    #         pf.write(prof3.key_averages().table())
    # prof3.export_chrome_trace(trace)


    # torch.cuda.memory._record_memory_history(enabled=None)
    # try:
    #     torch.cuda.memory._dump_snapshot(f"{snapshot}.pickle")
    # except Exception as e:
    #     logger.error(f"Failed to capture memory snapshot {e}")
