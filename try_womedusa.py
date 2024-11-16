
import os
# CUDAVISIBLE DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
from fastchat.model import load_model, get_conversation_template
from needle_in_a_haystack.prompt import Prompter
from torch.profiler import profile, record_function, ProfilerActivity
import time
from fastchat.serve.cli import SimpleChatIO
from fastchat.model.model_adapter import get_conversation_template

INT_MAX = torch.iinfo(torch.int64).max
# activities = [ProfilerActivity.CUDA]
# prof_file = "prof_file.csv"
# with profile(
#         use_cuda=True, use_kineto=True
#     ) as p:
#         pass
activities = [ProfilerActivity.CUDA]
prof_file = "/home/seliny2/Medusa/profiling/pytorch/amd/base/model-inference-decode.csv"

# "lmsys/vicuna-7b-v1.3"
with profile(activities=activities, use_cuda=True, profile_memory=True, record_shapes=True) as prof1:
    with record_function("model_load"):

# torch.cuda.memory._record_memory_history(
#         max_entries=INT_MAX
#     )

        model = LlamaForCausalLM.from_pretrained(
            "lmsys/vicuna-7b-v1.3",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_flash_attention_2=False,
            attn_implementation="eager"
            )

        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")

    # torch.cuda.memory._record_memory_history(enabled=None)
    # try:
    #     torch.cuda.memory._dump_snapshot("~/Medusa/profiling/pytorch/amd/base/model.pickle")
    # except Exception as e:
    #     print(f"Failed to capture memory snapshot {e}")

print("starting my prompter...")
prompter = Prompter(
    tokenizer
)
context = prompter.generate_context(500, 50)
inp = prompter.generate_prompt(context, 500, 50)
print(inp)
chatio = SimpleChatIO()
conv = get_conversation_template("lmsys/vicuna-7b-v1.3")
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()


# with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof2:
#     with record_function("tokenizer"):
input_ids = tokenizer.encode(prompt, return_tensors='pt')
input_ids_len = input_ids.size(1)

with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof3:
    with record_function("inference"):
# torch.cuda.memory._record_memory_history(
    #     max_entries=INT_MAX
    # )
        start_time = time.time()  # Record the start time
        outputs = model.generate(input_ids.cuda(), max_new_tokens=32, do_sample=False)
        end_time = time.time()  # Record the end time
# torch.cuda.memory._record_memory_history(enabled=None)
# try:
#     torch.cuda.memory._dump_snapshot("~/Medusa/profiling/pytorch/amd/base/inference.pickle")
# except Exception as e:
#     print(f"Failed to capture memory snapshot {e}")
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof2:
    with record_function("decode output"):
        print("Output\n:" , tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True))

with open(prof_file, 'a') as pf:
    pf.write(prof1.key_averages().table())
    pf.write(prof3.key_averages().table())
    pf.write(prof2.key_averages().table())