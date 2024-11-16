
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

# activities = [ProfilerActivity.CUDA]
# prof_file = "prof_file.csv"
# with profile(
#         use_cuda=True, use_kineto=True
#     ) as p:
#         pass


# "lmsys/vicuna-7b-v1.3"
# with profile(activities=activities, use_cuda=True, profile_memory=True, record_shapes=True) as prof1:
#     with record_function("model_load"):
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=False,
    attn_implementation="eager"
    )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

print("starting my prompter...")
prompter = Prompter(
    tokenizer
)
context = prompter.generate_context(500, 50)
inp = prompter.generate_prompt(context, 500, 50)
print(inp)

# with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof2:
#     with record_function("tokenizer"):
input_ids = tokenizer.encode(inp, return_tensors='pt')
input_ids_len = input_ids.size(1)

# with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof3:
#     with record_function("inference"):
outputs = model.generate(input_ids.cuda(), max_new_tokens=32, do_sample=False)
    
print("Output\n:" , tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True))

# with open(prof_file, 'a') as pf:
#     pf.write(prof1.key_averages().table())
#     pf.write(prof2.key_averages().table())
#     pf.write(prof3.key_averages().table())