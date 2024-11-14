
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

activities = [ProfilerActivity.CUDA]
prof_file = "prof_file.csv"
with profile(
        use_cuda=True, use_kineto=True
    ) as p:
        pass
with profile(activities=activities, use_cuda=True, profile_memory=True, record_shapes=True) as prof1:
    with record_function("model_load"):
        model = AutoModelForCausalLM.from_pretrained(
            "lmsys/vicuna-7b-v1.3",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_flash_attention_2=False
            )

        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")

        print("starting my prompter...")
        prompter = Prompter(
            tokenizer
        )
        context = prompter.generate_context(1000, 50)
        inp = prompter.generate_prompt(context, 1000, 50)
        print(inp)

        conv = get_conversation_template("lmsys/vicuna-7b-v1.3")
        conv.messages = []
        conv.append_message(conv.roles[0],inp )

        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof2:
    with record_function("tokenizer"):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        input_ids_len = input_ids.size(1)

with profile(activities=activities, use_cuda=True,profile_memory=True, record_shapes=True) as prof3:
    with record_function("inference"):
        outputs = model.generate(input_ids.cuda(), max_new_tokens=200, do_sample=False)
    
        print(tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True))

with open(prof_file, 'a') as pf:
    pf.write(prof1.key_averages().table())
    pf.write(prof2.key_averages().table())
    pf.write(prof3.key_averages().table())