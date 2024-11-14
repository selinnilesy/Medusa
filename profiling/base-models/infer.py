from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time 

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-33b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-33b-v1.3"
                                             , torch_dtype=torch.float16
                                             , max_length = 10000
                                            ).to(device)

input_file = "/home/seliny2/Medusa/profiling/vicuna-prompts/prompt_500.txt"
with open(input_file, 'r') as pf:
    inp = pf.read()

input_ids = tokenizer.encode(inp, return_tensors='pt', padding=True).to(device)
input_ids_len = input_ids.size(1)

# Generate text
with torch.no_grad():
    start_time = time.time()
    outputs = model.generate(input_ids)
    end_time = time.time()
    elapsed_time = end_time - start_time

 
printable_out = tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
print(printable_out)
print(f"Elapsed time: {elapsed_time:.3f} seconds")

