from importlib.metadata import version
import warnings
import transformers
from medusa.model.modeling_llama_kv import forward as llama_forward_4_46, prepare_inputs_for_generation_llama

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_forward_4_46

