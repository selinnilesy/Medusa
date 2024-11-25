import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import (
    logging,
)
from transformers.cache_utils import Cache, DynamicCache



logger = logging.get_logger(__name__)

def initialize_past_key_values(model, context_len):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    past_key_values_data = torch.zeros(
            config.num_hidden_layers * 2,
            batch_size,
            config.num_key_value_heads,
            0,
            config.hidden_size // config.num_attention_heads,
            device=model.device,
            dtype=model.dtype,
        )

    print("context_len: ", context_len)
    print("max_position_embeddings:", config.max_position_embeddings)
    print("config.hidden_size // config.num_attention_heads:", config.hidden_size // config.num_attention_heads)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    # past_key_values = DynamicCache(config.num_hidden_layers)
    past_key_values = DynamicCache()
    for i in range(config.num_hidden_layers):
        past_key_values.update(
            past_key_values_data[2*i ],
            past_key_values_data[2*i + 1],
            i
        )

    return past_key_values, past_key_values_data, current_length_data