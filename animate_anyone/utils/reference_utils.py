import os
from typing import Any, Tuple, Union

from safetensors import safe_open
from safetensors.torch import save_file

from ..models.unet_2d_condition import ReferenceHiddenStates

def _get_state_ids(key: str) -> Tuple[int, ...]:
    ids = [int(name.split("_")[-1]) for name in key.split(".")]
    return tuple(ids)

def save_reference_features(
    reference_hidden_states: ReferenceHiddenStates,
    save_path: Union[str, os.PathLike],
):
    tensors = {}
    for i, block_ref_states in enumerate(reference_hidden_states):
        for j, layer_ref_states in enumerate(block_ref_states):
            for k, tensor in enumerate(layer_ref_states):
                tensors[f"block_{i}.layer_{j}.tensor_{k}"] = tensor
    save_file(tensors, save_path)

def load_reference_features(file_path: Union[str, os.PathLike], device: Any) -> ReferenceHiddenStates:
    tensors = {}
    num_down_layers = 0
    num_mid_layers = 0
    num_up_layers = 0
    with safe_open(file_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to(device)

            block_id, layer_id, _ = _get_state_ids(k)
            match block_id:
                case 0:
                    num_down_layers = max(num_down_layers, layer_id + 1)
                case 1:
                    num_mid_layers = max(num_mid_layers, layer_id + 1)
                case 2:
                    num_up_layers = max(num_up_layers, layer_id + 1)
                case _:
                    raise ValueError(f"unexpected block id {block_id}")

    down_states = [() for _ in range(num_down_layers)]
    mid_states = [() for _ in range(num_mid_layers)]
    up_states = [() for _ in range(num_up_layers)]
    for key, tensor in tensors.items():
        block_id, layer_id, _ = _get_state_ids(key)
        match block_id:
            case 0:
                down_states[layer_id] += (tensor,)
            case 1:
                mid_states[layer_id] += (tensor,)
            case _:
                up_states[layer_id] += (tensor,)

    reference_hidden_states = [tuple(down_states), tuple(mid_states), tuple(up_states)]
    return reference_hidden_states
