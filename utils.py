import re
import tensorflow as tf
import pickle

import numpy as np
import os

import torch.nn


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def print_params_structure(params, indent=0, parent_key=''):
    if isinstance(params, dict):
        for key, value in params.items():
            current_key = f"{parent_key}/{key}" if parent_key else key
            print(' ' * indent + f"{key}:")
            print_params_structure(value, indent + 2, current_key)
    elif isinstance(params, list):
        for idx, item in enumerate(params):
            print(' ' * indent + f"[{idx}]:")
            print_params_structure(item, indent + 2, f"{parent_key}[{idx}]")
    else:
        # 处理 NumPy 数组和其他数据类型
        if hasattr(params, 'dtype') and hasattr(params, 'shape'):
            print(' ' * indent + f"array: shape{tuple(params.shape)}, dtype={params.dtype}")
        else:
            print(' ' * indent + f"value: {str(params)[:50]} (type={type(params).__name__})")


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch. left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_to_gpt(gpt, params):
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.W_query.weight = assign(
            gpt.trf_blocks[b].attention.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].attention.W_key.weight = assign(
            gpt.trf_blocks[b].attention.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].attention.W_value.weight = assign(
            gpt.trf_blocks[b].attention.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.W_query.bias = assign(
            gpt.trf_blocks[b].attention.W_query.bias, q_b
        )
        gpt.trf_blocks[b].attention.W_key.bias = assign(
            gpt.trf_blocks[b].attention.W_key.bias, k_b
        )
        gpt.trf_blocks[b].attention.W_value.bias = assign(
            gpt.trf_blocks[b].attention.W_value.bias, v_b
        )

        gpt.trf_blocks[b].attention.out_proj.weight = assign(
            gpt.trf_blocks[b].attention.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].attention.out_proj.bias = assign(
            gpt.trf_blocks[b].attention.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )

        gpt.trf_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["b"]
        )

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["b"]
        )
    gpt.final_norm.scale = assign(
        gpt.final_norm.scale, params["g"]
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift, params["b"]
    )
    gpt.out_head.weight = assign(
        gpt.out_head.weight, params["wte"]
    )
