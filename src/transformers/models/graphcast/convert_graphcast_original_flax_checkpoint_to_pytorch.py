from transformers import GraphCastConfig, GraphCastModel
import numpy as np
import torch
import re

import jax
from jax import numpy as jnp

from transformers.utils import logging

logger = logging.get_logger(__name__)

CONFIGS = {
    "graphcast-small": {
        "hidden_size": 512,
    }
}

checkpoint_path = "/Users/amyroberts/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
checkpoint = np.load(checkpoint_path, allow_pickle=True)


config = GraphCastConfig()
hf_model = GraphCastModel(config)
pt_layer_names = sorted(list(hf_model.state_dict().keys()))

# def copy_params_to_hf_model(flax_params, hf_model):
#     hf_model_params = hf_model.state_dict()
#     for k, v in flax_params.items():
#         hf_model_params[k] = torch.tensor(v)
#     hf_model.load_state_dict(hf_model_params)
#     return hf_model


flax_params = {k.removeprefix("params"): v for k, v in checkpoint.items() if k.startswith("params")}

with open("flax_layer_names.txt", "w") as f:
    for param_name in sorted(list(flax_params.keys())):
        f.write(param_name + "\n")

with open("hf_layer_names.txt", "w") as f:
    for param_name in sorted(list(pt_layer_names)):
        f.write(param_name + "\n")


def replace_flax_name_with_hf_name(flax_name):
    new_name = flax_name
    new_name = new_name.replace("/", ".")
    new_name = new_name.replace("~_networks_builder.", "")
    new_name = new_name.replace("~.", "")
    new_name = new_name.replace(":", "")

    model_name_prefix, *rest = new_name.split(".")
    rest = ".".join(rest)
    new_name = rest

    new_name = new_name.replace("_", ".")
    new_name = new_name.replace("layer.norm", "layer_norm")
    new_name = new_name.replace("linear.", "linear_")
    new_name = new_name.replace(".grid2mesh.", ".")
    new_name = new_name.replace(".mesh2grid.", ".")
    new_name = new_name.replace("grid.nodes", "grid_nodes")
    new_name = new_name.replace("mesh.nodes", "mesh_nodes")
    new_name = new_name.replace("mesh.", "")

    # new_name = new_name.replace("mesh_nodes", "")
    new_name = new_name.replace("layer_normoffset", "layer_norm.bias") # FIXME - double check
    new_name = new_name.replace("layer_normscale", "layer_norm.weight") # FIXME - double check
    new_name = new_name.replace("nodes.mesh_nodes", "mesh_nodes") # FIXME - double check
    new_name = new_name.replace("layer_normscale", "layer_norm.weight") # FIXME - double check

    new_name = re.sub(r"nodes.([0-9]*).mesh_nodes", r"mesh_nodes.\1", new_name)
    new_name = re.sub(r"nodes.([0-9]*).grid_nodes", r"grid_nodes.\1", new_name)
    new_name = re.sub(r"linear_([0-9])b", r"linear_\1.bias", new_name)
    new_name = re.sub(r"linear_([0-9])w", r"linear_\1.weight", new_name)

    new_name = model_name_prefix + "." + new_name

    new_name = new_name.replace("grid2mesh_gnn.encoder.mesh_nodes.", "grid2mesh_gnn.encoder.nodes.mesh_nodes.")
    new_name = new_name.replace("mesh2grid_gnn.decoder.nodes.grid_nodes.mlp", "mesh2grid_gnn.decoder")
    return new_name

flax_params = {
    replace_flax_name_with_hf_name(k): v for k, v in flax_params.items()
}

def reshape_and_rename_keys(hf_model, flax_params):
    pt_model_dict = hf_model.state_dict()

    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    for jax_key, array_value in flax_params.items():

        if jax_key not in pt_model_dict:
            unexpected_keys.append(jax_key)
            continue

        if jax_key.endswith("weight"):
            array_value = array_value.T

        if array_value.shape != pt_model_dict[jax_key].shape:
            raise ValueError(
                f"Flax checkpoint seems to be incorrect. Weight {jax_key} was expected "
                f"to be of shape {pt_model_dict[jax_key].shape}, but is {array_value.shape}."
            )
        else:
            pt_model_dict[jax_key] = torch.from_numpy(array_value)
            missing_keys.remove(jax_key)

    print(f"unexpected_keys: {unexpected_keys}")
    print(f"missing_keys: {sorted(list(missing_keys))}")

    hf_model.load_state_dict(pt_model_dict)
    return hf_model


hf_model = reshape_and_rename_keys(hf_model, flax_params)



