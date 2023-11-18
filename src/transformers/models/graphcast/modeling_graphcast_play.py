# Curerently just a copy-paste of the demo ipynb

import dataclasses
# import datetime
import functools
# import math
# import re
from typing import Optional

# import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
# from IPython.display import HTML
# import ipywidgets as widgets
import haiku as hk
import jax
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import animation
import numpy as np
import xarray


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


# from google.colab import auth
# auth.authenticate_user()

gcs_client = storage.Client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")

# @title Plotting functions

# def select(
#     data: xarray.Dataset,
#     variable: str,
#     level: Optional[int] = None,
#     max_steps: Optional[int] = None
#     ) -> xarray.Dataset:
#   data = data[variable]
#   if "batch" in data.dims:
#     data = data.isel(batch=0)
#   if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
#     data = data.isel(time=range(0, max_steps))
#   if level is not None and "level" in data.coords:
#     data = data.sel(level=level)
#   return data

# def scale(
#     data: xarray.Dataset,
#     center: Optional[float] = None,
#     robust: bool = False,
#     ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
#   vmin = np.nanpercentile(data, (2 if robust else 0))
#   vmax = np.nanpercentile(data, (98 if robust else 100))
#   if center is not None:
#     diff = max(vmax - center, center - vmin)
#     vmin = center - diff
#     vmax = center + diff
#   return (data, matplotlib.colors.Normalize(vmin, vmax),
#           ("RdBu_r" if center is not None else "viridis"))

# def plot_data(
#     data: dict[str, xarray.Dataset],
#     fig_title: str,
#     plot_size: float = 5,
#     robust: bool = False,
#     cols: int = 4
#     ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

#   first_data = next(iter(data.values()))[0]
#   max_steps = first_data.sizes.get("time", 1)
#   assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

#   cols = min(cols, len(data))
#   rows = math.ceil(len(data) / cols)
#   figure = plt.figure(figsize=(plot_size * 2 * cols,
#                                plot_size * rows))
#   figure.suptitle(fig_title, fontsize=16)
#   figure.subplots_adjust(wspace=0, hspace=0)
#   figure.tight_layout()

#   images = []
#   for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
#     ax = figure.add_subplot(rows, cols, i+1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title)
#     im = ax.imshow(
#         plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
#         origin="lower", cmap=cmap)
#     plt.colorbar(
#         mappable=im,
#         ax=ax,
#         orientation="vertical",
#         pad=0.02,
#         aspect=16,
#         shrink=0.75,
#         cmap=cmap,
#         extend=("both" if robust else "neither"))
#     images.append(im)

#   def update(frame):
#     if "time" in first_data.dims:
#       td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
#       figure.suptitle(f"{fig_title}, {td}", fontsize=16)
#     else:
#       figure.suptitle(fig_title, fontsize=16)
#     for im, (plot_data, norm, cmap) in zip(images, data.values()):
#       im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

#   ani = animation.FuncAnimation(
#       fig=figure, func=update, frames=max_steps, interval=250)
#   plt.close(figure.number)
#   return HTML(ani.to_jshtml())


params_file_options = [
    name for blob in gcs_bucket.list_blobs(prefix="params/")
    if (name := blob.name.removeprefix("params/"))]  # Drop empty string.


# random_mesh_size = widgets.IntSlider(
#     value=4, min=4, max=6, description="Mesh size:")
# random_gnn_msg_steps = widgets.IntSlider(
#     value=4, min=1, max=32, description="GNN message steps:")
# random_latent_size = widgets.Dropdown(
#     options=[int(2**i) for i in range(4, 10)], value=32,description="Latent size:")
# random_levels = widgets.Dropdown(
#     options=[13, 37], value=13, description="Pressure levels:")


random_mesh_size = 4
random_gnn_msg_steps = 4
random_latent_size = 32
random_levels = 13




# params_file = widgets.Dropdown(
#     options=params_file_options,
#     description="Params file:",
#     layout={"width": "max-content"})

# source_tab = widgets.Tab([
#     widgets.VBox([
#         random_mesh_size,
#         random_gnn_msg_steps,
#         random_latent_size,
#         random_levels,
#     ]),
#     params_file,
# ])
# source_tab.set_title(0, "Random")
# source_tab.set_title(1, "Checkpoint")
# widgets.VBox([
#     source_tab,
#     widgets.Label(value="Run the next cell to load the model. Rerunning this cell clears your selection.")
# ])



# source = source_tab.get_title(source_tab.selected_index)
source = "Random"


if source == "Random":
  params = None  # Filled in below
  state = {}
  model_config = graphcast.ModelConfig(
      resolution=0,
      # mesh_size=random_mesh_size.value,
      mesh_size=random_mesh_size,
      # latent_size=random_latent_size.value,
      latent_size=random_latent_size,
      # gnn_msg_steps=random_gnn_msg_steps.value,
      gnn_msg_steps=random_gnn_msg_steps,
      hidden_layers=1,
      radius_query_fraction_edge_length=0.6)
  task_config = graphcast.TaskConfig(
      input_variables=graphcast.TASK.input_variables,
      target_variables=graphcast.TASK.target_variables,
      forcing_variables=graphcast.TASK.forcing_variables,
      # pressure_levels=graphcast.PRESSURE_LEVELS[random_levels.value],
      pressure_levels=graphcast.PRESSURE_LEVELS[random_levels],
      input_duration=graphcast.TASK.input_duration,
  )
else:
  assert source == "Checkpoint"
  with gcs_bucket.blob(f"params/{params_file.value}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
  params = ckpt.params
  state = {}

  model_config = ckpt.model_config
  task_config = ckpt.task_config
  print("Model description:\n", ckpt.description, "\n")
  print("Model license:\n", ckpt.license, "\n")

model_config


dataset_file_options = [
    name for blob in gcs_bucket.list_blobs(prefix="dataset/")
    if (name := blob.name.removeprefix("dataset/"))]  # Drop empty string.

def data_valid_for_model(
    file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
  file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  return (
      model_config.resolution in (0, float(file_parts["res"])) and
      len(task_config.pressure_levels) == int(file_parts["levels"]) and
      (
          ("total_precipitation_6hr" in task_config.input_variables and
           file_parts["source"] in ("era5", "fake")) or
          ("total_precipitation_6hr" not in task_config.input_variables and
           file_parts["source"] in ("hres", "fake"))
      )
  )


dataset_file_options = [
    (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
    for option in dataset_file_options
    if data_valid_for_model(option, model_config, task_config)
]

dataset_file = dataset_file_options[0][1]


# dataset_file = widgets.Dropdown(
#     options=[
#         (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
#         for option in dataset_file_options
#         if data_valid_for_model(option, model_config, task_config)
#     ],
#     description="Dataset file:",
#     layout={"width": "max-content"})
# widgets.VBox([
#     dataset_file,
#     widgets.Label(value="Run the next cell to load the dataset. Rerunning this cell clears your selection and refilters the datasets that match your model.")
# ])


# @title Load weather data

# if not data_valid_for_model(dataset_file.value, model_config, task_config):
if not data_valid_for_model(dataset_file, model_config, task_config):
  raise ValueError(
      "Invalid dataset file, rerun the cell above and choose a valid dataset file.")

# with gcs_bucket.blob(f"dataset/{dataset_file.value}").open("rb") as f:
with gcs_bucket.blob(f"dataset/{dataset_file}").open("rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

# print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.value.removesuffix(".nc")).items()]))
print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))

example_batch


# plot_example_variable = widgets.Dropdown(
#     options=example_batch.data_vars.keys(),
#     value="2m_temperature",
#     description="Variable")
# plot_example_level = widgets.Dropdown(
#     options=example_batch.coords["level"].values,
#     value=500,
#     description="Level")
# plot_example_robust = widgets.Checkbox(value=True, description="Robust")
# plot_example_max_steps = widgets.IntSlider(
#     min=1, max=example_batch.dims["time"], value=example_batch.dims["time"],
#     description="Max steps")

# widgets.VBox([
#     plot_example_variable,
#     plot_example_level,
#     plot_example_robust,
#     plot_example_max_steps,
#     widgets.Label(value="Run the next cell to plot the data. Rerunning this cell clears your selection.")
# ])


# @title Choose training and eval data to extract
# train_steps = widgets.IntSlider(
#     value=1, min=1, max=example_batch.sizes["time"]-2, description="Train steps")
# eval_steps = widgets.IntSlider(
#     value=example_batch.sizes["time"]-2, min=1, max=example_batch.sizes["time"]-2, description="Eval steps")

# widgets.VBox([
#     train_steps,
#     eval_steps,
#     widgets.Label(value="Run the next cell to extract the data. Rerunning this cell clears your selection.")
# ])

train_steps = 1
eval_steps = example_batch.sizes["time"]-2


# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    # example_batch, target_lead_times=slice("6h", f"{train_steps.value*6}h"),
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    # example_batch, target_lead_times=slice("6h", f"{eval_steps.value*6}h"),
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)


# @title Load normalization data

with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()


# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)

predictions






