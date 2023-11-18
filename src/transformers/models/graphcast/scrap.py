# import graphcast
import numpy as np


# from graphcast import ModelConfig, TaskConfig

# model_config = graphcast.ModelConfig(
#     resolution=0,
#     mesh_size=4,
#     latent_size=32,
#     gnn_msg_steps=4,
#     hidden_layers=1,
#     radius_query_fraction_edge_length=0.6,
#     mesh2grid_edge_normalization_factor=None
# )
# task_config = graphcast.TaskConfig(
#     input_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity', 'toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask'),
#     target_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity'),
#     forcing_variables=('toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos'),
#     pressure_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
#     input_duration='12h'
# )

# checkpoint_path = "/Users/amyroberts/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"

# checkpoint = np.load(checkpoint_path, allow_pickle=True)

# params = {k: v for k, v in checkpoint.items() if k.startswith("params")}
# target_config = {k: v for k, v in checkpoint.items() if k.startswith("target_config")}
# model_config = {k: v for k, v in checkpoint.items() if k.startswith("model_config")}


from transformers import GraphCastModel, GraphCastConfig

config = GraphCastConfig()
model = GraphCastModel(config)
