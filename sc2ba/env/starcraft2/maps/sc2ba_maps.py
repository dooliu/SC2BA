from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class SC2BAMap(lib.Map):
    directory = "SC2BA_Maps"
    download = "https://github.com/dooliu/SC2BA/tree/main/sc2ba/env/starcraft2/maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "3m": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "red_units": {"marine": 3},
        "blue_units": {"marine": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "map_type": "marines",
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "8m": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "red_units": {"marine": 8},
        "blue_units": {"marine": 9},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "25m": {
        "n_agents": 25,
        "n_enemies": 25,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "red_units": {"marine": 25},
        "blue_units": {"marine": 25},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "5m_vs_6m": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "red_units": {"marine": 5},
        "blue_units": {"marine": 6},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "10m_vs_11m": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "red_units": {"marine": 10},
        "blue_units": {"marine": 11},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "MMM": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 10,
        "map_type": "MMM",
        "red_units": {"marine": 7, "marauder": 2, "medivac": 1},
        "blue_units": {"marine": 7, "marauder": 2, "medivac": 1},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "MMM2": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 10,
        "map_type": "MMM",
        "red_units": {"marine": 7, "marauder": 2, "medivac": 1},
        "blue_units": {"marine": 8, "marauder": 3, "medivac": 1},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "stalkers_and_zealots",
        "red_units": {"zealot": 3, "stalker": 2},
        "blue_units": {"zealot": 3, "stalker": 2},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "stalkers_and_zealots",
        "red_units": {"zealot": 5, "stalker": 3},
        "blue_units": {"zealot": 5, "stalker": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
    "1c3s5z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "colossi_stalkers_zealots",
        "red_units": {"colossus": 1, "zealot": 5, "stalker": 3},
        "blue_units": {"colossus": 1, "zealot": 5, "stalker": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": True,
        "map_name": "COMMON"
    },
}


def get_smac_map_registry():
    return map_param_registry


# for name, map_params in map_param_registry.items():
#     globals()[name] = type(
#         name, (SMACMap,), dict(filename=map_params["map_name"])
#     )

globals()['COMMON'] = type(
    'COMMON', (SC2BAMap,), dict(filename='COMMON')
)