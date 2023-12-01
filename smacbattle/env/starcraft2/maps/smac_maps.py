from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
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
        "n_enemies_real": 9,
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
    "8m_vs_9m": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
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
    "27m_vs_30m": {
        "n_agents": 27,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
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
    "3z_vs_3s": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "stalkers_and_zealots",
        "red_units": {"zealot": 3},
        "blue_units": {"stalker": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [1],
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "5z_vs_3s": {
        "n_agents": 5,
        "n_enemies": 3,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "stalkers_and_zealots",
        "red_units": {"zealot": 5},
        "blue_units": {"stalker": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [1],
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "3s5z_vs_3s6z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 10,
        "map_type": "stalkers_and_zealots",
        "red_units": {"zealot": 5, "stalker": 3},
        "blue_units": {"zealot": 6, "stalker": 3},
        "red_start_position": (9, 16),
        "blue_start_position": (23, 16),
        "blue_control_models": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "playable_area": {"lower_left": (0, 8), "upper_right": (32, 24)},
        "mirror_position": False,
        "map_name": "COMMON"
    },
    "3s_vs_3z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "map_name": "COMMON"
    },
    "3s_vs_4z": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "map_name": "COMMON"
    },
    "3s_vs_5z": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "map_name": "COMMON"
    },
    "2m_vs_1z": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "marines",
        "map_name": "COMMON"
    },
    "corridor": {
        "n_agents": 6,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
        "map_name": "COMMON"
    },
    "6h_vs_8z": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
        "map_name": "COMMON"
    },
    "2s_vs_1sc": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 300,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "map_name": "COMMON"
    },
    "so_many_baneling": {
        "n_agents": 7,
        "n_enemies": 32,
        "limit": 100,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
        "map_name": "COMMON"
    },
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 200,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 9,
        "map_type": "bane",
        "map_name": "COMMON"
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
        "map_name": "COMMON"
    },
}


def get_smac_map_registry():
    return map_param_registry


for name, map_params in map_param_registry.items():
    globals()[name] = type(
        name, (SMACMap,), dict(filename=map_params["map_name"])
    )
