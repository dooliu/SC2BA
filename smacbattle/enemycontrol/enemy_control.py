import copy
import os

import yaml
import collections
import torch as th

from smacbattle.enemycontrol.controllers import REGISTRY as mac_REGISTRY
from smacbattle.enemycontrol.components.episode_buffer import EpisodeBatch
from smacbattle.enemycontrol.components.transforms import OneHot
from types import SimpleNamespace as SN
from functools import partial

enemy_name = {
    "computer ai": 0,
    "qmix": 1,
    "vdn": 2,
    "dop": 3,
    "fop": 4,
    "qplex": 5,
    "coma": 6,
    "iql": 7,
    "qtran": 8,
}


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class EnemyControl:

    def __init__(self, env_info, map_name="3m"):
        # attention: for blue camp agents select action
        # process each algorithm selected probability
        self.batch = None
        self.mac = None
        # placeholder, algorithm index start at 1
        self.mac_pool = ["Computer AI"]
        # load all algorithm models
        self.algorithm_list = ['qmix', 'vdn', 'dop', 'fop', 'qplex', 'coma', 'iql', 'qtran']
        self.config_map = self.load_configs(env_info)
        print(self.config_map)
        # create mac controller
        n_actions = env_info["n_blue_actions"]
        scheme = {
            "state": {"vshape": env_info["state_blue_shape"]},
            "obs": {"vshape": env_info["obs_blue_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "actions_onehot": {"vshape": (n_actions,), "group": "agents", "dtype": th.float32},
            "avail_actions": {"vshape": (env_info["n_blue_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": env_info["n_enemies"]
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=env_info["n_blue_actions"])])
        }
        device = "cuda" if th.cuda.is_available() else "cpu"
        for config_name in self.algorithm_list:
            tmp_mac = mac_REGISTRY['basic_mac'](scheme, groups, self.config_map[config_name])
            load_path = os.path.join(os.path.dirname(__file__), "models", map_name, config_name)
            tmp_mac.load_models(load_path)
            self.mac_pool.append(tmp_mac)
        self.new_batch = partial(EpisodeBatch, scheme, groups, 1, env_info["episode_limit"] + 1,
                                 preprocess=preprocess, device=device)
        self.cuda()

    def reset(self, enemy_chose=1):
        # according this episode sample enemy, rebuild mac
        self.mac = self.mac_pool[enemy_chose]
        self.batch = self.new_batch()
        self.mac.init_hidden(batch_size=self.batch.batch_size)

    def cuda(self):
        if th.cuda.is_available():
            for tmp_mac in self.mac_pool:
                if tmp_mac != "Computer AI":
                    tmp_mac.cuda()

    def load_configs(self, env_info):
        """load all algorithm config"""
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)
        with open(os.path.join(os.path.dirname(__file__), "config", 'env', "sc2.yaml"), "r") as f:
            try:
                env_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)
        config_dict = recursive_dict_update(config_dict, env_dict)
        load_path = os.path.join(os.path.dirname(__file__), "config", "algs")
        config_map = {}
        for config_name in self.algorithm_list:
            with open(os.path.join(load_path, "{}.yaml".format(config_name)), "r") as f:
                try:
                    alg_config = yaml.load(f, Loader=yaml.FullLoader)
                    tmp_config_dict = copy.deepcopy(config_dict)
                    tmp_config_dict = recursive_dict_update(tmp_config_dict, alg_config)
                    args = SN(**tmp_config_dict)
                    args.n_agents = env_info["n_enemies"]
                    args.n_actions = env_info["n_blue_actions"]
                    args.state_shape = env_info["state_blue_shape"]
                    config_map[config_name] = args
                except Exception as exc:
                    assert False, "{} load error: {}".format(config_name, exc)
        return config_map

    def select_actions(self, state, available_action, observation, t_ep):
        # according current state and history data, select action
        pre_transition_data = {
            "state": [state],
            "avail_actions": [available_action],
            "obs": [observation]
        }
        self.batch.update(pre_transition_data, ts=t_ep)
        # red camp choose action
        actions = self.mac.select_actions(self.batch, t_ep=t_ep)
        return actions[0].tolist()

    def update_action_reward(self, actions, reward, terminated, t_ep):
        # update current
        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated,)],
        }
        self.batch.update(post_transition_data, ts=t_ep)
        # red camp choose action
