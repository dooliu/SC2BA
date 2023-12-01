from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import pdb
import random

from smacbattle.env.starcraft2.enums import Camp
from smacbattle.env.multiagentenv import MultiAgentEnv
from smacbattle.env.starcraft2.maps import get_map_params
from smacbattle.enemycontrol.enemy_control import EnemyControl

import atexit
from warnings import warn
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib import run_parallel
from pysc2.lib import portspicker
from pysc2.lib.units import Protoss, Terran, Zerg

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


def to_list(arg):
    return arg if isinstance(arg, list) else [arg]


def get_default(a, b):
    return b if a is None else a


class Agent(collections.namedtuple("Agent", ["race", "name"])):
    """Define an Agent. It can have a single race or a list of races."""

    def __new__(cls, race, name=None):
        return super(Agent, cls).__new__(cls, race, name or "<unknown>")


class Bot(collections.namedtuple("Bot", ["race", "name", "difficulty"])):
    """Define a Bot. race determined by Map configured；if not point Algorithm,Boot controll by Computer AI."""

    def __new__(cls, race, name, difficulty):
        return super(Bot, cls).__new__(
            cls, race, name, difficulty)


class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            map_name="8m",
            players=None,
            step_mul=8,
            move_amount=2,
            difficulty="7",
            game_version=None,
            seed=None,
            random_enemy=False,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=-100,
            reward_tie=-50,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        players: A list of Agent and Bot instances that specify who will play.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        random_enemy : bool, optional
            if true, the enemy will uniform random in each episode
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # impute data Validation
        if not players:
            raise ValueError("You must specify the list of players.")
        for p in players:
            if not isinstance(p, (Agent, Bot)):
                raise ValueError(
                    "Expected players to be of type Agent or Bot. Got: %s." % p)
        num_players = len(players)
        self._num_agents = sum(1 for p in players if isinstance(p, Agent))
        self._players = players
        if not 1 <= num_players <= 2 or not self._num_agents:
            raise ValueError("Only 1 or 2 players with at least one agent is "
                             "supported at the moment.")
        self.red_agent_flag = True if isinstance(self._players[0], Agent) else False
        self.blue_agent_flag = True if isinstance(self._players[1], Agent) else False
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_red_agents = map_params["n_agents"]
        # used to set model size
        self.n_blue_agents = map_params["n_enemies"]
        # used to set real enemy unit count, red_actions
        self.n_blue_agents_gen = map_params.setdefault("n_enemies_real", self.n_blue_agents)
        # 每个episode Step的限制
        self.episode_limit = map_params["limit"]
        # StarCraft2 real load map name
        # self.load_map_name = map_params["map_name"]
        # Red、Blue Camp unit config, used to create units
        self.red_units_config = map_params["red_units"]
        self.blue_units_config = map_params["blue_units"]
        # whether create Blue Camp Unit by mirror type(central symmetry)
        self.mirror_create_blue_camp = map_params.setdefault("mirror_position", False)
        # if red and blue units count is not equally, mirror creat flag is not use
        if self.n_red_agents != self.n_blue_agents_gen:
            self.mirror_create_blue_camp = False
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_tie = reward_tie
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        # red and blue two camp's actions
        self.n_red_actions = self.n_actions_no_attack + self.n_blue_agents
        self.n_blue_actions = self.n_actions_no_attack + self.n_red_agents
        # Blue Agents whether swap left right action
        # self.crosswise_swap_flag = np.zeros(self.n_blue_agents)

        # Map info
        self._red_race = map_params["a_race"]
        self._blue_race = map_params["b_race"]

        # random enemy
        self.random_enemy = random_enemy
        self.enemy_controller = None
        self.random_enemy_seq = None
        self.enemy_code = None

        # only P race have shield
        self.shield_bits_ally = 1 if self._red_race == "P" else 0
        self.shield_bits_enemy = 1 if self._blue_race == "P" else 0
        # unit type will need dim size
        self.unit_type_bits = map_params["unit_type_bits"]
        # used to help env chose unit type
        self.map_type = map_params["map_type"]
        self.red_start_position = map_params.setdefault("red_start_position", (9, 16))
        # if map mirror create units, this attribute will not use
        self.blue_start_position = map_params.setdefault("blue_start_position", (23, 16))
        # used to limit map playable area, keep each scenario same with original
        self.playable_area_range = map_params.setdefault("playable_area", None)
        self.random_enemy_seq = map_params.setdefault("blue_control_models", [0])

        # save unit's type where in entire map
        self._unit_types = None

        # max value, used to normalized camp's reward
        self.max_red_reward = (
                self.n_blue_agents * self.reward_death_value + self.reward_win
        )
        self.max_blue_reward = (
                self.n_red_agents * self.reward_death_value + self.reward_win
        )

        # create lists containing the names of attributes returned in states
        self.red_ally_state_attr_names = ["health", "energy/cooldown", "rel_x", "rel_y"]
        self.red_enemy_state_attr_names = ["health", "rel_x", "rel_y"]
        # red blue camp features are depart. mainly reason is shield bit
        self.blue_ally_state_attr_names = ["health", "energy/cooldown", "rel_x", "rel_y"]
        self.blue_enemy_state_attr_names = ["health", "rel_x", "rel_y"]

        # Red Or Blue whether has shield(race are P)
        if self.shield_bits_ally > 0:
            self.red_ally_state_attr_names += ["shield"]
            self.blue_enemy_state_attr_names += ["shield"]
        if self.shield_bits_enemy > 0:
            self.red_enemy_state_attr_names += ["shield"]
            self.blue_ally_state_attr_names += ["shield"]

        if self.unit_type_bits > 0:
            bit_attr_names = ["unit_type"]
            self.red_ally_state_attr_names += bit_attr_names
            self.red_enemy_state_attr_names += bit_attr_names
            self.blue_ally_state_attr_names += bit_attr_names
            self.blue_enemy_state_attr_names += bit_attr_names

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_red_agents)
        self.death_tracker_enemy = np.zeros(self.n_blue_agents_gen)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        # Red camp unit obs cache
        self.red_last_obs = None
        # Red and Blue last time step executed action
        self.red_last_action = np.zeros((self.n_red_agents, self.n_red_actions))
        self.blue_last_action = np.zeros((self.n_blue_agents, self.n_blue_actions))
        # min unit type in the map
        self._min_unit_type = 0
        # min unit type controlled by RL in the map
        self._min_rl_unit_type = 0
        # Red camp agent unit_type in whole Map
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None
        self._parallel = run_parallel.RunParallel()  # Needed for multiplayer.

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)
        print(f"Maps info is:{_map}")
        # get interface_format
        agent_interface_format = sc_pb.InterfaceOptions(raw=True, score=False)
        # create game process
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )

        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
            disable_fog=False
        )

        # Add player into Game, Participant(Agent control by RL)、Computer two types
        # Players order is important.
        # Computer、Participant，mean Agent control by RL located in the right
        for p in self._players:
            if isinstance(p, Agent):
                create.player_setup.add(type=sc_pb.Participant)
                agent_race = p.race
            else:
                # Add a Robot controlled by Computer AI.
                create.player_setup.add(
                    type=sc_pb.Computer, race=races[p.race], difficulty=p.difficulty)
        # crete Game
        self._controller.create_game(create)
        # create the join request
        join = sc_pb.RequestJoinGame(
            race=races[agent_race], options=agent_interface_format
        )

        print(f"controller's:{self._controller}, joins:{join}")
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        # print(f"Game Info is:{game_info}")
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        # narrow playable area in Map
        if self.playable_area_range is not None:
            # left and downer point
            lower_left = self.playable_area_range["lower_left"]
            if lower_left[0] > map_play_area_min.x:
                map_play_area_min.x = lower_left[0]
            if lower_left[1] > map_play_area_min.y:
                map_play_area_min.y = lower_left[1]
            upper_right = self.playable_area_range["upper_right"]
            if upper_right[0] < map_play_area_max.x:
                map_play_area_max.x = upper_right[0]
            if upper_right[1] < map_play_area_max.y:
                map_play_area_max.y = upper_right[1]

        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y
        self.center_x = self.map_x / 2
        self.center_y = self.map_y / 2

        # 生成地图中的路径信息
        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )

        # Map playable area take intersection with map config()
        if self.playable_area_range is not None:
            adaptive_pa = np.zeros((self.map_x, self.map_y), dtype=bool)
            adaptive_pa[map_play_area_min.x: map_play_area_max.x, map_play_area_min.y: map_play_area_max.y] = True
            self.pathing_grid = np.logical_and(adaptive_pa, self.pathing_grid)

        # 地图中的地形高度
        self.terrain_height = (
                np.flip(
                    np.transpose(
                        np.array(list(map_info.terrain_height.data)).reshape(
                            self.map_x, self.map_y
                        )
                    ),
                    1,
                ) / 255
        )

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
            # random enemy setting
            if self.random_enemy:
                self.enemy_controller = EnemyControl(self.get_env_info(), self.map_name)
                # self.random_enemy_seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                # self.random_enemy_seq = self.map_type
                random.shuffle(self.random_enemy_seq)
                # get selected enemy code
                self.enemy_code = self.random_enemy_seq[0]
        else:
            # restart a new episode
            # print(f"restart a new game!")
            self._restart()
            if self.random_enemy:
                episode_index = self._episode_count % len(self.random_enemy_seq)
                if episode_index == 0:
                    random.shuffle(self.random_enemy_seq)
                # get selected enemy code
                self.enemy_code = self.random_enemy_seq[episode_index]
        # print(f"choosed enemy_code is :{self.enemy_code}")
        self._episode_steps = 0
        # 0 represent enemy control by embed Computer AI
        # when generate unit, enemy will be set to origin unit otherwise RL_unit
        if self.random_enemy and self.enemy_code != 0:
            self.enemy_controller.reset(self.enemy_code)

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_red_agents)
        self.death_tracker_enemy = np.zeros(self.n_blue_agents_gen)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.red_last_action = np.zeros((self.n_red_agents, self.n_red_actions))
        self.blue_last_action = np.zeros((self.n_blue_agents, self.n_blue_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_red_agents

        try:
            # CHEAT order: control enemies unit
            debug_command = [
                d_pb.DebugCommand(game_state=d_pb.control_enemy)
            ]
            self._controller.debug(debug_command)
            # get game info from pysc2 api
            self._obs = self._controller.observe()
            # print(f"从Controller中获取的观察空间为：{self._obs}")
            # create unit and init env
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            print(f"Game reset raise exception!")
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        When Game have two agent player's, Env need hard reset,
        two controllers leave game, and join game again.
        (process、controllers will not rebuild)
        """
        try:
            # Need to support restart for fast-restart of mini-games.
            # self._controller.restart()
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one.
           This will cost expensive.
        """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward(Red & Blue), terminated, info."""
        # slice as array
        actions_int = [int(a) for a in actions]
        # according to players order, generate red and blue camp unit action
        # if camp is controlled by computer, it's action list is none
        red_actions = blue_actions = []
        # red camp is located in left position
        # set action slide position, determined blue camp action is always correct
        action_slide = 0
        if isinstance(self._players[0], Agent):
            red_actions = actions_int[0:self.n_red_agents]
            action_slide = self.n_red_agents
        elif self.random_enemy and self.enemy_code != 0:
            red_actions = self.enemy_controller.select_actions(self.get_state(), self.get_avail_actions(),
                                                               self.get_obs(), self._episode_steps)
        #  red camp is located in left position
        if isinstance(self._players[1], Agent):
            blue_actions = actions_int[action_slide:action_slide + self.n_blue_agents]
        elif self.random_enemy and self.enemy_code != 0:
            blue_actions = self.enemy_controller.select_actions(self.get_state(Camp.BLUE),
                                                                self.get_avail_actions(Camp.BLUE),
                                                                self.get_obs(Camp.BLUE), self._episode_steps)
        # print(f"当前Step:{self._episode_steps}红蓝方的动作为：{red_actions}-{blue_actions}")
        # if red or blue camp is controlled by computer, it‘s last action default is stop
        if len(red_actions) == 0:
            self.red_last_action = np.eye(self.n_red_actions)[np.array([1] * self.n_red_agents)]
        else:
            self.red_last_action = np.eye(self.n_red_actions)[np.array(red_actions)]
        if len(blue_actions) == 0:
            self.blue_last_action = np.eye(self.n_blue_actions)[np.array([1] * self.n_blue_agents)]
        else:
            self.blue_last_action = np.eye(self.n_blue_actions)[np.array(blue_actions)]

        # Collect individual actions
        red_blue_actions = []
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(red_actions):
            if not self.heuristic_ai:
                # print(f"游戏玩家RL选择执行动作！")
                sc_action = self.get_agent_action(a_id, action)
            else:
                # print(f"heuristic_ai选择动作！")
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)
        red_blue_actions.extend(sc_actions)
        # b_player is Bot(Controlled by Computer AI)
        en_actions = []
        for e_id, action in enumerate(blue_actions):
            if not self.heuristic_ai:
                # print(f"游戏玩家RL选择执行动作！")
                en_action = self.get_agent_action(e_id, action, camp=Camp.BLUE)
            else:
                # print(f"heuristic_ai选择动作！")
                en_action, action_num = self.get_agent_action_heuristic(
                    e_id, action
                )
                actions[3 + e_id] = action_num
            if en_action:
                en_actions.append(en_action)
        red_blue_actions.extend(en_actions)
        # use one controller control two camp unit(Use Cheat order)
        # red_blue_action[0].extend(en_actions)
        # print(f"Blue Agents 生成的动作列表为：{en_actions}")
        # Send action request
        try:
            req_actions = sc_pb.RequestAction(actions=red_blue_actions)
            # execute action and speed up
            self._controller.actions(req_actions)
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
            # print(f"执行动作后的观察空间为：{self._obs}")
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return (0, 0), True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        # 如果一方全部死亡，则游戏结束（输或赢）
        game_end_code = self.update_units()

        terminated = False
        # calculate red、blue camp's reward
        red_reward, blue_reward = self.reward_battle()
        episode_info = {"red_battle_won": False, "blue_battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        episode_info["dead_red"] = dead_allies
        episode_info["dead_blue"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                episode_info["red_battle_won"] = True
                episode_info["blue_battle_won"] = False
                if not self.reward_sparse:
                    red_reward += self.reward_win
                    blue_reward += self.reward_defeat
                else:
                    red_reward, blue_reward = (1, -1)
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                episode_info["red_battle_won"] = False
                episode_info["blue_battle_won"] = True
                if not self.reward_sparse:
                    red_reward += self.reward_defeat
                    blue_reward += self.reward_win
                else:
                    red_reward, blue_reward = (-1, 1)
        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                episode_info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1
            if not self.reward_sparse:
                red_reward += self.reward_tie
                blue_reward += self.reward_tie
            else:
                red_reward, blue_reward = (-1, -1)

        if self.debug:
            logging.debug("Reward = {}-{}".format(red_reward, blue_reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            # print(f"缩放之前的奖励为：{red_reward}-{blue_reward}")
            red_reward /= self.max_red_reward / self.reward_scale_rate
            blue_reward /= self.max_blue_reward / self.reward_scale_rate
        self.reward = (red_reward, blue_reward)
        # if random enemy, save this action execute reward
        if self.random_enemy and self.enemy_code != 0:
            if self.red_agent_flag is False:
                tmp_actions = red_actions
                tmp_reward = red_reward
            else:
                tmp_actions = blue_actions
                tmp_reward = blue_reward
            self.enemy_controller.update_action_reward(tmp_actions, tmp_reward,
                                                       terminated != episode_info.get("episode_limit", False),
                                                       self.episode_steps)
        return (red_reward, blue_reward), terminated, episode_info

    def get_agent_action(self, a_id, action, camp=Camp.RED):
        """Construct the action for agent a_id.
           select target enemy according last obs and action
        """
        # TODO the action is weather changed
        avail_actions = self.get_avail_agent_actions(a_id, camp=camp)
        assert (
                avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        if camp == Camp.RED:
            unit = self.get_unit_by_id(a_id)
            ally_agents = self.agents
            target_agents = self.enemies
        else:
            unit = self.get_enemy_unit_by_id(a_id)
            ally_agents = self.enemies
            target_agents = self.agents
            # 当蓝方选择的动作为左右移动时，交换左右动作
            if action == 4 or action == 5:
                action = 9 - action
                # swap action cannot perform, keep pre-state
                if avail_actions[action] != 1:
                    action = 9 - action
            if action == 2 or action == 3:
                action = 5 - action
                # swap action cannot perform, keep pre-state
                if avail_actions[action] != 1:
                    action = 5 - action
        # unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            # according to target id and agent obs determine target unit
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = ally_agents[target_id]
                action_name = "heal"
            else:
                target_unit = target_agents[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        """根据unit的id来获取智能执行的动作"""
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        # 如果单位是医疗车，则目标单位是己方的士兵
        if unit.unit_type == self.medivac_id:
            if (
                    target is None
                    or self.agents[target].health == 0
                    or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (
                            al_unit.health != 0
                            and al_unit.health != al_unit.health_max
                    ):
                        dist = self.distance(
                            unit.pos.x,
                            unit.pos.y,
                            al_unit.pos.x,
                            al_unit.pos.y,
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["heal"]
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (
                            unit.unit_type == self.marauder_id
                            and e_unit.unit_type == self.medivac_id
                    ):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["attack"]
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (
                self.heuristic_rest
                and self.get_avail_agent_actions(a_id)[action_num] == 0
        ):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y
                    )
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y
                    )
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount
                    )
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False,
            )
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        return red_reward, blue_reward
        """
        if self.reward_sparse:
            return 0, 0

        red_reward = blue_reward = 0
        red_delta_deaths = 0
        red_delta_ally = 0
        red_delta_enemy = 0
        blue_delta_deaths = 0
        blue_delta_ally = 0
        blue_delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died, update death record
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        red_delta_deaths -= self.reward_death_value * neg_scale
                    red_delta_ally += prev_health * neg_scale
                    # for blue, red camp suffer damage, is rewarded
                    blue_delta_deaths += self.reward_death_value
                    blue_delta_enemy += prev_health
                else:
                    # still alive
                    red_delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )
                    # for blue, red camp suffer damage, is rewarded
                    blue_delta_enemy += prev_health - al_unit.health - al_unit.shield

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    # for blue, e_unit death will suffer penalty
                    if not self.reward_only_positive:
                        blue_delta_deaths -= self.reward_death_value * neg_scale
                    blue_delta_ally += prev_health * neg_scale
                    red_delta_deaths += self.reward_death_value
                    red_delta_enemy += prev_health
                else:
                    # still alive
                    blue_delta_ally += neg_scale * (
                            prev_health - e_unit.health - e_unit.shield
                    )
                    red_delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            red_reward = abs(red_delta_enemy + red_delta_deaths)  # shield regeneration
            blue_reward = abs(blue_delta_enemy + blue_delta_deaths)
        else:
            red_reward = red_delta_enemy + red_delta_deaths - red_delta_ally
            blue_reward = blue_delta_enemy + blue_delta_deaths - blue_delta_ally

        return red_reward, blue_reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_red_actions, self.n_blue_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self, prefix=None):
        """Save a replay."""
        prefix = self.replay_prefix or prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(),
            replay_dir=replay_dir,
            prefix=prefix,
        )
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_obs_agent(self, agent_id, camp=Camp.RED):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information
            and pathing grid)
        - enemy features (available_to_attack-e_id, distance/sight, relative_x, relative_y,
            health, shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, health, shield,
            unit_type, last_action)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order.
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        # 获取Red阵营的观测信息的维度
        direction_negative = 1
        robot_flag = None
        if camp == Camp.RED:
            unit = self.get_unit_by_id(agent_id)
            move_feats_dim = self.get_obs_move_feats_size()
            ally_feats_dim, enemy_feats_dim = self.get_red_obs_feats_size()
            # enemy_feats_dim = self.get_obs_enemy_feats_size()
            # ally_feats_dim = self.get_obs_ally_feats_size()
            shield_bits_ally = self.shield_bits_ally
            shield_bits_enemy = self.shield_bits_enemy
            robot_flag = (not self.red_agent_flag) and self.random_enemy
        else:
            # 获取Blue阵营的观测信息维度
            unit = self.get_enemy_unit_by_id(agent_id)
            move_feats_dim = self.get_obs_move_feats_size()
            # 当阵营为蓝方时，敌我双方数据互换
            ally_feats_dim, enemy_feats_dim = self.get_blue_obs_feats_size()
            # enemy_feats_dim = self.get_obs_ally_feats_size()
            # ally_feats_dim = self.get_obs_enemy_feats_size()

            shield_bits_ally = self.shield_bits_enemy
            shield_bits_enemy = self.shield_bits_ally
            # 当阵营为蓝方时，且其不在最左侧时，relative_x改变其正负(direction_negative=-1)
            # red_x_list = np.array([tmp_agent.pos.x for a_id, tmp_agent in self.agents.items() if tmp_agent.health > 0])
            # red_agent_min_x = red_x_list.min(initial=self.map_x)
            # 如果是蓝方，判断是否在所有红方Agent左侧
            direction_negative = -1
            robot_flag = (not self.blue_agent_flag) and self.random_enemy

        # own feats dim
        own_feats_dim = self.get_obs_own_feats_size(camp=camp)
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id, camp=camp)
            # print(f"阵营：{camp.name}中的智能体：{agent_id},可执行动作空间为：{avail_actions}")
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                ind: ind + self.n_obs_pathing  # noqa
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # 如果是Blue阵营的智能体，敌人为agents列表中的数据
            if camp == Camp.RED:
                enemy_agents = self.enemies
                ally_agents = self.agents
            else:
                enemy_agents = self.agents
                ally_agents = self.enemies
            # Enemy features
            for e_id, e_unit in enemy_agents.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                # if enemy visible and alive, add this enemy's features into obs
                if (dist < sight_range and e_unit.health > 0) or (robot_flag and self._episode_steps > 5):
                    # Sight range > shoot range
                    # print(f"debug info:{avail_actions}-{e_id}-{self.n_actions_no_attack + e_id}")
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                        ]  # available
                    enemy_feats[e_id, 1] = 1 if dist / sight_range > 1 else dist / sight_range   # distance
                    # multiple direction_negative
                    # if bigger than 1, change value as 1.
                    # if lower than -1, change value as -1.
                    relative_x = (e_x - x) * direction_negative / sight_range
                    if relative_x > 1 or relative_x < -1:
                        relative_x = 1 if relative_x > 1 else relative_x
                        relative_x = -1 if relative_x < -1 else relative_x
                    enemy_feats[e_id, 2] = relative_x  # relative X
                    relative_y = (e_y - y) * direction_negative / sight_range
                    if relative_y > 1 or relative_y < -1:
                        relative_y = 1 if relative_y > 1 else relative_y
                        relative_y = -1 if relative_y < -1 else relative_y
                    enemy_feats[e_id, 3] = relative_y  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit)
                        enemy_feats[e_id, ind] = type_id / self.unit_type_bits  # unit type

            # 如果泛化的敌军数量大于训练的模型，则进行enemy_features的优减
            if self.n_blue_agents < self.n_blue_agents_gen:
                stay_remove_count = self.n_blue_agents_gen - self.n_blue_agents
                for tmp_ind in reversed(range(self.n_blue_agents_gen)):
                    if enemy_feats[tmp_ind][0] == 0:
                        enemy_feats = np.delete(enemy_feats, tmp_ind, 0)
                        stay_remove_count -= 1
                    if stay_remove_count == 0:
                        break
                if stay_remove_count > 0:
                    enemy_feats = enemy_feats[:-stay_remove_count]

            # Ally features exclude itself
            al_ids = [
                al_id for al_id, tmp_agent in ally_agents.items() if tmp_agent.tag != unit.tag
            ]
            # al_ids = [
            #     al_id for al_id in range(self.n_red_agents) if al_id != agent_id
            # ]
            for i, al_id in enumerate(al_ids):

                al_unit = ally_agents[al_id]
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                # 如果在视野范围内，则更新特征，否则默认特征都为0
                if dist < sight_range and al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) * direction_negative / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) * direction_negative / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = al_unit.health / al_unit.health_max  # health
                        ind += 1
                        if shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = al_unit.shield / max_shield  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit)
                        ally_feats[i, ind] = type_id / self.unit_type_bits
                        ind += 1

                    if self.obs_last_action:
                        if camp == Camp.RED:
                            ally_feats[i, ind:] = self.red_last_action[al_id]
                        else:
                            ally_feats[i, ind:] = self.blue_last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit)
                own_feats[ind] = type_id / self.unit_type_bits

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))
        # print(f"Camp:{camp},Obs Agent: {agent_id},Move feats {move_feats}")
        # print(f"Camp:{camp},Obs Agent: {agent_id},Enemy feats {enemy_feats}")
        # print(f"Camp:{camp},Obs Agent: {agent_id},Ally feats {ally_feats}")
        # print(f"Camp:{camp},Obs Agent: {agent_id},Own feats {own_feats}")
        return agent_obs

    def get_obs(self, camp=Camp.RED):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        if camp == Camp.RED:
            # red camp Observe
            agents_obs = [self.get_obs_agent(i, camp=camp) for i in range(self.n_red_agents)]
            self.red_last_obs = agents_obs
        else:
            # blue camp Observe
            # init crosswise_swap_flag value
            # self.crosswise_swap_flag = np.zeros(self.n_blue_agents)
            agents_obs = [self.get_obs_agent(i, camp=camp) for i in range(self.n_blue_agents)]
        return agents_obs

    def get_state(self, camp=Camp.RED):
        """Returns the global state.
        NOTE: This function should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(camp), axis=0).astype(
                np.float32
            )
            return obs_concat
        # get point camp view state
        state_dict = self.get_state_dict(camp=camp)
        # pdb.set_trace()
        state = np.append(
            state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())
        if "timestep" in state_dict:
            state = np.append(state, state_dict["timestep"])

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(state_dict["allies"]))
            logging.debug("Enemy state {}".format(state_dict["enemies"]))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.red_last_action))

        return state

    def get_ally_num_attributes(self, camp=Camp.RED):
        if camp == Camp.RED:
            return len(self.red_ally_state_attr_names)
        else:
            return len(self.blue_ally_state_attr_names)

    def get_enemy_num_attributes(self, camp=Camp.RED):
        if camp == Camp.RED:
            return len(self.red_enemy_state_attr_names)
        else:
            return len(self.blue_enemy_state_attr_names)

    def get_state_dict(self, camp=Camp.RED):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """
        direction_negative = 1
        if camp == Camp.RED:
            ally_count = self.n_red_agents
            ally_agents = self.agents
            enemy_count = self.n_blue_agents_gen
            enemy_agents = self.enemies
            shield_bits_ally = self.shield_bits_ally
            shield_bits_enemy = self.shield_bits_enemy
            last_action = self.red_last_action
        else:
            ally_count = self.n_blue_agents
            ally_agents = self.enemies
            enemy_count = self.n_red_agents
            enemy_agents = self.agents
            shield_bits_ally = self.shield_bits_enemy
            shield_bits_enemy = self.shield_bits_ally
            last_action = self.blue_last_action
            # blue camp reverse map in x axis
            direction_negative = -1

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes(camp)
        nf_en = self.get_enemy_num_attributes(camp)

        ally_state = np.zeros((ally_count, nf_al))
        enemy_state = np.zeros((enemy_count, nf_en))

        center_x = self.center_x
        center_y = self.center_y

        # ally features, all agents feature
        for al_id, al_unit in ally_agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                        al_unit.health / al_unit.health_max
                )  # health
                # 医疗飞机
                if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = al_unit.weapon_cooldown / max_cd  # cooldown
                ally_state[al_id, 2] = (x - center_x) * direction_negative / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (y - center_y) * direction_negative / self.max_distance_y  # relative Y
                ind = 4
                if shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = al_unit.shield / max_shield  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit)
                    ally_state[al_id, ind] = type_id / self.unit_type_bits

        for e_id, e_unit in enemy_agents.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.health_max  # health
                enemy_state[e_id, 1] = (x - center_x) * direction_negative / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (y - center_y) * direction_negative / self.max_distance_y  # relative Y
                ind = 3
                if shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = e_unit.shield / max_shield  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit)
                    enemy_state[e_id, ind] = type_id / self.unit_type_bits

        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit

        # print(f"Camp:{camp},State allie feats {state['allies']}")
        # print(f"Camp:{camp},State enemies feats {state['enemies']}")
        # print(f"Camp:{camp},State last_action feats {state['last_action']}")
        return state

    def get_red_obs_feats_size(self):
        """Returns Red Camp's the dimensions of the matrix containing ally features.
        Size is n_allies x n_features(include last action if needed), n_enemy, n_features.
        can get ally agent's last action.
        """
        nf_al = 4 + (1 if self.unit_type_bits > 0 else 0)
        # red ally agents features size
        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
        if self.obs_last_action:
            nf_al += self.n_red_actions

        nf_en = 4 + (1 if self.unit_type_bits > 0 else 0)
        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return (self.n_red_agents - 1, nf_al), (self.n_blue_agents_gen, nf_en)

    def get_blue_obs_feats_size(self):
        """Returns Blue Camp's the dimensions of the matrix containing ally features.
        Size is n_allies x n_features(include last action if needed), n_enemy, n_features.
        """
        nf_al = 4 + (1 if self.unit_type_bits > 0 else 0)
        # red ally agents features size
        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_enemy
        if self.obs_last_action:
            nf_al += self.n_blue_actions

        nf_en = 4 + (1 if self.unit_type_bits > 0 else 0)
        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_ally

        return (self.n_blue_agents - 1, nf_al), (self.n_red_agents, nf_en)

    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_blue_agents x n_features.
        """
        nf_en = 4 + (1 if self.unit_type_bits > 0 else 0)

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_blue_agents, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + (1 if self.unit_type_bits > 0 else 0)

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_red_actions

        return self.n_red_agents - 1, nf_al

    def get_obs_own_feats_size(self, camp=Camp.RED):
        """
        Returns the size of the vector containing the agents' own features.
        """
        own_feats = (1 if self.unit_type_bits > 0 else 0)
        if camp == Camp.RED:
            shield_bits = self.shield_bits_ally
        else:
            shield_bits = self.shield_bits_enemy
        if self.obs_own_health:
            own_feats += 1 + shield_bits
        if self.obs_timestep_number:
            own_feats += 1

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-
        related features.
        include: move_feats,n_obs_pathing,n_obs_height, last two default False
        """
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_size(self, camp=Camp.RED):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size(camp)
        move_feats = self.get_obs_move_feats_size()
        # 根据阵营获取对应的
        if camp == Camp.RED:
            ally_feats_dim, enemy_feats_dim = self.get_red_obs_feats_size()
        else:
            ally_feats_dim, enemy_feats_dim = self.get_blue_obs_feats_size()

        n_enemies, n_enemy_feats = enemy_feats_dim[0], enemy_feats_dim[1]
        n_allies, n_ally_feats = ally_feats_dim[0], ally_feats_dim[1]
        # n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        # n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self, camp=Camp.RED):
        """Returns the size of the global state."""
        # if obs_instead_of_state, need add Blue obs
        if self.obs_instead_of_state:
            return self.get_obs_size(camp) * self.n_red_agents

        if camp == Camp.RED:
            nf_al = 4 + self.shield_bits_ally + (1 if self.unit_type_bits > 0 else 0)
            nf_en = 3 + self.shield_bits_enemy + (1 if self.unit_type_bits > 0 else 0)

            enemy_state = self.n_blue_agents * nf_en
            ally_state = self.n_red_agents * nf_al
        else:
            nf_al = 4 + self.shield_bits_enemy + (1 if self.unit_type_bits > 0 else 0)
            nf_en = 3 + self.shield_bits_ally + (1 if self.unit_type_bits > 0 else 0)

            enemy_state = self.n_red_agents * nf_en
            ally_state = self.n_blue_agents * nf_al

        size = enemy_state + ally_state

        # obs include agent last action
        if self.state_last_action:
            if camp == Camp.RED:
                size += self.n_red_agents * self.n_red_actions
            else:
                size += self.n_blue_agents * self.n_blue_actions
        if self.state_timestep_number:
            size += 1

        return size

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions
        (n_red_agents, n_red_agents + n_blue_agents) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_red_agents, self.n_red_agents + self.n_blue_agents),
            dtype=np.bool,
        )

        for agent_id in range(self.n_red_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if dist < sight_range and e_unit.health > 0:
                        # visible and alive
                        arr[agent_id, self.n_red_agents + e_id] = 1

                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_red_agents) if al_id > agent_id
                ]
                for _, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if dist < sight_range and al_unit.health > 0:
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1

        return arr

    def get_unit_type_id(self, unit):
        """Returns the ID of unit type in the given scenario.
           All scenarios have same unit_type

           Parameters
           ----------
           unit: RL unit and Melee unit return same unit type
        """
        if unit.unit_type in (self.marine_id, Terran.Marine):
            return 1
        if unit.unit_type in (self.marauder_id, Terran.Marauder):
            return 2
        if unit.unit_type in (self.medivac_id, Terran.Medivac):
            return 3
        if unit.unit_type in (self.stalker_id, Protoss.Stalker):
            return 4
        if unit.unit_type in (self.zealot_id, Protoss.Zealot):
            return 5
        if unit.unit_type in (self.colossus_id, Protoss.Colossus):
            return 6
        if unit.unit_type in (self.zergling_id, Zerg.Zergling):
            return 7
        if unit.unit_type in (self.hydralisk_id, Zerg.Hydralisk):
            return 8
        if unit.unit_type in (self.baneling_id, Zerg.Baneling):
            return 9

    def get_avail_agent_move_actions(self, agent_id):
        """Returns the available actions for agent_id(Red & Blue).
           actions generate according to agents obs
        """
        unit = self.get_unit_by_id(agent_id)
        avail_action_count = self.n_actions_no_attack
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * avail_action_count

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1
        else:
            # only no-op allowed
            return [1] + [0] * (avail_action_count - 1)

    def get_avail_agent_actions(self, agent_id, camp=Camp.RED):
        """Returns the available actions for agent_id(Red & Blue).
           actions generate according to agents obs
           if obs not exist
        """
        if camp == Camp.RED:
            unit = self.get_unit_by_id(agent_id)
            avail_action_count = self.n_red_actions
        else:
            unit = self.get_enemy_unit_by_id(agent_id)
            avail_action_count = self.n_blue_actions
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * avail_action_count

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)
            if camp == Camp.RED:
                if self.n_blue_agents == self.n_blue_agents_gen:
                    target_items = self.enemies.items()
                else:
                    wait_pop_item = list(range(self.n_blue_agents_gen))
                    if self.red_last_obs is not None:
                        agent_obs = self.red_last_obs[agent_id]
                        # pick enemy info
                        ally_feats_dim, enemy_feats_dim = self.get_red_obs_feats_size()
                        enemy_features = agent_obs[avail_action_count:avail_action_count+self.n_blue_agents * enemy_feats_dim[1]]
                        print(f"slice enemy_features:{len(enemy_features)}-{enemy_features}")
                        enemy_features = np.array(enemy_features, np.float)
                        agent_obs_enemy = enemy_features.reshape(self.n_blue_agents, enemy_feats_dim[1])
                        agent_obs_ids = agent_obs_enemy[:0]
                    else:
                        pop_count = self.n_blue_agents_gen - self.n_blue_agents
                        agent_obs_ids = list(range(self.n_blue_agents, self.n_blue_agents_gen))
                        # pdb.set_trace()
                    # separate agent not obs enemy
                    target_items = [
                        (t_id, t_unit)
                        for (t_id, t_unit) in self.enemies.items()
                        if t_id not in agent_obs_ids
                    ]
                    # target_items = self.enemies.items()
                        # target_items = agent_obs_enemy
                    ally_item = self.agents.items()
            else:
                target_items = self.agents.items()
                ally_item = self.enemies.items()
            # 如果地图类型是MMM，并且当前Agent是救援直升机，则目标单位是己方的目标
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in ally_item
                    if t_unit.unit_type != self.medivac_id and t_id < (avail_action_count - self.n_actions_no_attack)
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1
            # print(f"获取agent的可操作动作，{camp.name}-{agent_id}, 可执行的action为：{avail_actions}")
            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (avail_action_count - 1)

    def get_avail_actions(self, camp=Camp.RED):
        """Returns the available actions of all agents in a list."""
        if camp == Camp.RED:
            agents_list = self.n_red_agents
        else:
            agents_list = self.n_blue_agents
        avail_actions = []
        for agent_id in range(agents_list):
            avail_agent = self.get_avail_agent_actions(agent_id, camp=camp)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self, mode="human"):
        if self.renderer is None:
            from render import StarCraft2Renderer
            self.renderer = StarCraft2Renderer(self, mode)
        assert (
                mode == self.renderer.mode
        ), "mode must be consistent across render calls"
        return self.renderer.render(mode)

    def _kill_all_units(self):
        """Kill all units on the map.
           Only One Player can use this function!
        """
        units_alive = [
                          unit.tag for unit in self.agents.values() if unit.health > 0
                      ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        # print(f"Map 中存活的单元为：{units_alive}")
        self._controller.debug(debug_command)
        # check the units are dead, and then restart one episode again
        while len(self._obs.observation.raw_data.units) > 0:
            self._controller.step(2)
            self._obs = self._controller.observe()

    def create_all_units(self):
        """Create all units on the map.
           according to Red、Blue unit config in map_info
        """
        # create unit debug order list
        debug_command = []
        red_agents_flag = self.red_agent_flag
        # if red camp are agents, create RL units
        # if red camp are robots and enemy is no computer Ai, create RL unit
        # if self.red_agent_flag is False and self.random_enemy and self.enemy_code != 0:
        #     red_agents_flag = True
        red_start_pos = sc_common.Point2D(x=self.red_start_position[0], y=self.red_start_position[1])
        for unit_name, unit_count in self.red_units_config.items():
            unit_type_id = self._get_unit_type(unit_name, red_agents_flag)
            debug_command.append(
                d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=unit_type_id,
                        owner=Camp.RED.value,
                        pos=red_start_pos,
                        quantity=unit_count,
                    )
                )
            )
        # mirror create blue camp
        # create Blue camp, get red position, then calculate opposite pos
        blue_agents_flag = self.blue_agent_flag
        # if self.blue_agent_flag is False and self.random_enemy and self.enemy_code != 0:
        #     blue_agents_flag = True
        if self.mirror_create_blue_camp:
            # cr
            self._controller.debug(debug_command)
            # wait Red Camp create accomplish
            while len(self._obs.observation.raw_data.units) < self.n_red_agents:
                self._controller.step(2)
                self._obs = self._controller.observe()
            debug_command.clear()
            # self._obs = self._controller.observe()

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                # key=attrgetter("unit_type", "tag"),
                reverse=False,
            )
            for unit in ally_units_sorted:
                position = self._center_symmetry_shift([unit.pos.x, unit.pos.y])
                blue_start_pos = sc_common.Point2D(x=position[0], y=position[1])
                # get unit name from unit map, and then get blue camp unit_type
                unit_name = self._get_unit_name(unit.unit_type, red_agents_flag)
                unit_type_id = self._get_unit_type(unit_name, blue_agents_flag)
                debug_command.append(
                    d_pb.DebugCommand(
                        create_unit=d_pb.DebugCreateUnit(
                            unit_type=unit_type_id,
                            owner=Camp.BLUE.value,
                            pos=blue_start_pos,
                            quantity=1,
                        )
                    )
                )
        else:
            blue_start_pos = sc_common.Point2D(x=self.blue_start_position[0], y=self.blue_start_position[1])
            for unit_name, unit_count in self.blue_units_config.items():
                unit_type_id = self._get_unit_type(unit_name, blue_agents_flag)
                debug_command.append(
                    d_pb.DebugCommand(
                        create_unit=d_pb.DebugCreateUnit(
                            unit_type=unit_type_id,
                            owner=Camp.BLUE.value,
                            pos=blue_start_pos,
                            quantity=unit_count,
                        )
                    )
                )
        # execute create unit debug order
        self._controller.debug(debug_command)
        # wait Blue Camp finish create order
        while len(self._obs.observation.raw_data.units) < (self.n_red_agents + self.n_blue_agents):
            self._controller.step(2)
            self._obs = self._controller.observe()

    def init_units(self):
        """Initialise(Create Units) the units, include Red(ally) and Blue(enemy) team."""
        # print(f"init unit raw data:{self._obs.observation.raw_data}")
        # init all units(include RL units and Melee unit)
        if self._episode_count == 0:
            self._init_unit_types()
        # Create units
        self.create_all_units()
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                # key=attrgetter("unit_type", "tag"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                unit = ally_units_sorted[i]
                self.agents[i] = unit
                # 计算敌方单位的总血量，得到最大奖励范围
                if self._episode_count == 0:
                    self.max_blue_reward += unit.health_max + unit.shield_max
            # print(f"init red agents result:{self.agents}")
            enemy_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]
            enemy_units_sorted = sorted(
                enemy_units,
                # key=attrgetter("unit_type", "pos.x", "pos.y"),
                key=attrgetter("unit_type", "tag"),
                reverse=False,
            )
            # # process enemy order, corresponding
            # for unit_name, unit_count in self.red_units_config.items():
            for i in range(len(enemy_units_sorted)):
                unit = enemy_units_sorted[i]
                self.enemies[i] = unit
                # 计算敌方单位的总血量，得到最大奖励范围
                if self._episode_count == 0:
                    self.max_red_reward += unit.health_max + unit.shield_max
            # print(f"init blue agents result:{self.enemies}")

            all_agents_created = len(self.agents) == self.n_red_agents
            all_enemies_created = len(self.enemies) == self.n_blue_agents_gen

            self._unit_types = [
                                   unit.unit_type for unit in ally_units_sorted
                               ] + [
                                   unit.unit_type for unit in enemy_units_sorted
                               ]
            # print(f"after init unit type:{self._unit_types}")
            # print(f"after init env's reward:{self.max_red_reward}-{self.max_blue_reward}-{self.reward_scale_rate}")
            if all_agents_created and all_enemies_created:  # all good
                return
            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def get_unit_types(self):
        if self._unit_types is None:
            warn(
                "unit types have not been initialized yet, please call"
                "env.reset() to populate this and call t1286he method again."
            )

        return self._unit_types

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
                n_ally_alive == 0
                and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)
        ):
            return -1  # lost
        if (
                n_ally_alive > 0
                and n_enemy_alive == 0
                or self.only_medivac_left(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_unit_types(self):
        """Initialise ally unit types. Should be called once from the
           init_units function.
           assume all rl unit in one map, so each unit have unique unit_type_id
        """
        # if both camp agents are controlled by RL, use min unit type as base
        # unit type are 9, each race have three Main fighting force
        # each version have different count units, so use sc2 api get unit count
        num_rl_units = 9
        self._min_rl_unit_type = (
                len(self._controller.data().units) - num_rl_units
        )
        # self._min_rl_unit_type = 2005
        self.colossus_id = self._min_rl_unit_type
        self.baneling_id = self._min_rl_unit_type + 1
        self.marine_id = self._min_rl_unit_type + 2
        self.marauder_id = self._min_rl_unit_type + 3
        self.medivac_id = self._min_rl_unit_type + 4
        self.zealot_id = self._min_rl_unit_type + 5
        self.stalker_id = self._min_rl_unit_type + 6
        self.zergling_id = self._min_rl_unit_type + 7
        self.hydralisk_id = self._min_rl_unit_type + 8

        self.rl_unit_map = {
            "baneling": self.baneling_id,
            "colossus": self.colossus_id,
            "hydralisk": self.hydralisk_id,
            "marauder": self.marauder_id,
            "marine": self.marine_id,
            "medivac": self.medivac_id,
            "stalker": self.stalker_id,
            "zealot": self.zealot_id,
            "zergling": self.zergling_id,
        }
        #
        self.melee_unit_map = {
            "baneling": Zerg.Baneling,
            "colossus": Protoss.Colossus,
            "hydralisk": Zerg.Hydralisk,
            "marauder": Terran.Marauder,
            "marine": Terran.Marine,
            "medivac": Terran.Medivac,
            "stalker": Protoss.Stalker,
            "zealot": Protoss.Zealot,
            "zergling": Zerg.Zergling,
        }

    def _get_unit_type(self, unit_name, rl_unit_flag=True):
        if rl_unit_flag:
            return self.rl_unit_map[unit_name]
        else:
            return self.melee_unit_map[unit_name]

    def _get_unit_name(self, unit_type, rl_unit_flag=True):
        """according unit type get corresponding unit name"""
        iterate_map = None
        if rl_unit_flag:
            iterate_map = self.rl_unit_map
        else:
            iterate_map = self.melee_unit_map
        for name, type_id in iterate_map.items():
            if type_id == unit_type:
                return name

    def _center_symmetry_shift(self, points):
        """根据中心点，计算出给定点的中心对称位置,
           支持同时传入多个points
        """
        points = np.array(points)
        center_point = np.array([self.center_x, self.center_y])
        # 获取带变换点相对于中心点的偏移量
        center_shift = points - center_point
        results = center_point - center_shift
        return results

    def _init_ally_unit_types(self, min_ally_unit_type, min_enemy_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        min_unit_type = min(min_ally_unit_type, min_enemy_unit_type)
        self._min_unit_type = min_unit_type
        # used record RL control agent's unit type id
        max_unit_type = max(min_ally_unit_type, min_enemy_unit_type)
        # if both camp agents are controlled by RL, use min unit type as base
        if "battle" in self.map_name:
            self._min_rl_unit_type = min_unit_type
        else:
            self._min_rl_unit_type = max_unit_type
        if self.map_type == "marines":
            self.marine_id = self._min_rl_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = self._min_rl_unit_type
            self.zealot_id = self._min_rl_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = self._min_rl_unit_type
            self.stalker_id = self._min_rl_unit_type + 1
            self.zealot_id = self._min_rl_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = self._min_rl_unit_type
            self.marine_id = self._min_rl_unit_type + 1
            self.medivac_id = self._min_rl_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = self._min_rl_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = self._min_rl_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = self._min_rl_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = self._min_rl_unit_type
        elif self.map_type == "bane":
            self.baneling_id = self._min_rl_unit_type
            self.zergling_id = self._min_rl_unit_type + 1

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_enemy_unit_by_id(self, e_id):
        """Get unit by ID."""
        return self.enemies[e_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = self.red_ally_state_attr_names
        env_info["enemy_features"] = self.blue_ally_state_attr_names
        return env_info

    @property
    def episode_steps(self):
        return self._episode_steps
