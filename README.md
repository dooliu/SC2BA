```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SC2BA (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10.
```

# SC2BA

# StarCraft+: Benchmarking Multi-Agent Algorithms in Adversary Paradigm

[SC2BA](https://github.com/dooliu/SC2BA) is an environment for research in the field of competitive multi-agent reinforcement learning (MARL) based on [Blizzard](http://blizzard.com)'s [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) RTS game. SC2BA makes use of Blizzard's [StarCraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and [DeepMind](https://deepmind.com)'s [PySC2](https://github.com/deepmind/pysc2) to provide a convenient interface for autonomous agents to interact with StarCraft II engine, getting observations and performing actions. Unlike the [PySC2](https://github.com/deepmind/pysc2), SC2BA concentrates on *decentralised micromanagement* scenarios like SMAC, where each unit of the game is controlled by an individual RL agent. However, the enemies in SMAC are actually controlled by built-in AI with fixed strategies, resulting in a lack of difficulty and challenge in the environment. This deficiency leads to insufficient diversity and generality in algorithm evaluation. SC2BA refresh the benchmarking of MARL algorithms in an adversary paradigm,  both multi-agent teams to be controlled by designed MARL algorithms in a continuous adversarial paradigm.

<img src="https://img-blog.csdnimg.cn/direct/bb8c78ac99e3453eb6e93277a41f74e5.png"></img>

Grounding in SC2BA, we benchmark those classic MARL algorithms in two types of adversarial modes: dual-algorithm paired adversary and multi-algorithm mixed adversary, where the former conducts the adversary of pairwise algorithms while the latter focuses on the adversary to multiple behaviors from a group of algorithms. The extensive benchmark experiments exhibit some thought-provoking observations/problems in the effectivity, sensibility and scalability of these completed algorithms.


Please refer to the accompanying paper for the outline of our motivation for using SC2BA as a testbed for MARL research and the initial experimental results.

## About

Together with SMAC we also release [APyMARL](https://github.com/dooliu/APyMARL) - our [PyTorch](https://github.com/pytorch/pytorch) framework for adversary MARL research, which includes implementations of several state-of-the-art and classical algorithms, such as DOP, QPLEX,  [QMIX](https://arxiv.org/abs/1803.11485) and [COMA](https://arxiv.org/abs/1705.08926).

Should you have any question, please reach to [lizishu@njust.edu.cn](lizishu@njust.edu.cn) or commit in issues.

# Quick Start

## Installing SC2BA

You can install SC2BA by using the following command:

```shell
pip install git+https://github.com/dooliu/SC2BA.git
```

Alternatively, you can clone the SMAC repository and then install `smac` with its dependencies:

```shell
git clone https://github.com/dooliu/SC2BA.git
pip install -e smac/
```

*NOTE*: If you want to extend SC2BA, release please install the package as follows:

```shell
git clone https://github.com/dooliu/SC2BA.git
cd smac
pip install -e ".[dev]"
pre-commit install
```

You may also need to upgrade pip: `pip install --upgrade pip` for the install to work.

## Installing StarCraft II

SC2BA is based on the full game of StarCraft II (versions >= 3.16.1). To install the game, follow the commands bellow.

### Linux

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

### MacOS/Windows

Please install StarCraft II from [Battle.net](https://battle.net). The free [Starter Edition](http://battle.net/sc2/en/legacy-of-the-void/) also works. PySC2 will find the latest binary should you use the default install location. Otherwise, similar to the Linux version, you would need to set the `SC2PATH` environment variable with the correct location of the game.

*NOTE*: If you are Chinese, due to the Bobby Kotick, CN play cant own their  sever again.  You can download StarCraft II by this [video](https://www.bilibili.com/video/BV1As4y147NP/?buvid=XY6E01868C47C929FEFCE4A6DBF0A4ECFFB64&is_story_h5=false&mid=y0%2Bkb3rZVEwQ9j34NFXkLA%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=android&share_plat=android&share_session_id=2e6181cb-fa27-4ce1-9b2e-f126f39267d5&share_source=COPY&share_tag=s_i&timestamp=1674580642&unique_k=rPeGgmE&up_id=149681985&vd_source=0553fe84b5ad759606360b9f2e687a01), and set Battle.net.

## SC2BA maps

SC2BA is composed of many combat scenarios with pre-configured maps. Before SMAC can be used, these maps need to be downloaded into the `Maps` directory of StarCraft II.

Download the [SMC2BA_Maps](https://github.com/dooliu/SC2BA/blob/main/sc2ba/env/starcraft2/maps/SC2BA_Maps/COMMON.SC2Map) and put it to your `$SC2PATH/Maps` directory. If you installed SMAC via git, simply copy the `SMAC_Maps` directory from `smac/env/starcraft2/maps/` into `$SC2PATH/Maps` directory.

### List the maps

To see the list of SC2BA maps, together with the number of ally and enemy units and episode limit, run:

```shell
python -m sc2ba.bin.map_list 
```

### Creating new maps

We integrate all combat units into one map, allowing any battle scenarios to be implemented within this unified map, while the previous paradigm employs an individual map file for each scene, which makes scene definition/modification tedious and error-prone.

The settings of battle force, multi-agent attributes as well as scene elements are completely formatted with text prompts, thus the operability of scene configuration could be greatly enhanced and the burden of algorithm developers is reduced.

So, we can create a new combat scenario through adding some text prompt in `sc2ba_maps` file.

```json
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
}
```

*NOTE*: SC2BA support nine units includes: Marines, Medivac, Marauders,  Colossus, Stalkers, Zealots, Zergling, Hydralisk and Baneling.

## Testing SC2BA

Please run the following command to make sure that `smac` and its maps are properly installed. 

```bash
python -m sc2ba.examples.random_agents
```

## Saving and Watching StarCraft II Replays

### Saving a replay

If you’ve using our [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent RL, here’s what needs to be done:

1. **Saving models**: We run experiments on *Linux* servers with `save_model = True` (also `save_model_interval` is relevant) setting so that we have training checkpoints (parameters of neural networks) saved (click [here](https://github.com/oxwhirl/pymarl#saving-and-loading-learnt-models) for more details).
2. **Loading models**: Learnt models can be loaded using the `checkpoint_path` parameter. If you run PyMARL on *MacOS* (or *Windows*) while also setting `save_replay=True`, this will save a .SC2Replay file for `test_nepisode` episodes on the test mode (no exploration) in the Replay directory of StarCraft II. (click [here](https://github.com/oxwhirl/pymarl#watching-starcraft-ii-replays) for more details).

If you want to save replays without using PyMARL, simply call the `save_replay()` function of SMAC's StarCraft2Env in your training/testing code. This will save a replay of all epsidoes since the launch of the StarCraft II client.

The easiest way to save and later watch a replay on Linux is to use [Wine](https://www.winehq.org/).

### Watching a replay

You can watch the saved replay directly within the StarCraft II client on MacOS/Windows by *clicking on the corresponding Replay file*.

# Documentation 

For the detailed description of the environment, read the [SMAC documentation](docs/smac.md). 

The initial results of our experiments using SMAC can be found in the [accompanying paper](https://arxiv.org/abs/1902.04043).

# Citing  SMAC 

If you use SMAC in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# Code Examples

Below is a small code example which illustrates how SMAC can be used. Here, individual agents execute random policies after receiving the observations and global state from the environment.  

If you want to try the state-of-the-art algorithms (such as [QMIX](https://arxiv.org/abs/1803.11485) and [COMA](https://arxiv.org/abs/1705.08926)) on SMAC, make use of [PyMARL](https://github.com/oxwhirl/pymarl) - our framework for MARL research.

```python
from smacbattle.env import StarCraft2Env
import numpy as np


def main():
    map_name = "3m"
    players = []
    map_params = get_map_params(map_name)
    # if play is controlled by built-in AI，class set to Bot, otherwise Agent
    players.append(Agent(map_params["a_race"], Camp.RED.name))
    agent2 = "Agent"
    if agent2 == "Bot":
        players.append(Bot(map_params["b_race"], Camp.BLUE.name + "(Computer)", difficulties["7"]))
    else:
        players.append(Agent(map_params["b_race"], Camp.BLUE.name))
    print(f"players value:{players}")
    env = StarCraft2BAEnv(map_name=map_name, players=players)
    env_info = env.get_env_info()
    print(f"Envs states is：{env_info}")
    # n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_enemies = env_info["n_enemies"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            red_state = env.get_state(camp=Camp.RED)
            red_obs = env.get_obs(camp=Camp.RED)
            # print(f"Red Camp get state:{red_state}")
            # print(f"Red Camp get obs:{red_obs}")
            blue_state = env.get_state(camp=Camp.BLUE)
            blue_obs = env.get_obs(camp=Camp.BLUE)
            # print(f"Blue Camp get state:{blue_state}")
            # print(f"Blue Camp get obs:{blue_obs}")
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            for enemy_id in range(n_enemies):
                avail_actions = env.get_avail_agent_actions(enemy_id, camp=Camp.BLUE)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            print(f"random_agent, 生成的actions：{actions}")
            reward, terminated, _ = env.step(actions)
            print(f"当前回合奖励值为：{reward}")
            episode_reward += reward[1]
            time.sleep(0.1)

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

```



