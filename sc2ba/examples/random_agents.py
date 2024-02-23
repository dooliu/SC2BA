from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from sc2ba.env import StarCraft2BAEnv, Agent, Bot, Camp, difficulties
import numpy as np

from sc2ba.env.starcraft2.maps import get_map_params


def main():
    map_name = "3m"
    players = []
    map_params = get_map_params(map_name)
    # Agent的class
    players.append(Agent(map_params["a_race"], Camp.RED.name))
    # 如果玩家2是机器人，则设置为电脑玩家
    agent2 = "Agent"
    # 如果玩家2是机器人，则设置为电脑玩家
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
        print(f"环境初始化成功！")
        terminated = False
        episode_reward = 0

        while not terminated:
            red_state = env.get_state(camp=Camp.RED)
            red_obs = env.get_obs(camp=Camp.RED)
            # print(f"Red Camp get state:{red_state}")
            # print(f"Red Camp get obs:{red_obs}")
            # 获取全局状态，用于集中训练（红蓝双方应该有各自的全局状态）
            blue_state = env.get_state(camp=Camp.BLUE)
            blue_obs = env.get_obs(camp=Camp.BLUE)
            # print(f"Blue Camp get state:{blue_state}")
            # print(f"Blue Camp get obs:{blue_obs}")
            # env.render()  # Uncomment for rendering
            # observe_space = env.get_obs_agent(1)
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
            # print(f"env中的全局状态为：{env.get_state()}")
            reward, terminated, _ = env.step(actions)
            print(f"当前回合奖励值为：{reward}")
            episode_reward += reward[1]
            time.sleep(0.1)

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
