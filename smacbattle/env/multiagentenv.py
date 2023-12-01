from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smacbattle.env.starcraft2.enums import Camp


class MultiAgentEnv(object):
    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        n_red_actions, n_blue_actions = self.get_total_actions()
        env_info = {
            "state_red_shape": self.get_state_size(),
            "state_blue_shape": self.get_state_size(Camp.BLUE),
            "obs_red_shape": self.get_obs_size(),
            "obs_blue_shape": self.get_obs_size(Camp.BLUE),
            "n_red_actions": n_red_actions,
            "n_blue_actions": n_blue_actions,
            "n_agents": self.n_red_agents,
            "n_enemies": self.n_blue_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
