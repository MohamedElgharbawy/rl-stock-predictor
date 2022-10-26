import numpy as np

from MLP_policy import MLPPolicy
from utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule

class TradingAgent(object):

    def __init__(self, env, params):
        self.env = env
        self.window_size = window_size
        self.policy = MLPPolicy(ac_dim=1,
                ob_dim=window_size,
                n_layers=2,
                size=32,
                discrete=False,
                learning_rate=1e-4,
                training=True,
                nn_baseline=False)
        self.rl_trainer = RL_Trainer(self.params)
        # self.replay_buffer = MemoryOptimizedReplayBuffer(
        #     agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)

    # def add_to_replay_buffer(self, paths):
    #     pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
          obs = self.env.reset()
          self.last_obs = obs

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        
        

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
        self.policy.update(self)

        return log
