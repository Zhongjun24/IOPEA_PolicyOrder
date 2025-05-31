import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import queuing_model
import baselines
import my_tools
import torch
from tqdm import tqdm


class DVFSGymEnv(gym.Env):
    """
    Gym wrapper for your DVFSMDP model.
    """
    metadata = {'render.modes': []}

    def __init__(self, param_, dynamics_,seed_):
        super().__init__()
        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.step_count = 0 
        self.horizon = 50
        self.state = 0
        self.initial_state = 0

        self.observation_space = spaces.Discrete(param_.hyper_set['S'] + 1)
        self.action_space = spaces.Discrete(param_.hyper_set['Amax'] + 1)

        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()


    def reset(self, *, seed=None, options=None):
        self.step_count = 0

        obs = np.array(self.state, dtype=np.int32)
        info = {}

        return obs, info

    def step(self, action):
        a = int(np.clip(action, 0, self.dynamics_.Amax))
        next_s, reward = self.dynamics_.sample(self.state, a, self.one_seed)
        self.state = next_s
        self.step_count += 1

        done = (self.step_count >= self.horizon)
        terminated = False        # no “natural” terminal in your MDP
        truncated  = done         # only horizon causes episode end

        obs   = np.array(self.state, dtype=np.int32)
        info  = {}                # or some dict if you need

        return obs, (self.dynamics_.rmax - reward)/self.dynamics_.rmax, terminated, truncated, info


class PPOKernel:
    """
    Encapsulates PPO training and evaluation for a DVFSMDP.
    """
    def __init__(self, param_, dynamics_, seed_):
        self.env_fn = lambda: DVFSGymEnv(param_, dynamics_, seed_=seed_)
        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.ppo_params = {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 256,
            'batch_size': 64,
            'gamma': 0.99,
            'ent_coef': 1e-4,
            'verbose': 0,
            'device': 'cpu'
        }
        self.model = None
        self.total_timesteps = self.dynamics_.time_horizon


    def run(self):
        vec_env = DummyVecEnv([self.env_fn])
        self.model = PPO(
            env=vec_env,
            **self.ppo_params)
        self.model.learn(total_timesteps=self.total_timesteps)
        return self.evaluate(self.model)

    def evaluate(self, model):
        print('===PPO: evaluation===')
        env = self.env_fn()
        state = env.initial_state
        total_cost = 0
        steps = 0

        model.policy.eval()
        with torch.no_grad():
            with tqdm(total=self.param_.testing_horizon, desc="Running", unit="step") as pbar:
                while steps < self.param_.testing_horizon:
                    action, _ = model.predict(np.array([state], dtype=np.int32), deterministic=True)
                    next_state, reward, _, _, _= env.step(action)
                    total_cost += (self.dynamics_.rmax - self.dynamics_.rmax * reward)
                    state = next_state
                    steps += 1
                    pbar.update(1)

        avg_cost = total_cost / (steps + 1e-6)
        return avg_cost, avg_cost

