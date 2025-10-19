import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import my_tools
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import torch.nn as nn

optuna.logging.set_verbosity(optuna.logging.WARNING)






class ContinuousEnv(gym.Env):
    def __init__(self, param_, dynamics_, S_box, A_box, dimension_, H, seed_=0):
        super().__init__()
        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.H = H
        self.step_count = 0
        self.state = [param_.hyper_set['policy_set'][0][0]] * dimension_

        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()

        self.T = dynamics_.time_horizon
        self.initial_state = [param_.hyper_set['policy_set'][0][0]] * dimension_

        self.observation_space = spaces.Box(
            low=np.full(dimension_, S_box[0][0], dtype=np.float32),
            high=np.full(dimension_, S_box[0][1], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in A_box], dtype=np.float32),
            high=np.array([b[1] for b in A_box], dtype=np.float32),
            dtype=np.float32
        )

        self.L = self.param_.hyper_set['L']


    def reset(self):
        self.step_count = 0
        return np.array(self.state, dtype=np.float32) if self.state is not None else np.array(self.initial_state, dtype=np.float32)

        '''
        base = np.array(self.initial_state, dtype=np.float32)
        noise = self.np_random.normal(loc=0, scale=0, size=base.shape)
        self.state = np.clip(base + noise, self.observation_space.low, self.observation_space.high)
        return np.array(self.state, dtype=np.float32) if self.state is not None else np.array(self.initial_state, dtype=np.float32)
        '''

    def step(self, action):
        s = self.state if self.state is not None else self.initial_state

        demands = self.dynamics_.demand_func(
            size=1 + self.L,
            distribution=self.param_.hyper_set['distribution'],
            maximum_demand=self.param_.hyper_set['maximum_demand'],
            one_seed=self.one_seed
        )

        if self.param_.current_problem == 'inventory_control':
            current_policy = np.sum(s) + action
            current_policy = np.clip(current_policy, self.action_space.low, self.action_space.high)
            temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(
                1 + self.L, current_policy, demands,
                if_initial_state=1, x1=s[0], x2=None, x3=s[1:]
            )
            cost = np.sum(temp_cost[:-self.L])
            next_state = [crucial_state[1]] + pipeline_state[-self.L:].tolist()

        elif self.param_.current_problem == 'dual_index':
            l = self.param_.hyper_set['l']
            current_policy = [np.sum(s[:2 * self.param_.hyper_set['l'] + 2]) + action[0], np.sum(s) + action[0] + action[1]]
            current_policy = np.clip(current_policy, self.action_space.low, self.action_space.high)
            temp_cost, _, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(
                1 + self.L, current_policy, demands,
                if_initial_state=1, x1=s[0], x2=s[1:l + 1], x3=s[l + 1:]
            )
            cost = np.sum(temp_cost[:-self.L])
            next_state = [crucial_state[1]] + shortline_state[2:2 + l].tolist() + pipeline_state[-self.L:].tolist()

        else:
            raise NotImplementedError("Unknown problem type")

        reward = 1-cost
        self.state = next_state
        self.step_count += 1
        done = self.step_count >= self.H
        return np.array(self.state, dtype=np.float32), reward, done, {}

class ActionNormWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_low, self.orig_high = env.action_space.low, env.action_space.high
        # Agent “看到” [-1,1]
        self.action_space = spaces.Box(-1., 1., shape=self.orig_low.shape, dtype=np.float32)

    def action(self, a_norm):
        # Agent 输出 a_norm ∈ [-1,1] → Env 接收 a_raw ∈ [low,high]
        return self.orig_low + (a_norm + 1.0) * 0.5 * (self.orig_high - self.orig_low)

class PPOTrainer:
    def __init__(self, env_class, param_, dynamics_, S_box, A_box, dimension_, H, seed_=0):
        self.env_fn = lambda: env_class(param_, dynamics_, S_box, A_box, dimension_, H, seed_=seed_)
        self.param_ = param_
        self.dynamics_ = dynamics_
        self.S_box = S_box
        self.A_box = A_box
        self.H = H
        self.seed_ = seed_
        self.best_params = None

        self.n_envs = 8

    def run(self, total_timesteps=None, tune=True):
        # 调参略...
        # 1) 构造训练环境：ActionNorm + VecNormalize + DummyVecEnv
        def make_env():
            return ActionNormWrapper(self.env_fn())
        vec_env = DummyVecEnv([make_env for _ in range(self.n_envs)])
        vec_env = VecNormalize(vec_env,
                               norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0)

        policy_kwargs = dict(
            net_arch=[256,256,128,128,64,64],
            activation_fn=nn.ReLU,
            log_std_init=np.log(0.1)  # 初始 σ≈100
        )
        model = PPO("MlpPolicy", vec_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=1e-4,
                    n_steps=512,
                    batch_size=32,
                    gamma=0.9,
                    ent_coef=1e-4,
                    verbose=1, device='cpu')

        model.learn(total_timesteps=int(self.dynamics_.time_horizon))

        # 2) 评估：手写循环，保留包装，逐步打印
        print("=== PPO: manual evaluation ===")

        def make_eval_env():
            # self.env_fn() 返回一个 ContinuousEnv 实例
            return ActionNormWrapper(self.env_fn())

        # 2. 用 DummyVecEnv 把它变成 VecEnv（内部有 num_envs=1）
        eval_vec_env = DummyVecEnv([make_eval_env])

        # 3. 套上 VecNormalize，做 obs/reward 归一化
        eval_vec_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        # 评估时不再更新统计量
        eval_vec_env.training    = False
        eval_vec_env.norm_reward = False

        # 4. 手写评估循环
        obs_batch = eval_vec_env.reset()  # 返回 shape (1, obs_dim)
        total_cost = 0
        for i in range(self.param_.testing_horizon):
            # 传入 batch
            a_norm_batch, _ = model.predict(obs_batch, deterministic=True)
            # a_norm_batch.shape == (1, action_dim)
            
            next_obs_batch, r_norm_batch, done_batch, infos = eval_vec_env.step(a_norm_batch)
            # 都是 batch 形式，需要取第 0 个元素
            obs = next_obs_batch[0]
            r_norm = r_norm_batch[0]
            total_cost += 1 - r_norm
            done = done_batch[0]
            info = infos[0]

            # 如果想看原始 env 的内部状态和映射后的动作：
            raw_env = eval_vec_env.venv.envs[0]        # 拿到你的 ContinuousEnv
            raw_state = raw_env.state                 # 它的内部 state
            low, high = raw_env.action_space.low, raw_env.action_space.high
            a_raw = low + (a_norm_batch[0] + 1.0) * 0.5 * (high - low)
            obs_batch = next_obs_batch

        avg_cost = total_cost / (i + 1e-6)
        print(f"Average evaluation cost: {avg_cost:.4f}")
        return avg_cost, avg_cost