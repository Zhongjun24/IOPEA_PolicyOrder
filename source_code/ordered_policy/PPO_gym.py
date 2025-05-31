import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import my_tools
from stable_baselines3.common.vec_env import DummyVecEnv
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

    def tune_hyperparameters(self, n_trials=10, timeout=None):
        def objective(trial):
            lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            gamma = trial.suggest_uniform('gamma', 0.90, 0.9999)
            ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)

            vec_env = DummyVecEnv([self.env_fn])
            model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef,
                verbose=0,
                device='cpu'
            )

            model.learn(total_timesteps=int(1e4))

            eval_env = self.env_fn()
            mean_reward, _ = evaluate_policy(model, eval_env, 
                                             n_eval_episodes=10,
                                             deterministic=True)
            return mean_reward

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.best_params = study.best_params
        #print("Best hyperparameters:", self.best_params)

    def run(self, total_timesteps=None, model_path="ppo_model", tune=True):
        if tune:
            self.tune_hyperparameters()

        hp = self.best_params or {
            'learning_rate': 1e-3,
            'n_steps': 100,
            'batch_size': 32,
            'gamma': 0.999,
            'ent_coef': 1e-4
        }

        total_timesteps = self.dynamics_.time_horizon
        vec_env = DummyVecEnv([self.env_fn])
        policy_kwargs = dict(net_arch=[256, 256, 128],activation_fn=nn.ReLU)
        model = PPO(
            "MlpPolicy", vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=hp['learning_rate'],
            n_steps=hp['n_steps'],
            batch_size=hp['batch_size'],
            gamma=hp['gamma'],
            ent_coef=hp['ent_coef'],
            verbose=0,
            device='cpu'
        )

        model.learn(total_timesteps=total_timesteps)
        return self.evaluate(model)

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
                    action, _ = model.predict(state, deterministic=True)
                    next_state, reward, _, _ = env.step(action)
                    total_cost += (1 - reward)
                    state = next_state
                    steps += 1
                    pbar.update(1)

        avg_cost = total_cost / (steps + 1e-6)
        print(f"Average evaluation cost: {avg_cost:.4f}")
        return avg_cost, avg_cost