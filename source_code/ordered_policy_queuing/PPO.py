import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from tqdm import tqdm
import my_tools


'''
if torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS M1/M2 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA CUDA GPU
else:
    device = torch.device("cpu")  # CPU fallback

print(f"Using device: {device}")
'''

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        x = self.shared(state)
        return self.policy_head(x), self.value_head(x)

class PPOAgent:
    def __init__(self, action_space, state_keys, action_dim, gamma=0.99, clip_eps=0.2, lr=1e-3, epochs=20, s_a_pair = None):
        self.action_space = action_space
        self.state_keys = state_keys
        self.state_dim = len(state_keys)
        self.action_dim = action_dim
        self.model = PPOPolicy(self.state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs

        self.valid_pairs = s_a_pair

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def one_hot(self, s):
        e = np.zeros(self.state_dim, dtype=np.float32)
        idx = self.state_keys.index(tuple(s))
        e[idx] = 1.0
        return e

    def valid_action_mask(self, s_tuple):
        mask = torch.tensor([((tuple(s_tuple), a) in self.valid_pairs) for a in self.action_space], dtype=torch.bool)
        return mask

    def act(self, state):
        state_v = torch.FloatTensor(self.one_hot(state))
        probs, value = self.model(state_v)

        mask = self.valid_action_mask(state)
        probs = probs.masked_fill(~mask, 0.0)
        if probs.sum().item() == 0:
            probs = torch.ones_like(probs) / len(probs)  # fallback: uniform dist
        else:
            probs = probs / probs.sum()

        dist = Categorical(probs)
        action_idx = dist.sample()
        action = self.action_space[action_idx.item()]
        return action, dist.log_prob(action_idx), value.detach()

    def store(self, state, action, reward, log_prob, value):
        action_idx = self.action_space.index(action)
        self.states.append(torch.FloatTensor(state))
        self.actions.append(action_idx)
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob.detach())
        self.values.append(value.float())

    def finish_path_and_update(self):
        returns, advantages = [], []
        G = 0
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(self.values).squeeze().detach()  # Detach values here
        advantages = returns - values
        #print(np.max(np.array(values)), np.min(np.array(values)))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack([torch.FloatTensor(s) for s in self.states])
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs)

        for _ in range(self.epochs):
            new_probs, new_values = self.model(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

class FuncPPO:
    def __init__(self, param_, dynamics_, X, S, A, H, seed_):
        self.action_list = A
        self.X = X
        self.S = S  
        self.H = H  
        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_

        self.rng_factory = None
        self.one_seed = None

        self.T = dynamics_.time_horizon
        self.state_key = [tuple(s) for s in self.S]
        self.initial_state = self.state_key[0]

    def _next_step(self, s, a):
        if self.param_.current_problem == 'inventory_control':
            demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            current_policy = np.sum(s) + a
            temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= s[0], x2 = None, x3 = s[1:])
            current_cost = np.sum(temp_cost[: -self.param_.hyper_set['L']])
            return [crucial_state[1]] + pipeline_state[-self.param_.hyper_set['L']:].tolist(), current_cost

        if self.param_.current_problem == 'dual_index':
            demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            current_policy = [np.sum(s[:2 * self.param_.hyper_set['l'] + 2]) + a[0], np.sum(s) + a[0] + a[1]]
            #validation check
            if current_policy[0] > current_policy[1] or np.max(current_policy) > self.param_.hyper_set['policy_set'][1][1]:
                print('dual index PPO algorithm poilicy check failed')
            temp_cost, _, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= s[0], x2 = s[1:self.param_.hyper_set['l'] + 1], x3 = s[self.param_.hyper_set['l'] + 1:])
            current_cost = np.sum(temp_cost[: -self.param_.hyper_set['L']])
            return [crucial_state[1]] + shortline_state[2:2 + self.param_.hyper_set['l']].tolist() + pipeline_state[-self.param_.hyper_set['L']:].tolist(), current_cost


    def run(self):
        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()
        agent = PPOAgent(
            action_space=self.action_list, 
            state_keys=self.state_key, 
            action_dim=len(self.action_list), 
            s_a_pair=self.X
        )
        current_cost = 0
        current_t = 0
        state = self.initial_state
    
        buffer_counter = 0   
        batch_size = 2048   
    
        with tqdm(total=self.T, desc="Running", unit="step") as pbar:
            while current_t <= self.T:
                action, log_prob, value = agent.act(state=state)
                next_state, reward = self._next_step(state, action)
                current_cost += reward
                current_t += 1
                buffer_counter += 1
    
                agent.store(agent.one_hot(state), action, 1 - reward, log_prob, value)
                state = next_state
                pbar.update(1)
    
                if buffer_counter >= batch_size:
                    agent.finish_path_and_update()
                    buffer_counter = 0   
    
        if buffer_counter > 0:
            agent.finish_path_and_update()

        '''
        print("\nFinal learned policy (state -> action probabilities):")
        agent.model.eval() 
        with torch.no_grad():
            for state_tuple in self.state_key[0:10] + self.state_key[-10:]:
                state_vector = torch.FloatTensor(agent.one_hot(state_tuple))
                action_probs, _ = agent.model(state_vector)
                action_probs = action_probs.cpu().numpy()
        
                print(f"State {state_tuple}:")
                for action, prob in zip(self.action_list, action_probs):
                    print(f"  Action {action}: {prob:.4f}")

        '''
        eval_state = self.initial_state
        eval_total_reward = 0
        eval_steps = 0
    
        agent.model.eval()
        with torch.no_grad():
            print('===PPO: evaoluation===')
            with tqdm(total=self.param_.testing_horizon, desc="Running", unit="step") as pbar:
                while eval_steps < self.param_.testing_horizon:
                    action, _, _ = agent.act(state=eval_state)  
                    next_state, reward = self._next_step(eval_state, action)
                    eval_total_reward += reward  
                    eval_state = next_state
                    eval_steps += 1
                    pbar.update(1)
        avg_eval_reward = eval_total_reward / (eval_steps + 1e-6)

        return current_cost / (current_t + 1e-6), avg_eval_reward
        

    def close(self):
        del self.large_array

