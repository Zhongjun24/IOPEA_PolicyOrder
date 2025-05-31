import random
import numpy as np
import itertools
from tqdm import tqdm
import numpy as np
import my_tools

class FeedbackGraph:
    def __init__(self, param_, dynamics_, X, S, A, H, S_hat, seed_, delta = 0.01):
        self.X = X  # set of state-action pairs
        self.S = S  # number of states
        self.A = A  # number of actions
        self.H = H  # episode length
        self.delta = delta
        self.S_hat = S_hat

        self.state_key = [tuple(s) for s in self.S]

        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.T = dynamics_.time_horizon

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.one_seed = None

        self.rng_factory = None

        # Initialize statistics
        self.n = {x: 0 for x in X}
        self.r_hat = {x: 0.0 for x in X}
        self.r2_hat = {x: 0.0 for x in X}
        self.P_hat = {x: np.eye(1, len(S), 0).flatten() for x in X}

        self.g = lambda G: np.array([G.get(k, 0.0) for k in self.state_key])
        self.sample_paths = None
        self.V_tilde = [{} for _ in range(self.H + 1)]
        self.V_bar = [{} for _ in range(self.H + 1)]
        self.pi_ = {}


    def _next_step(self, s, a, exo_info_flag, exo_info):
        if self.param_.current_problem == 'inventory_control':
            if not exo_info_flag:
                demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            elif exo_info_flag:
                demands = exo_info
            current_policy = np.sum(s) + a
            current_policy = np.clip(current_policy, self.policy_set[0][0], self.policy_set[0][1])
            temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= s[0], x2 = None, x3 = s[1:])
            current_cost = np.sum(temp_cost[: -self.param_.hyper_set['L']])
            return [crucial_state[1]] + pipeline_state[-self.param_.hyper_set['L']:].tolist(), current_cost, demands

        if self.param_.current_problem == 'dual_index':
            if not exo_info_flag:
                demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            elif exo_info_flag:
                demands = exo_info
            current_policy = [np.sum(s[:2 * self.param_.hyper_set['l'] + 2]) + a[0], np.sum(s) + a[0] + a[1]]
            current_policy = np.clip(current_policy, [self.policy_set[0][0], self.policy_set[1][0]], [self.policy_set[0][1], self.policy_set[1][1]])
            temp_cost, _, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= s[0], x2 = s[1:self.param_.hyper_set['l'] + 1], x3 = s[self.param_.hyper_set['l'] + 1:])
            current_cost = np.sum(temp_cost[: -self.param_.hyper_set['L']])
            return [crucial_state[1]] + shortline_state[2:2 + self.param_.hyper_set['l']].tolist() + pipeline_state[-self.param_.hyper_set['L']:].tolist(), current_cost, demands

    #only works for our Exo-MDPs, thus we dont' need s_next and r to infer the rich feedbacks.
    def _get_richfeedback(self, s, exo_info, a = None, s_next = None, r = None):
        if self.param_.current_problem == 'inventory_control':
            rich_feedback_list = []
            for s_prime in self.S:
                #condition of existence of edge from s to s'
                if s_prime[0] + s_prime[1] <= s[0] + s[1]:
                    for a_prime in self.A:
                        if not (tuple(s_prime), a_prime) in self.P_hat:
                            continue
                        s_next, r, _ = self._next_step(s_prime, a_prime, exo_info_flag = 1, exo_info = exo_info)
                        rich_feedback_list.append([(tuple(s_prime), a_prime), 1 - r, s_next])
            return rich_feedback_list

        if self.param_.current_problem == 'dual_index':
            rich_feedback_list = []
            for s_prime in self.S:
                for a_prime in self.A:
                    if s_prime[0] + s_prime[1] + a_prime[0] <= s[0] + s[1] + a[0]:
                        if not (tuple(s_prime), a_prime) in self.P_hat:
                            continue
                        s_next, r, _ = self._next_step(s_prime, a_prime, exo_info_flag = 1, exo_info = exo_info)
                        rich_feedback_list.append([(tuple(s_prime), a_prime), 1 - r, s_next])
            return rich_feedback_list

    def m(self, s):
        s = np.array(s)
        return  tuple(min(self.S, key=lambda x: np.sum(np.abs(x - s))))

    def run(self):
        current_cost = 0
        current_t = 0
        updating_flag = 0
        s = self.S[0]
         
        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()
        with tqdm(total=self.T, desc="Running", unit="step") as pbar:
            while current_t <= self.T:
                if current_t in [self.T * i // 100 for i in range(10, 101, 10)]:
                    self.optimist_plan(updating_flag, first_stage = 1)
                else:
                    self.optimist_plan(updating_flag, first_stage = 0)
                for h in range(1, self.H + 1):
                    a = self.pi_[(self.m(tuple(s)), h)]
                    s_next, r, exo_info = self._next_step(s, a, exo_info_flag = 0, exo_info = None)
                    
                    obs_set = self._get_richfeedback(s = s, exo_info = exo_info, a = a, s_next = None, r = None)  # returns a list of (x, r, s')
                    for x, r_obs, s_prime in obs_set:
                        self.update_stats(x, r_obs, s_prime)
        
                    s = s_next
                    current_cost += r
                    current_t += 1

                    pbar.update(1)

                updating_flag = 1
                self.sample_paths = list(set([v[0][0] for v in obs_set]))
        return current_cost/(current_t + 1e-6), self.evaluate()

    def update_stats(self, x, r, s_prime):
        n_old = self.n[x]
        self.n[x] += 1
        n_new = self.n[x]

        self.r_hat[x] = (n_old * self.r_hat[x] + r) / n_new
        self.r2_hat[x] = (n_old * self.r2_hat[x] + r**2) / n_new

        e_s_prime = np.zeros(len(self.S))
        i_ = self.state_key.index(self.m(tuple(s_prime)))
        e_s_prime[i_] = 1.0
        self.P_hat[x] = (n_old * self.P_hat[x] + e_s_prime) / n_new

    def optimist_plan(self, updating_flag, first_stage):

        def sample_tuples(my_list, percent):
            total = len(my_list)
            k = int(percent * total)
            indices = self.one_seed.choice(total, size=k, replace=False)
            return [my_list[i] for i in indices]

        for h in reversed(range(1, self.H + 1)):
            if not updating_flag:
                s_set_enumerate = self.S   
            else:
                update_ratio = 1 if first_stage else self.param_.feedbackgraph_update_fraction
                s_set_enumerate = sample_tuples(self.sample_paths, update_ratio)
            for s in s_set_enumerate:
                Q_ucb, Q_lcb = [], []
                for a in self.A:
                    x = (tuple(s), a)
                    if x not in self.n:
                        continue
                    if self.n[x] == 0 or h == self.H:
                        q_ucb = self.H - h + 1
                        q_lcb = -(self.H - h + 1)
                        
                    else:
                        bonus = self.compute_bonus(x, self.V_tilde[h], self.V_tilde[h], self.V_bar[h])
                        P_x = self.P_hat[x]
                        r_x = self.r_hat[x]
                        q_ucb = r_x + np.dot(P_x, self.g(self.V_tilde[h])) + bonus
                        q_lcb = r_x + np.dot(P_x, self.g(self.V_bar[h])) - bonus
                    q_ucb = np.clip(q_ucb, 0, self.H - h + 1)
                    q_lcb = np.clip(q_lcb, 0, self.H - h + 1)
                    Q_ucb.append((q_ucb, a))
                    Q_lcb.append((q_lcb, a))
                max_val = max(xi[0] for xi in Q_ucb)
                candidates = [i for i in range(len(Q_ucb)) if Q_ucb[i][0] == max_val]
                k = random.choice(candidates)
                self.pi_[(tuple(s), h)] = Q_ucb[k][1]
                self.V_tilde[h - 1][tuple(s)] = Q_ucb[k][0]
                self.V_bar[h - 1][tuple(s)] = Q_lcb[k][0]

        return None

    def compute_bonus(self, x, V_next, V_tilde_next, V_bar_next):

        g = lambda G: np.array([G.get(k, 0.0) for k in self.state_key])
        n_x = self.n[x]
        if n_x == 0:
            return float('inf')
    
        # reward and next-state value variance
        var_r = max(self.r2_hat[x] - self.r_hat[x] ** 2, 0)
        var_v = max(np.dot(self.P_hat[x], self.g(V_next) ** 2) - np.dot(self.P_hat[x], self.g(V_next)) ** 2, 0)
        eta = np.sqrt(var_r) + np.sqrt(var_v)
    
        log_term = np.log(len(self.X) * self.H * np.log(max(n_x, 2)) / self.delta)
    
        # bonus terms
        term_1 = (1 / self.H) * np.dot(self.P_hat[x], self.g(V_tilde_next) - self.g(V_bar_next))
        term_2 = np.sqrt(eta / n_x * log_term)
        term_3 = (self.S_hat * self.H ** 2 / n_x) * log_term
    
        return (term_1 + term_2 + term_3) * self.param_.bouns_scale_factor

    def close(self):
        del self.large_array

    def evaluate(self):
        """
        Evaluate the final learned policy by rollout self.T steps
        with h=0 fixed (since this is an infinite horizon setting).
        """
        s = self.S[0]  # initial state
        eval_total_reward = 0
        eval_steps = 0
    
        # Plan the final optimistic policy
        print('===feedback graph (Dann 2020) testing===')
        with tqdm(total=self.T, desc="Running", unit="step") as pbar:
            while eval_steps < self.param_.testing_horizon:
                h_eval = 1  # fixed h=0 for evaluation
                if (tuple(s), h_eval) in self.pi_:
                    a = self.pi_[(tuple(s), h_eval)]
                else:
                    # fallback if (s, 0) is not in pi_: pick any valid action
                    a = self.pi_[(self.m(tuple(s)), h_eval)]
                s_next, r, _ = self._next_step(s, a, exo_info_flag=0, exo_info=None)
                eval_total_reward += r
                s = s_next
                eval_steps += 1
                pbar.update(1)
    
        avg_eval_reward = eval_total_reward / (eval_steps + 1e-6)
        return avg_eval_reward
