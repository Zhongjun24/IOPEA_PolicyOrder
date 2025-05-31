import baselines
import numpy as np
import my_tools
from tqdm import tqdm
import copy



class UCRL2Learner:
    def __init__(self, param_, dynamics_, delta, measure, seed_):
        self.M = dynamics_
        self.param_ = param_
        self.S = param_.hyper_set['S'] + 1
        self.A = param_.hyper_set['Amax'] + 1
        self.delta = delta
        self.measure = measure
        self.seed_ = seed_
        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()

        self.horizon = self.param_.hyper_set['time_horizon']

        self.evaluation = baselines.PolicyEvaluation(self.param_, self.M, one_seed = self.one_seed)

        # time and state
        self.t = 1
        self.s = None

        # counts
        self.N = np.zeros((self.S, self.A), dtype=int)
        self.nu = np.zeros((self.S, self.A), dtype=int)
        self.R_sum = np.zeros((self.S, self.A), dtype=float)
        self.P_count = np.zeros((self.S, self.A, self.S), dtype=int)

        # estimates and bounds
        self.r_hat = np.zeros((self.S, self.A))
        self.p_hat = np.zeros((self.S, self.A, self.S))
        self.beta_r = np.zeros((self.S, self.A))
        self.beta_p = np.zeros((self.S, self.A))

        # current policy
        self.pi = np.zeros(self.S, dtype=int)

    def initialize(self, s1: int):
        self.s = s1
        self.t = 1
        self.N.fill(0)
        self.R_sum.fill(0)
        self.P_count.fill(0)

    def _start_new_episode(self):
        # snapshot counts
        N_tk = self.N.copy()
        self.nu.fill(0)

        # empirical r_hat, p_hat
        for s in range(self.S):
            for a in range(self.A):
                n = max(1, N_tk[s,a])
                self.r_hat[s,a] = self.R_sum[s,a]/n
                self.p_hat[s,a,:] = self.P_count[s,a,:]/n

        # confidence intervals
        t_k = self.t
        log_r = np.log(2*self.S*self.A*t_k/self.delta)
        log_p = np.log(2*self.A*t_k/self.delta)
        for s in range(self.S):
            for a in range(self.A):
                n = max(1, N_tk[s,a])
                self.beta_r[s,a] = self.M.rmax * np.sqrt(7*log_r/(2*n))
                self.beta_p[s,a] = np.sqrt(14*self.S*log_p/n)

        # sample parameters and set policy
        self.discretize_by_parameters()

    def run(self, s1 = 0):
        self.initialize(s1)
        episode = 0
        with tqdm(total=self.horizon, desc="Running", unit="step") as pbar:
            while self.t <= self.horizon:
                episode += 1
                self._start_new_episode()
                t_count = 0
                while self.t <= self.horizon:
                    s, a = self.s, self.pi[self.s]
                    if self.nu[s,a] >= max(1, self.N[s,a]):
                        break
                    s_next, r = self.M.sample(s,a, self.one_seed)
                    # update
                    self.N[s,a] += 1
                    self.nu[s,a] += 1
                    self.R_sum[s,a] += r
                    self.P_count[s,a,s_next] += 1
                    self.s = s_next
                    self.t += 1
                    t_count += 1
                pbar.update(t_count)
        return 1, self.evaluation.run(self.pi)

    def evaluate_policy(self, policy, r_mat=None, p_mat=None, num_iters=100):
        r_est = r_mat if r_mat is not None else self.r_hat
        p_est = p_mat if p_mat is not None else self.p_hat
        P = np.zeros((self.S, self.S))
        r_vec = np.zeros(self.S)
        for s in range(self.S):
            a = policy[s]
            r_vec[s] = r_est[s,a]
            P[s,:] = p_est[s,a,:]
        pi = np.ones(self.S)/self.S
        for _ in range(num_iters):
            pi = pi @ P
        return pi @ r_vec

    def discretize_by_parameters(self):
        measure = min(self.measure ** 2, 100)
        best_val = np.inf
        best_pol = None
        for _ in range(measure):
            # reward perturbation
            r_tilde = self.r_hat + self.one_seed.uniform(-1,1,self.r_hat.shape)*self.beta_r
            # transition perturbation
            p_tilde = np.zeros_like(self.p_hat)
            for s in range(self.S):
                for a in range(self.A):
                    noise = self.one_seed.uniform(-1,1,self.S)*self.beta_p[s,a]
                    vec = np.maximum(self.p_hat[s,a,:]+noise, 0)
                    if vec.sum()>0:
                        vec /= vec.sum()
                    else:
                        vec = np.ones(self.S)/self.S
                    p_tilde[s,a,:] = vec
            # greedy on r_tilde

            min_cost_inner = np.inf
            best_policy_inner = None
            current_policy_set = self.M.policy_set

            for i, current_policy in enumerate(current_policy_set):
                current_cost = self.evaluate_policy(policy = current_policy, r_mat = r_tilde, p_mat = p_tilde)

                if current_cost < min_cost_inner:
                    min_cost_inner = current_cost
                    best_policy_inner = current_policy

            if min_cost_inner < best_val:
                best_val = min_cost_inner
                best_pol = best_policy_inner

        self.pi = copy.deepcopy(best_pol)
        return 0

