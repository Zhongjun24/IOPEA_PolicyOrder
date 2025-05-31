import numpy as np
from itertools import product
import itertools
import os
import time
import uuid



class InventoryControl:
    def __init__(self, policy_set, time_horizon, holding_cost, shortage_penalty, purchasing_cost, demand_zero_rate, L, l, maximum_demand, distribution, purchasing_cost_expedit = 0):
        self.policy_set = policy_set
        self.time_horizon = time_horizon
        self.holding_cost = holding_cost
        self.shortage_penalty = shortage_penalty
        self.purchasing_cost = purchasing_cost
        self.demand_zero_rate = demand_zero_rate
        self.L = L 
        self.l = l
        self.maximum_demand = maximum_demand
        self.distribution = distribution
        self.purchasing_cost_expedit = purchasing_cost_expedit

        self.min_cost = -(self.holding_cost + self.shortage_penalty) * self.policy_set[-1][1]
        self.max_cost = (self.holding_cost + self.purchasing_cost)  * self.policy_set[-1][1] + purchasing_cost_expedit * self.policy_set[0][1]


    def _generate_r_net(self, b, r):
        """Generate r-net on [0, b]"""
        return np.round(np.arange(0, b + r, r), 10)  # round to avoid float imprecision
    
    def construct_state_action_space(self, b, L, r, l = None):
        """Construct (S, A, X) for given b, L, r"""
        grid = self._generate_r_net(b, r)
        
        # State space: all r-grid points in [0, b]^L
        if l == None:
            S = [np.array(s) for s in product(grid, repeat=L)]
            A = [a for a in grid]
        else:
            S = [np.array(s) for s in product(grid, repeat=(L + l))]
            A0 = [a for a in grid]
            A = list(itertools.product(A0, A0))
            A = [x for x in A if x[0] <= x[1]]
    
        # Valid (s, a): sum(s) + a <= b
        X = []
        for s in S:
            for a in A:
                if np.sum(s) + np.max(a) <= b:
                    X.append((tuple(s), a))


        S_subset = [s for s in S if tuple(s) in {x[0] for x in X}]
    
        return S_subset, A, X

    def _cost_normalize(self, v, boundary_checking):
        v = (v - self.min_cost)/(self.max_cost - self.min_cost + 1e-6)
        if boundary_checking == None:
            if v < 0 or v > 1:
                print('cost out of range [0,1]? memory leak signal 138')
        return v

    def denormalize_(self, results, one_seed):
        erm_mean = np.mean(self.demand_func(size = 100000, distribution = self.distribution, maximum_demand = self.maximum_demand, one_seed = np.random.default_rng(one_seed)))
        a = self.max_cost - self.min_cost
        b = self.min_cost
        for algo in results:
            for T in results[algo]:
                results[algo][T] = [[v1 * a + b + erm_mean * self.shortage_penalty, v2 * a + b + erm_mean * self.shortage_penalty] for v1, v2 in results[algo][T]]

        return results

    def demand_func(self, size, distribution, maximum_demand, one_seed, min_val = 0, **kwargs):

        def randomize_demands(demands: np.ndarray, x: float) -> np.ndarray:
            n = len(demands)
            randomized_demands = demands.copy()
            mask = np.random.rand(n) < x
            randomized_demands[mask] = 0
            return randomized_demands

        max_val = maximum_demand

        def secure_rng():
            pid = os.getpid()
            t = int(time.time() * 1e9)
            u = uuid.uuid4().int % (2 ** 63)
            mix = (pid << 48) ^ (t << 16) ^ u
            ss = np.random.SeedSequence(mix)
            return np.random.default_rng(ss)

        rng = one_seed
        if distribution == "uniform":
            inital_demand_sequence = rng.uniform(min_val, max_val, size)

        elif distribution == "normal":
            mean = kwargs.get("mean", (min_val + max_val) / 2)
            std_dev = kwargs.get("std_dev", (max_val - min_val) / 6)  # Approximate rule for range
            samples = rng.normal(mean, std_dev, size)
            inital_demand_sequence = np.clip(samples, min_val, max_val)  # Ensuring values stay in range


        elif distribution == "exponential":
            scale = kwargs.get("scale", (max_val - min_val) / 3)
            samples = rng.exponential(scale, size) + min_val
            inital_demand_sequence = np.clip(samples, min_val, max_val)

        elif distribution == "beta":
            alpha = kwargs.get("alpha", 2)
            beta = kwargs.get("beta", 5)
            samples = rng.beta(alpha, beta, size)
            inital_demand_sequence = min_val + samples * (max_val - min_val)  # Scale to range

        else:
            raise ValueError("Unsupported distribution. Choose from: uniform, normal, exponential, beta.")

        return randomize_demands(inital_demand_sequence, self.demand_zero_rate)

    def policy_playing(self, T, alpha_r, demands, if_initial_state, x1 = None, x2 = None, x3 = None, boundary_checking = None):
        L = self.L 
        l = self.l  
        c1 = self.purchasing_cost  
        c2 = self.purchasing_cost_expedit
        h = self.holding_cost
        p = self.shortage_penalty 
        s = 0  
        gamma = 1  
        X_t = 0  # Initial inventory
        orders_s = np.zeros(T+1)  # Short-lead orders arriving per period
        orders_r = np.zeros(T+1)  # Long-lead orders arriving per period
        inventory_levels = [0]
        state_list = [0]
        total_cost = []

        if if_initial_state:
            X_t = x1 
            state_list[0] = x1
            if x2 is not None:
                orders_s[1:len(x2)+1] = x2
            if x3 is not None:
                orders_r[1:len(x3)+1] = x3

        
        TIP_r = np.zeros(T+1)  # Long-lead TIPs
        TIP_s = np.zeros(T+1)  # Short-lead TIPs
        
        # Compute TIPs (assuming each channel is ignored while computing the other)
        if isinstance(alpha_r, (int, float, np.integer)):
            TIP_r[1:T+1] = alpha_r  # Approximate demand smoothing
        else:
            if len(alpha_r) == 1:
                TIP_r[1:T + 1] = alpha_r
            elif len(alpha_r) == 2:
                TIP_s[1:T+1] = alpha_r[0]
                TIP_r[1:T+1] = alpha_r[1]
        
        # Main loop
        for t in range(1, T+1):
            # Place orders
            cost_t = 0 
            if t + l <= T:
                orders_s[t + l] = max(TIP_s[t] - (X_t + sum(orders_s[t:t+l + 1]) + sum(orders_r[t:t+l+1])), 0)
                cost_t += orders_s[t + l] * c2
            if t + L <= T:
                orders_r[t + L] = max(TIP_r[t] - (X_t + sum(orders_s[t:t+l + 1]) + sum(orders_r[t:t+L+1])), 0)
                cost_t += orders_r[t + L] * c1
            
            # Realize demand
            D_t = demands[t-1]
            
            # Compute cost
            cost_t = h * max(X_t + orders_s[t] + orders_r[t], 0) \
                     -(h+p)*min(X_t + orders_s[t] + orders_r[t], D_t)

            total_cost.append((gamma ** (t-1)) * self._cost_normalize(cost_t, boundary_checking = boundary_checking))
            
            # Update inventory level
            inventory_levels.append(X_t + orders_s[t] + orders_r[t])
            X_t = max(X_t + orders_s[t] + orders_r[t] - D_t, 0)
            state_list.append(X_t)

        return total_cost, inventory_levels, orders_s, orders_r, state_list

