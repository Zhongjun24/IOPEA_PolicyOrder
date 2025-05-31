import random
import numpy as np
import itertools
from tqdm import tqdm
import my_tools

class Random:
    def __init__(self, param_, dynamics_, seed_):
        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.T = dynamics_.time_horizon
        self.beta_k_coeff = 1

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.one_seed = None


    def run(self):
        #child_seed = np.random.SeedSequence(self.seed_)
        rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = rng_factory()
        self.evaluation = PolicyEvaluation(self.param_, self.dynamics_, one_seed = self.one_seed)
        if self.param_.current_problem == 'inventory_control':
            current_t = 1
            current_cost = 0
            while current_t < self.T:
                demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                current_policy = random.uniform(self.policy_set[0][0],self.policy_set[0][1])
                if current_t == 1:
                    temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 0)
                else:
                    temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= crucial_state[1], x2 = None, x3 = pipeline_state[-self.param_.hyper_set['L']:], boundary_checking = 0 )
                current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                current_t += 1


            return current_cost/self.T, self.evaluation.run(current_policy)

        if self.param_.current_problem == 'dual_index':
            current_t = 1 
            current_cost = 0
            while current_t < self.T:
                demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                current_policy = sorted([random.uniform(self.policy_set[0][0],self.policy_set[0][1]), random.uniform(self.policy_set[1][0],self.policy_set[1][1])])
                if current_t == 1:
                    temp_cost, _, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 0)
                else:
                    temp_cost, _, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= crucial_state[1], x2 = shortline_state[1:1 + self.param_.hyper_set['l']], x3 = pipeline_state[-self.param_.hyper_set['L']:], boundary_checking = 0)
                current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                current_t += 1
            return current_cost/self.T, self.evaluation.run(current_policy)

        if self.param_.current_problem == 'M1L':
            current_t = 1 
            current_cost = 0
            state_ = 0
            current_policy_set = self.dynamics_.policy_set
            while current_t < self.T:
                policy_index = self.one_seed.integers(low = 0, high = len(current_policy_set))
                current_policy = current_policy_set[policy_index]
                s_next, reward_ = self.dynamics_.sample(state_, current_policy[state_], self.one_seed)
                current_cost += reward_
                current_t += 1
                state_ = s_next
            return current_cost/self.T, self.evaluation.run(current_policy) 


    def close(self):
        del self.large_array


class Optimal:
    def __init__(self, param_, dynamics_, seed_):
        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.T = dynamics_.time_horizon
        self.beta_k_coeff = 1
        self.current_policy_set = None

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.one_seed = None

    def discretization(self):
        r = 1 if self.policy_set[-1][1] > 100 else 0.1
        bounds = self.policy_set  
        grid_axes = []
    
        for (a, b) in bounds:
            # Ensure inclusion of upper bound by a small epsilon adjustment
            axis_points = np.arange(a, b + r / 2, r)
            grid_axes.append(axis_points)
    
        # Create Cartesian product of all axis points
        grid_points = list(itertools.product(*grid_axes))
    
        # Convert to numpy arrays
        net = [np.array(point) for point in grid_points]
        
        return net



    def run(self, arg1 = None):
        rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = rng_factory()
        self.evaluation = PolicyEvaluation(self.param_, self.dynamics_, one_seed = self.one_seed)

        if self.param_.current_problem in ['inventory_control', 'dual_index']:
            self.current_policy_set = self.discretization()

            min_cost = np.inf
            best_policy = None
            for i, current_policy in enumerate(self.current_policy_set):
                current_t = 1
                current_cost = 0
                demands = self.dynamics_.demand_func(size = self.T + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(self.T + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 0)
                current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                current_t += 1
                if current_cost/self.T < min_cost:
                    min_cost = current_cost/self.T
                    best_policy = current_policy
                #print(i)
            return min_cost, self.evaluation.run(best_policy)

        if self.param_.current_problem == 'M1L':

            min_cost = np.inf
            best_policy = None
            current_policy_set = self.dynamics_.policy_set

            for i, current_policy in enumerate(current_policy_set):
                current_cost = self.evaluation.run(current_policy, arg1) 

                if current_cost < min_cost:
                    min_cost = current_cost
                    best_policy = current_policy
            return min_cost, self.evaluation.run(best_policy)


    def close(self):
        del self.large_array


class EmpiricalHindsight:
    def __init__(self, param_, dynamics_, seed_):
        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.T = dynamics_.time_horizon
        self.beta_k_coeff = 1
        self.current_policy_set = None
        self.policy_updating_frequency = int(self.T/10)

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.one_seed = None

    def discretization(self):
        r = 1 if self.policy_set[-1][1] > 100 else 0.1
        bounds = self.policy_set  
        grid_axes = []
    
        for (a, b) in bounds:
            # Ensure inclusion of upper bound by a small epsilon adjustment
            axis_points = np.arange(a, b + r / 2, r)
            grid_axes.append(axis_points)
    
        # Create Cartesian product of all axis points
        grid_points = list(itertools.product(*grid_axes))
    
        # Convert to numpy arrays
        net = [np.array(point) for point in grid_points]
        
        return net


    def run(self):
        rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = rng_factory()
        self.evaluation = PolicyEvaluation(self.param_, self.dynamics_, one_seed = self.one_seed)
        
        if self.param_.current_problem in ['inventory_control', self.param_.current_problem == 'dual_index']:
            self.current_policy_set = self.discretization()
    
            # Initialize
            current_policy = self.current_policy_set[0]  
            total_cost = 0
            current_t = 0
            k = self.policy_updating_frequency  
    
            all_demands = self.dynamics_.demand_func(
                size=self.T + self.param_.hyper_set['L'],
                distribution=self.param_.hyper_set['distribution'],
                maximum_demand=self.param_.hyper_set['maximum_demand'],
                one_seed = self.one_seed
            )
            with tqdm(total=self.T, desc="Running", unit="step") as pbar:
                while current_t < self.T:
                    remaining_steps = min(k, self.T - current_t)
        
                    temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(
                        remaining_steps + self.param_.hyper_set['L'], 
                        current_policy, 
                        all_demands[current_t:current_t + remaining_steps + self.param_.hyper_set['L']], 
                        if_initial_state=0
                    )
                    total_cost += np.sum(temp_cost[:remaining_steps])
        
                    # update time
                    current_t += remaining_steps
        
                    recent_demands = all_demands[:current_t + self.param_.hyper_set['L']]
        
                    min_cost = np.inf
                    best_policy = current_policy 
        
                    for candidate_policy in self.current_policy_set:
                        temp_cost, _, _, _, _ = self.dynamics_.policy_playing(
                            len(recent_demands), 
                            candidate_policy, 
                            recent_demands,
                            if_initial_state=0
                        )
                        avg_cost = np.sum(temp_cost[:len(recent_demands)]) / len(recent_demands)
                        if avg_cost < min_cost:
                            min_cost = avg_cost
                            best_policy = candidate_policy
        
                    current_policy = best_policy
                    pbar.update(remaining_steps)
        
            return total_cost / (self.T + 1e-6), self.evaluation.run(current_policy)

        if self.param_.current_problem == 'M1L':
            current_policy_set = self.dynamics_.policy_set
            total_cost = 0
            current_t = 0
            state_ = 0
            k = self.policy_updating_frequency 
            current_policy = current_policy_set[0] 
            while current_t < self.T:
                remaining_steps = min(k, self.T - current_t)
    
                t_temp = 0
                while t_temp <= remaining_steps:
                    s_next, reward_ = self.dynamics_.sample(state_, current_policy[state_], self.one_seed)
                    total_cost += reward_
                    t_temp += 1
                    state_ = s_next
    
                # update time
                current_t += remaining_steps
        
                min_cost = np.inf
                best_policy = current_policy 
    
                for candidate_policy in current_policy_set:
                    t_temp = 0
                    total_cost_temp = 0
                    state_ = 0
                    while t_temp <= remaining_steps:
                        s_next, reward_ = self.dynamics_.sample(state_, candidate_policy[state_], self.one_seed)
                        total_cost_temp += reward_
                        t_temp += 1
                        state_ = s_next
    
                    if total_cost_temp < min_cost:
                        min_cost = total_cost_temp
                        best_policy = candidate_policy
    
                current_policy = best_policy

            return total_cost / (self.T + 1e-6), self.evaluation.run(current_policy)


    def close(self):
        del self.large_array

class PolicyEvaluation:
    def __init__(self, param_, dynamics_, one_seed):
        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.T = param_.testing_horizon
        self.beta_k_coeff = 1
        self.current_policy_set = None

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.one_seed = one_seed


    def run(self, current_policy, arg1 = None):
        if self.param_.current_problem in ['inventory_control', 'dual_index']:
            demands = self.dynamics_.demand_func(size = self.T + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(self.T + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 0)
            current_cost = np.sum(temp_cost[: -self.param_.hyper_set['L']])
            return current_cost/self.T

        if self.param_.current_problem == 'M1L':
            current_t = 1 
            current_cost = 0
            state_ = 0
            while current_t < self.T:
                s_next, reward_ =self.dynamics_.sample(state_, current_policy[state_], self.one_seed, arg1 = arg1)
                current_cost += reward_
                current_t += 1
                state_ = s_next
            return current_cost/self.T



    def close(self):
        del self.large_array




