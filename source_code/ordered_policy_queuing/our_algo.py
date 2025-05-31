import numpy as np
import itertools
from tqdm import tqdm
import baselines
import my_tools
import queuing_model



class InformationOrderedPolicyElimination:
    def __init__(self, param_, dynamics_, alpha_, seed_):
        self.env = None
        self.policy_set = dynamics_.policy_set  # Set of parameterized policies
        self.r = np.round(1/np.sqrt(dynamics_.time_horizon), 2)
        self.T = dynamics_.time_horizon
        self.alpha_ = alpha_
        self.current_policy_set = None
        self.beta_k_coeff = 1

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_
        self.one_seed = None

        self.rng_factory = None

    def discretization(self):
        bounds = self.policy_set  
        r = self.r  
        grid_axes = []
    
        for (a, b) in bounds:
            # Ensure inclusion of upper bound by a small epsilon adjustment
            #axis_points = np.arange(a, b + r / 2, r)
            num_points = int(min(50, np.ceil((b - a) / r)))
            axis_points = np.linspace(a, b, num_points)
            grid_axes.append(axis_points)
    
        # Create Cartesian product of all axis points
        grid_points = list(itertools.product(*grid_axes))
    
        # Convert to numpy arrays
        net = [np.array(point) for point in grid_points]
        
        return net


    def _return_to_start(self, x1, x2, x3):
        if self.param_.current_problem == 'inventory_control' or 'dual_index':
            temp_T = 1000
            demands = self.dynamics_.demand_func(size=int(temp_T), distribution=self.param_.hyper_set['distribution'],
                                                 maximum_demand=self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
            total_cost, _, order_s, orders_r, crucial_state = self.dynamics_.policy_playing(int(temp_T), [0,0], demands, if_initial_state = 1, x1 = x1, x2 = x2, x3 = x3)
            def _find_return_index(a1, a2, a3):
                for i in range(len(a3) - 1, -1, -1):
                    if max(a1[i],a2[i], a3[i]) > 0:
                        return i
                print('Line 51, Our_algo, return_time is not enough')
                return 0
            return_index = _find_return_index(crucial_state, order_s, orders_r)
            return np.sum(total_cost[:return_index]), return_index

    def _info_maximum_policy(self):

        def _select_row(arr):
            max_col2 = np.max(arr[:, 1])
            rows_with_max_col2 = arr[arr[:, 1] == max_col2]
        
            max_col1 = np.max(rows_with_max_col2[:, 0])
            rows_with_max_both = rows_with_max_col2[rows_with_max_col2[:, 0] == max_col1]
        
            return rows_with_max_both[0]
        if self.param_.current_problem == 'inventory_control':
            return np.max(np.array(self.current_policy_set))
        if self.param_.current_problem == 'dual_index':
            return _select_row(np.array(self.current_policy_set))

    def _counterfactual_estimate(self, demands, inventory_level, beta_k, current_policy):
        if self.param_.current_problem == 'inventory_control':
            x_list =  self.current_policy_set
            f_values = [np.sum(self.dynamics_.policy_playing(len(demands), x, demands, if_initial_state = 0)[0][: -self.param_.hyper_set['L']]) for x in self.current_policy_set]
            min_f = min(f_values)
            selected_x = [x for x, fx in zip(x_list, f_values) if fx < min_f + 2 * beta_k]
        if self.param_.current_problem == 'dual_index':
            x_list =  self.current_policy_set
            demands_part_1 = demands[: -self.param_.hyper_set['L']]
            demands_part_2 = demands[-self.param_.hyper_set['L']:]
            inventory_level_measurable = inventory_level[1: -self.param_.hyper_set['L']]
            counterfactual_demands = demands_part_1[np.array(inventory_level_measurable) == current_policy[1]]
            if len(counterfactual_demands) == 0:
                return x_list
            else:
                counterfactual_demands_final = np.concatenate((counterfactual_demands, demands_part_2))
                f_values = [np.sum(self.dynamics_.policy_playing(len(counterfactual_demands_final), x, counterfactual_demands_final, if_initial_state = 0)[0][: -self.param_.hyper_set['L']]) for x in self.current_policy_set]
                min_f = min(f_values)
                selected_x = [x for x, fx in zip(x_list, f_values) if fx < min_f + 2 * beta_k]

        return selected_x

    def _MM1_counterfactual_estimate(self, state_sequence, beta_k, current_policy, N_k):

        def process_queue(states, L, times=None):

            N = len(states)
            if times is None:
                times = list(range(N))
            interarrival = []
            last_arrival_time = None
            last_leave_time = None
            service_times = {j: [] for j in range(L+1)}
            for i in range(1, N):
                dt = states[i] - states[i-1]
                t  = times[i]
                if dt == 1 and states[i] <= L:
                    if last_arrival_time is not None:
                        interarrival.append(t - last_arrival_time)
                    last_arrival_time = t
                    last_leave_time = None
                if dt == -1:
                    prev_state = states[i-1]
                    if last_leave_time is not None:
                        service_times[prev_state].append(t - last_leave_time)
                    last_leave_time = t
            service_times_by_state = [service_times[j] for j in range(L+1)]
            return interarrival, service_times_by_state, times

        def estimate_mu_at_full(states, L):
            T = sum(1 for s in states[:-1] if s == L)
        
            D = sum(
                1
                for i in range(len(states)-1)
                if states[i] == L and states[i+1] == L-1
            )
        
            if T == 0:
                return np.nan
        
            rate_full = D / T
            return rate_full


        x_list = self.current_policy_set
        f_values = []
        hat_lambda, _, _ = process_queue(state_sequence, self.param_.hyper_set['S'])
        hat_mu = estimate_mu_at_full(state_sequence, self.param_.hyper_set['S'])

        
        if not hat_lambda or np.isnan(hat_mu):
            return x_list

        hat_lambda = 1 / np.mean(hat_lambda) * self.dynamics_.U
        hat_mu = (hat_mu * self.dynamics_.U - current_policy[-1]) / self.param_.hyper_set['S']

        if hat_lambda <= 0 or hat_mu <= 0 or hat_lambda > self.param_.hyper_set['lambda_max'] or hat_mu > self.param_.hyper_set['mu_max']:
            return x_list

        counterfactual_setting = queuing_model.DVFSMDP(
                    self.param_.hyper_set['S'],
                    self.param_.hyper_set['Amax'],
                    hat_lambda,
                    self.param_.hyper_set['lambda_max'],
                    hat_mu,
                    self.param_.hyper_set['mu_max'],
                    self.param_.hyper_set['C'])
        for current_policy in x_list:
            state_ = 0 
            temp_t = 0
            counter_cost = 0
            while temp_t < N_k:
                s_next, reward_ = counterfactual_setting.sample(state_, current_policy[state_], self.one_seed)
                counter_cost += reward_
                temp_t += 1
                state_ = s_next
            f_values.append(counter_cost)
        min_f = min(f_values)
        selected_x = [x for x, fx in zip(x_list, f_values) if fx < min_f + 2 * beta_k]
        return selected_x
        





    def run(self):
        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()
        self.evaluation = baselines.PolicyEvaluation(self.param_, self.dynamics_, one_seed = self.one_seed)

        if self.param_.current_problem == 'inventory_control':
            self.current_policy_set = self.discretization()
            current_t = 1 
            current_k = 1
            current_cost = 0 
            with tqdm(total=self.T, desc="Running", unit="step") as pbar:
                while current_t < self.T:
                    N_k = 4 ** current_k * np.round((1/self.alpha_), 0) + self.param_.hyper_set['L']
                    beta_k = self.beta_k_coeff * (np.sqrt(1/(self.alpha_ * N_k)))
                    demands = self.dynamics_.demand_func(size = int(N_k), distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                    current_policy = self._info_maximum_policy()
                    #current_policy = self._info_maximum_policy()
                    temp_cost, inventory_level, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(int(N_k), current_policy, demands, if_initial_state = 0)
                    current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                    self.current_policy_set = self._counterfactual_estimate(demands, inventory_level, beta_k, current_policy)
        
                    current_t += 4 ** current_k * np.round((1/self.alpha_), 0)
                    current_k += 1
                    a0, a1 = self._return_to_start(crucial_state[1], None, pipeline_state[-self.param_.hyper_set['L']:])
                    current_cost += a0 
                    current_t += a1
                    pbar.update(1)
            return current_cost/(current_t + 1e-6), self.evaluation.run(current_policy)

        if self.param_.current_problem == 'dual_index':
            self.current_policy_set = self.discretization()
            current_t = 1
            current_k = 1
            current_cost = 0
            with tqdm(total=self.T, desc="Running", unit="step") as pbar:
                while current_t < self.T:
                    N_k = 4 ** current_k * np.round((1/self.alpha_), 0) + self.param_.hyper_set['L']
                    beta_k = self.beta_k_coeff * (np.sqrt(1/(self.alpha_ * N_k)))
                    demands = self.dynamics_.demand_func(size = int(N_k), distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                    current_policy = [v for v in self._info_maximum_policy()]
                    #current_policy = self._info_maximum_policy()
                    temp_cost, inventory_level, shortline_state, pipeline_state, crucial_state = self.dynamics_.policy_playing(int(N_k), current_policy, demands, if_initial_state = 0)
                    current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                    self.current_policy_set = self._counterfactual_estimate(demands, inventory_level, beta_k, current_policy)
    
                    current_t += 4 ** current_k * np.round((1/self.alpha_), 0)
                    current_k += 1
                    a0, a1 = self._return_to_start(crucial_state[1], shortline_state[2:2 + self.param_.hyper_set['l']], pipeline_state[-self.param_.hyper_set['L']:])
                    current_cost += a0
                    current_t += a1
                    pbar.update(1)
            return current_cost/(current_t + 1e-6), self.evaluation.run(current_policy)

        if self.param_.current_problem == 'M1L':
            current_t = 1 
            current_k = 1
            current_cost = 0 
            self.current_policy_set = self.dynamics_.policy_set
            state_ = 0
            with tqdm(total=self.T, desc="Running", unit="step") as pbar:
                while current_t < self.T:
                    N_k = 4 ** current_k * np.round((1/self.alpha_), 0)
                    beta_k = self.beta_k_coeff * (np.sqrt(1/(self.alpha_ * N_k))) * 10000
                    current_policy = self.current_policy_set[0]
                    state_sequence = []
                    temp_t = 0
                    while temp_t < N_k:
                        s_next, reward_ = self.dynamics_.sample(state_, current_policy[state_], self.one_seed)
                        current_cost += reward_
                        current_t += 1
                        temp_t += 1
                        state_ = s_next
                        state_sequence.append(state_)
                    self.current_policy_set = self._MM1_counterfactual_estimate(state_sequence, beta_k, current_policy, N_k)
                    if len(self.current_policy_set) == 1:
                        break

                    current_k += 1
                    current_t += N_k
                    pbar.update(N_k)
            return current_cost/(current_t + 1e-6), self.evaluation.run(current_policy)
            

    def close(self):
        del self.large_array







