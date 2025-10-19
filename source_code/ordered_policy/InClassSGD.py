import numpy as np
import my_tools
import baselines

class FiniteDifferenceGD:
    """
    Finite-difference SGD on a single real trajectory.
    Each gradient evaluation consumes 2*k time steps in order:
      - roll k steps with (theta + delta), updating the real carry_state
      - roll k steps with (theta - delta), updating the real carry_state again
    Then update theta and continue, until total consumed steps reach T.
    """

    def __init__(self, param_, dynamics_, seed_,
                 k: int = 100,
                 lr0: float = 100,
                 delta: float = 0.9,
                 step_rule: str = 'constant',
                 project: bool = True):
        self.policy_set = dynamics_.policy_set
        self.T = dynamics_.time_horizon

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.seed_ = seed_

        self.k = int(k)
        self.lr0 = lr0
        self.delta = delta
        assert step_rule in ('constant', 'sqrt')
        self.step_rule = step_rule
        self.project = project

        self.one_seed = None

    # ---- projection / ordering ----
    def _project_policy(self, theta):
        if self.param_.current_problem == 'inventory_control':
            lo, hi = self.policy_set[0]
            return float(np.clip(float(theta), lo, hi))
        else:
            (lo_s, hi_s), (lo_r, hi_r) = self.policy_set
            s, r = float(theta[0]), float(theta[1])
            s = float(np.clip(s, lo_s, hi_s))
            r = float(np.clip(r, lo_r, hi_r))
            if s > r:
                s, r = r, s
            s = float(np.clip(s, lo_s, hi_s))
            r = float(np.clip(r, lo_r, hi_r))
            return [s, r]

    # ---- run exactly m steps on the real trajectory (starting at index t0) ----
    def _roll_m_steps(self, policy, demand_stream, t0, carry_state, m):
        """
        Calls policy_playing once with T_call = m + L; sums first m costs.
        Returns (cost_sum, next_carry_state, t0_after).
        """
        L = self.param_.hyper_set['L']
        l = self.param_.hyper_set.get('l', 0)

        T_call = int(m + L)
        # Use the exact segment with L-step lookahead
        demands_seg = demand_stream[t0 : t0 + T_call]

        if carry_state is None:
            res = self.dynamics_.policy_playing(T_call, policy, demands_seg, if_initial_state=0)
        else:
            if self.param_.current_problem == 'inventory_control':
                res = self.dynamics_.policy_playing(
                    T_call, policy, demands_seg, if_initial_state=1,
                    x1=carry_state["x1"], x2=None, x3=carry_state["x3"], boundary_checking=0
                )
            else:
                res = self.dynamics_.policy_playing(
                    T_call, policy, demands_seg, if_initial_state=1,
                    x1=carry_state["x1"], x2=carry_state["x2"], x3=carry_state["x3"], boundary_checking=0
                )

        # Unpack consistently: (total_cost, inventory_levels, orders_s, orders_r, state_list)
        temp_cost, _, orders_s, orders_r, state_list = res

        # Sum only the first m costs
        cost_sum = float(np.sum(temp_cost[:m]))

        # Build next carry-over state at time t0 + m
        next_state = {"x1": state_list[m]}
        # orders at arrivals within next L periods after advancing m steps
        next_state["x3"] = orders_r[-L:]
        if self.param_.current_problem == 'dual_index':
            next_state["x2"] = orders_s[m + 1 : m + 1 + l]

        return cost_sum, next_state, t0 + m

    # ---- main loop over time ----
    def run(self):
       rng_factory = my_tools.make_rng_factory(self.seed_)
       self.one_seed = rng_factory()
       evaluator = baselines.PolicyEvaluation(self.param_, self.dynamics_, one_seed=self.one_seed)

       # initialize theta at interval midpoints
       if self.param_.current_problem == 'inventory_control':
           lo, hi = self.policy_set[0]
           theta = 0.5 * (lo + hi)
       elif self.param_.current_problem == 'dual_index':
           (lo_s, hi_s), (lo_r, hi_r) = self.policy_set
           theta = [0.5 * (lo_s + hi_s), 0.5 * (lo_r + hi_r)]
           theta = self._project_policy(theta)
       else:
           raise ValueError("Unsupported problem type")

       # pre-sample a demand stream long enough to cover the final update without special tail logic
       L = self.param_.hyper_set['L']
       dist = self.param_.hyper_set['distribution']
       max_d = self.param_.hyper_set['maximum_demand']
       extra = 4 * self.k  # worst-case extra consumption for dual_index (4 segments)
       demand_stream = self.dynamics_.demand_func(
           size=int(self.T + L + extra),
           distribution=dist,
           maximum_demand=max_d,
           one_seed=self.one_seed
       )

       # single real-trajectory pointers/state
       t0 = 0
       carry_state = None
       total_cost = 0.0
       update_idx = 0

       while t0 < self.T:
           update_idx += 1
           lr_t = self.lr0 if self.step_rule == 'constant' else (self.lr0 / max(1.0, np.sqrt(update_idx)))

           if self.param_.current_problem == 'inventory_control':
               # +delta
               theta_plus = self._project_policy(theta + self.delta)
               cost_plus, carry_state, t0 = self._roll_m_steps(theta_plus, demand_stream, t0, carry_state, self.k)

               # -delta
               theta_minus = self._project_policy(theta - self.delta)
               cost_minus, carry_state, t0 = self._roll_m_steps(theta_minus, demand_stream, t0, carry_state, self.k)

               # gradient and update
               f_plus = cost_plus / self.k
               f_minus = cost_minus / self.k
               grad = (f_plus - f_minus) / (2.0 * self.delta)
               theta = self._project_policy(theta - lr_t * float(grad))

               total_cost += (cost_plus + cost_minus)

           else:
               # dual_index: coordinate-wise central differences (4 segments)
               theta_np = np.array(theta, dtype=float)

               # s + delta
               theta_s_plus = self._project_policy((theta_np + np.array([self.delta, 0.0])).tolist())
               cost_s_plus, carry_state, t0 = self._roll_m_steps(theta_s_plus, demand_stream, t0, carry_state, self.k)

               # s - delta
               theta_s_minus = self._project_policy((theta_np - np.array([self.delta, 0.0])).tolist())
               cost_s_minus, carry_state, t0 = self._roll_m_steps(theta_s_minus, demand_stream, t0, carry_state, self.k)

               f_s_plus = cost_s_plus / self.k
               f_s_minus = cost_s_minus / self.k
               grad_s = (f_s_plus - f_s_minus) / (2.0 * self.delta)

               # r + delta
               theta_r_plus = self._project_policy((theta_np + np.array([0.0, self.delta])).tolist())
               cost_r_plus, carry_state, t0 = self._roll_m_steps(theta_r_plus, demand_stream, t0, carry_state, self.k)

               # r - delta
               theta_r_minus = self._project_policy((theta_np - np.array([0.0, self.delta])).tolist())
               cost_r_minus, carry_state, t0 = self._roll_m_steps(theta_r_minus, demand_stream, t0, carry_state, self.k)

               f_r_plus = cost_r_plus / self.k
               f_r_minus = cost_r_minus / self.k
               grad_r = (f_r_plus - f_r_minus) / (2.0 * self.delta)

               # vector update
               grad_vec = np.array([grad_s, grad_r], dtype=float)
               theta = (theta_np - lr_t * grad_vec).tolist()
               theta = self._project_policy(theta)

               total_cost += (cost_s_plus + cost_s_minus + cost_r_plus + cost_r_minus)

       final_avg_cost = total_cost / self.T
       return final_avg_cost, evaluator.run(theta)

    def close(self):
        pass