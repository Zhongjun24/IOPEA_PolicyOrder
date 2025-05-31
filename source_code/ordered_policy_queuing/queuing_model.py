import numpy as np
from itertools import product



class DVFSMDP:
    """
    MDP model for a DVFS-controlled birth-death process with transition-dependent rewards.

    States: S = {0, ..., S-1} (number of jobs in system)
    Actions: A = {0, ..., Amax} (processing speed)

    After uniformization, transitions at state i under action a:
        p_arrival = lambda_i(i) / U (i -> i+1 if i < S-1)
        p_miss    = i * mu_rate / U       (deadline miss, i -> i-1)
        p_service = a / U                 (service completion, i -> i-1)
        p_stay    = 1 - p_arrival - p_miss - p_service

    Rewards depend on the transition:
        - Arrival and stay transitions: cost = w(a)/U, reward = rmax - cost
        - Service transition:        cost = w(a)/U, reward = rmax - cost
        - Miss transition:           cost = w(a)/U + C, reward = rmax - cost
    """
    def __init__(
        self,
        S,
        Amax,
        lambda_rate,
        lambda_max,
        mu_rate,
        mu_max,
        C,
        time_horizon = None,
    ):
        self.S = S
        self.Amax = Amax
        self.lambda_rate = lambda_rate
        self.lambda_max = lambda_max
        self.mu_rate = mu_rate
        self.mu_max = mu_max
        self.C = C
        self.time_horizon = time_horizon
        # uniformization constant
        self.U = lambda_max + (S-1)*mu_max + Amax
        # max reward to ensure positivity
        self.rmax = C + Amax**2/mu_rate

        self.policy_set = self._generate_all_policies()

    def lambda_i(self, i):
        """Decaying arrival rate at state i."""
        return self.lambda_rate

    def transition_outcomes(self, state, action, arg1 = None):
        """
        Returns list of (next_state, prob, reward) tuples for each possible transition.
        """
        i, a = state, action
        # transition probabilities
        p_arrival = self.lambda_i(i) / self.U if i < self.S else 0.0
        p_miss    = i * self.mu_rate / self.U if i > 0 else 0.0
        p_service = a / self.U
        p_stay    = 1.0 - (p_arrival + p_miss + p_service)

        if p_stay < 0:
            print('invalid args in estimation of mu and lambda')

        # common cost components
        cost_hit  = a**2/self.U
        cost_miss = cost_hit + self.C

        # prepare outcomes
        outcomes = []
        # arrival
        if p_arrival > 0:
            reward = cost_hit
            outcomes.append((i+1, p_arrival, reward))
        # miss
        if p_miss > 0:
            reward = cost_miss
            outcomes.append((i-1, p_miss, reward))
        # service
        if p_service > 0:
            reward = cost_hit
            # if i==0, service leads to stay
            next_s = i-1 if i>0 else i
            outcomes.append((next_s, p_service, reward))
        # stay
        if p_stay > 0:
            reward = cost_hit
            outcomes.append((i, p_stay, reward))

        return outcomes

    def expected_reward(self, state, action):
        """Compute expected reward summing over transition outcomes."""
        return sum(p * r for _, p, r in self.transition_outcomes(state, action))

    def one_step_transition(self, state, action):
        """Return dict of next_state -> probability (marginalized over reward)."""
        probs = {}
        for s_next, p, _ in self.transition_outcomes(state, action):
            probs[s_next] = probs.get(s_next, 0) + p
        return probs

    def sample(self, state, action, one_seed, arg1 = None):
        """
        Sample one transition: returns (next_state, reward).
        """
        outcomes = self.transition_outcomes(state, action, arg1)
        next_states, probs, rewards = zip(*outcomes)
        idx = one_seed.choice(len(outcomes), p=probs)
        return next_states[idx], rewards[idx]

    def _generate_all_policies(self):
        """
        Generate all deterministic policies for the MDP.
        Each policy is represented as a tuple of length S:
            policy[s] = action at state s.

        Returns:
            List[Tuple[int, ...]]  of length (Amax+1)^S.
        """
        S = self.S
        Amax = self.Amax
        action_space = range(Amax + 1)

        all_policies = list(product(action_space, repeat=S+1))
        return all_policies
