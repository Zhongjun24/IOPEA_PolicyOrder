import numpy as np
import sys
from tqdm import tqdm
import my_tools
import baselines


DEBUG = False # parameter for printing out parameters during code execution

class ConvexityAlgo():
    """
    Algorithm for learning optimal base stock policy online using an online convex optimization scheme

    References:
        Agarwal et al. 2020
        Agarwal et al. 2011
    
    Attributes:
        base_stock: the case stock level
    """

    #def __init__(self, num_episodes, epLen, max_base_stock_value, conf_cons = 1/4, INT_FLAG=True):
    def __init__(self, param_, dynamics_, H, seed_, INT_FLAG = 0):
        '''
        Arguments:
            num_episode : number of episodes k
            epLen : number of steps H
            max_base_stock_value : maximum value for the base stock
            conf_cons : constant in the confidence interval calculations
            INT_FLAG: flag to restrict to integral base stock values
        '''
        # Meta parameters for the algorithm

        self.param_ = param_
        self.dynamics_ = dynamics_
        self.T = dynamics_.time_horizon
        self.seed_ = seed_
        self.one_seed = None
        self.rng_factory = None



        self.num_episodes = int(dynamics_.time_horizon/H) + 1
        self.epLen = H
        self.max_base_stock_value = dynamics_.policy_set[0][1]
        self.conf_cons = 1/50
        self.initial_state = [param_.hyper_set['policy_set'][0][0]] * (param_.hyper_set['L'] + 1)

        self.current_episode = 0
        self.INT_FLAG = INT_FLAG # flag for only considering integral policies

            # supposed to be the sub-Gaussian parameter on the noise
            # but our estimates for V_\tau are in [0,1] so we use (b-a)^2 / 4 = 1/4 by default

        # Current algorithm stage
        self.current_epoch = 0 # current epoch
        self.epoch_step = 0 # timestep within the epoch
        
        self.update_base_stock_estimates(0, self.max_base_stock_value) # calculates the current base stock values to evaluate


        self.index = 0 # index of low, center, high for execution



    def reset(self): # rests back so no information is carried over to next iteration

        self.current_episode = 0

        # Current algorithm stage
        self.current_epoch = 0 # current epoch
        self.epoch_step = 0 # timestep within the epoch
        
        self.update_base_stock_estimates(0, self.max_base_stock_value) # calculates the 1/4, 2/4, 3/4 width of the current
                # working interval

        self.index = 0 # index of low, center, high for execution


    def update_base_stock_estimates(self, low, high):
        """
        Helper function. Takes as input the low value and a width and updates:
            - the three base stock values (i.e. the mid low, mid center, and mid hight)
            - the three base stock policy estimates
        """

        self.low = low
        self.high = high
        self.width = high - low

        if not self.INT_FLAG: # allowing algorithm to check non integral base stock values
            self.mid_low = np.floor(self.low + self.width / 4) # gets out the 1/4, 2/4, 3/4 of the interval
            self.mid_center = np.floor(self.low + self.width / 2)
            self.mid_high = np.ceil(self.low + (3 * self.width / 4))
            
            self.base_values = [self.mid_low, self.mid_center, self.mid_high] # list of base stock values we are evaluating
            self.base_estimates = [0., 0., 0.] # and their estimates
            

        if self.INT_FLAG and high-low > 2: # if we are only checking integral base stock, need to round them all
            self.mid_low = np.floor(self.low + self.width / 4) # gets out the 1/4, 2/4, 3/4 of the interval
            self.mid_center = np.floor(self.low + self.width / 2)
            self.mid_high = np.ceil(self.low + (3 * self.width / 4))

            self.base_values = [self.mid_low, self.mid_center, self.mid_high] # list of base stock values and their estimates
            self.base_estimates = [0., 0., 0.]

        elif self.INT_FLAG and high - low == 2: # however once the window is with three values, just evaluate each one individually
            
            self.base_values = [self.low, self.low+1, self.high]
            self.base_estimates = [0., 0., 0.]

        elif self.INT_FLAG and high - low == 1: # down to two base stock values
            self.base_values = [self.low, self.high]
            self.base_estimates = [0., 0.]

        elif self.INT_FLAG and high - low == 0: # narrowed on the optimal base stock value
            self.base_values = [self.low]
            self.base_estimates = [0.]

        if DEBUG: print(f'Updating the interval! Current interval: {self.low, self.high}')
        if DEBUG: print(f'Current base stock values: {self.base_values}')


    def update_info(self, cost):
        """
            Function that takes as input observed counts and adds that to the estimate of the
            base stock level
        
        """

        # Adding on to the observed costs based on which index / which base stock policy is used
        self.base_estimates[self.index] += cost


    def pick_action(self, state, timestep, episode):
        """
        Picks an action according to the current base stock level.

        Args:
            state: current inventory orders in the system

        Returns:
            action: an ordering amount
        
        """

        total_inventory = np.sum(state) # calculates the total inventory in the pipeline

        if total_inventory <= self.base_values[self.index]: # if the total inventory is smaller than the base stock level
            return (self.base_values[self.index] - total_inventory) # order the difference

        else:
            return 0 # otherwise order nothing


    def update_param(self):
        
        self.current_episode += 1

        self.epoch_step += 1 # end of an episode, so add one to the counter for number of episodes used
            # to estimate that base stock policy

        if self.epoch_step >= (2 * self.conf_cons * np.log(self.num_episodes) * (4 ** (self.current_epoch + 1))): # collected enough data from this policy
            if DEBUG: print(f'Moving to next base stock value')
            num_datapoints = self.epoch_step * self.epLen # number of datapoints in cost estimate

            self.index += 1 # update index to evaluate next base stock policy
            self.epoch_step = 0

        if self.index >= len(self.base_values): # evaluated all of the current base stock policies

            if DEBUG: print(f'Finished evaluating base stock policies: {self.base_values}')
            self.current_epoch += 1 # adds one to counter of number of epochs

            mean_estimates = np.asarray(self.base_estimates) / num_datapoints # calculate average cost estimates
            if DEBUG: print(f'Cost estimates: {mean_estimates}')

            conf_interval = 2 ** ((-1) * (self.current_epoch + 1)) # calculates confidence width

            low_estimates = mean_estimates - (conf_interval / 2) # gets lower and upper confidence bound estimates on base stock performance
            high_estimates = mean_estimates + (conf_interval / 2)

            if DEBUG: print(f'Conf Intervals: {low_estimates, mean_estimates, high_estimates}')

            self.index = 0 # resets index to zero
            self.epoch_step = 0 # step within the epoch to zero

            if not self.INT_FLAG: # If we are not checking only integral base stock values
                # Taken from Agrawal's paper on updating the confidence intervals
                if max(low_estimates[0], low_estimates[2]) >= min(high_estimates[0], high_estimates[2]) + conf_interval:
                    if low_estimates[0] >= low_estimates[2]:
                        self.update_base_stock_estimates(self.base_values[0], self.high)
                    else:
                        self.update_base_stock_estimates(self.low, self.base_values[2])

                elif max(low_estimates[0], low_estimates[2]) >= high_estimates[1] + conf_interval:
                    if low_estimates[0] >= low_estimates[2]:
                        self.update_base_stock_estimates(self.base_values[0], self.high)
                    else:
                        self.update_base_stock_estimates(self.low, self.base_values[2])

                else:
                    self.base_estimates = [0., 0., 0.]
                    if DEBUG: print(f'Not updating the working interval!')


            elif self.INT_FLAG and self.high - self.low > 2: # If we are not checking only integral base stock values
                # Taken from Agrawal's paper on updating the confidence intervals
                if max(low_estimates[0], low_estimates[2]) >= min(high_estimates[0], high_estimates[2]) + conf_interval:
                    if low_estimates[0] >= low_estimates[2]:
                        self.update_base_stock_estimates(self.base_values[0]+1, self.high)
                    else:
                        self.update_base_stock_estimates(self.low, self.base_values[2]-1)

                elif max(low_estimates[0], low_estimates[2]) >= high_estimates[1] + conf_interval:
                    if low_estimates[0] >= low_estimates[2]:
                        self.update_base_stock_estimates(self.base_values[0]+1, self.high)
                    else:
                        self.update_base_stock_estimates(self.low, self.base_values[2]-1)

                else:
                    self.base_estimates = [0., 0., 0.]
                    if DEBUG: print(f'Not updating the working interval!')


            elif self.INT_FLAG and self.high - self.low == 2: # down to three base stock values
                if low_estimates[0] >= max(high_estimates[1], high_estimates[2]): # narrow down if one is provably worse
                    self.update_base_stock_estimates(self.low+1, self.high)
                elif low_estimates[2] >= max(high_estimates[0], high_estimates[1]):
                    self.update_base_stock_estimates(self.low, self.low+1)
                else:
                    self.base_estimates = [0., 0., 0.]
                    if DEBUG: print(f'Not updating the working interval!')

            elif self.INT_FLAG and self.high - self.low == 1: # down to just two base stock values
                if high_estimates[0] <= low_estimates[1]: # first policy is provably better
                    self.update_base_stock_estimates(self.low, self.low) # converge on the first one
                elif high_estimates[1] <= low_estimates[0]:
                    self.update_base_stock_estimates(self.high, self.high) # converge on the second one
                else:
                    self.base_estimates = [0., 0.] # keep going, not converged yet to within confidence estimates
                    if DEBUG: print(f'Not updating the working interval!')

            elif self.INT_FLAG and self.high - self.low == 0: # just continue, since already picked final base stock value
                self.base_estimates = [0.]


        
    def update_Qend(self, k):
        return

    def run(self):
        self.rng_factory = my_tools.make_rng_factory(self.seed_)
        self.one_seed = self.rng_factory()
        self.evaluation = baselines.PolicyEvaluation(self.param_, self.dynamics_, one_seed = self.one_seed)
        current_t = 1
        current_cost = 0  
        state = self.initial_state
        with tqdm(total=self.T, desc="Running", unit="step") as pbar:
            for k in range(self.num_episodes):
                for step in range(self.epLen):
                    action = self.pick_action(state, step, k)
                    demands = self.dynamics_.demand_func(size = 1 + self.param_.hyper_set['L'], distribution = self.param_.hyper_set['distribution'], maximum_demand = self.param_.hyper_set['maximum_demand'], one_seed = self.one_seed)
                    current_policy = np.sum(state) + action  
                    temp_cost, _, _, pipeline_state, crucial_state = self.dynamics_.policy_playing(1 + self.param_.hyper_set['L'], current_policy, demands, if_initial_state = 1, x1= state[0], x2 = None, x3 = state[1:])
                    new_state = (crucial_state[1], *pipeline_state[-self.param_.hyper_set['L']:].tolist())
                    self.update_info(cost = np.sum(temp_cost[: -self.param_.hyper_set['L']]))
                    state = new_state
    
                    current_t += 1 
                    current_cost += np.sum(temp_cost[: -self.param_.hyper_set['L']])
                    pbar.update(1)
                self.update_param()

        return current_cost/(current_t + 1e-6), self.evaluate()

    def evaluate(self):
        """
        Evaluate the final learned base stock policy by rollout self.T steps.
        """
    
        eval_state = self.initial_state
        eval_total_reward = 0
        eval_steps = 0

        final_base_stock_value = self.pick_action(eval_state, 0, episode = None) + np.sum(eval_state)
        return self.evaluation.run(final_base_stock_value)













