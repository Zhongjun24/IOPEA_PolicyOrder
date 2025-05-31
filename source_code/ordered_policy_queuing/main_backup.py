import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import multiprocessing

import hyperparameters
import our_algo
import ic
import baselines
import feedback_graph
import PPO
import online_convex
import gc
from tqdm import tqdm
import datetime

#notice that in our algo, we set the policy to be integer in line 79 to align with other algos especially the Q-learning ones.
#purchasing cost has to be 0
#the epsilon for optimal/empirical_hindsight are 0.1, fixed in the baselines file.



# Parameters
param_ = hyperparameters.HyperParameters()
time_horizon_list = param_.time_horizon_list
k = param_.exp_repeat_times
algo_names = param_.hyper_algo_set  

results = {algo: {T: [] for T in time_horizon_list} for algo in algo_names}

# Main experiment loop
print(f"\n=== {param_.current_problem} ===")
for T in time_horizon_list:
    print(f"\n=== Time Horizon: {T} ===")
    param_.hyper_set['time_horizon'] = T

    for run_idx in range(k):
        print(f"--- Run {run_idx+1}/{k} ---")

        dynamics_ = ic.InventoryControl(
            param_.hyper_set['policy_set'],
            param_.hyper_set['time_horizon'],
            param_.hyper_set['holding_cost'],
            param_.hyper_set['shortage_penalty'],
            param_.hyper_set['purchasing_cost'],
            param_.hyper_set['demand_zero_rate'],
            param_.hyper_set['L'],
            param_.hyper_set['l'],
            param_.hyper_set['maximum_demand']
        )
        if param_.current_problem == 'inventory_control':
            S, A, X = dynamics_.construct_state_action_space(
                b=param_.hyper_set['policy_set'][0][1],
                L=param_.hyper_set['L']+1,
                r=param_.hyper_set['discretization_radius_Qlearning']
            )
            alpha_partial_order = 1
        else:
            S, A, X = dynamics_.construct_state_action_space(
                b=param_.hyper_set['policy_set'][1][1],
                L=param_.hyper_set['L']+1,
                r=param_.hyper_set['discretization_radius_Qlearning'],
                l=param_.hyper_set['l']
            )
            alpha_partial_order = (param_.hyper_set['demand_zero_rate']**param_.hyper_set['L'])/2
        if param_.testing_flag:
            H = param_.hyper_set['H_for_testing']
        else:  
            H = np.round((1/(param_.hyper_set['demand_zero_rate']))**param_.hyper_set['L'], 0).astype(int) if param_.hyper_set['demand_zero_rate'] > 0 else 100
        if H > 1e7:
            print('warning: H may be too large:>')


        for algo in algo_names:
            if algo == 'our_algo':
                exp_env = our_algo.InformationOrderedPolicyElimination(param_, dynamics_, alpha_=alpha_partial_order)
            elif algo == 'random':
                exp_env = baselines.Random(param_, dynamics_)
            elif algo == 'optimal':
                exp_env = baselines.Optimal(param_, dynamics_)
            elif algo == 'feedback_graph':
                S_hat = len(S)
                exp_env = feedback_graph.FeedbackGraph(param_, dynamics_, X=X, S=S, A=A, H=H, S_hat=S_hat)
            elif algo == 'PPO':
                exp_env = PPO.FuncPPO(param_, dynamics_, X, S, A, H)
            elif algo == 'heuristic':
                exp_env = online_convex.ConvexityAlgo(param_, dynamics_, H, initial_state=X[0][0])
            elif algo == 'empirical_hindsight':
                exp_env = baselines.EmpiricalHindsight(param_, dynamics_)
            else:
                raise ValueError(f"Unknown algo {algo}")

            result = exp_env.run()
            results[algo][T].append(result)

if param_.testing_flag:
    print(results)
if param_.saving_ploting_flag:

    np.save('experiment_results.npy', results)
    plt.figure(figsize=(10, 6))
    
    for algo in algo_names:
        means = []
        lowers = []
        uppers = []
        for T in time_horizon_list:
            data = np.array(results[algo][T])
            mean = np.mean(data)
            sem = scipy.stats.sem(data)  
            ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., len(data)-1)  # 95%CI
            
            means.append(mean)
            lowers.append(mean - ci)
            uppers.append(mean + ci)
    
        plt.plot(time_horizon_list, means, label=algo)
        plt.fill_between(time_horizon_list, lowers, uppers, alpha=0.2)
    
    plt.xlabel('Time Horizon')
    plt.ylabel('Average Reward (or Cost)')
    plt.title('Performance vs Time Horizon')
    plt.legend()
    plt.grid(True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.savefig(f'performance_vs_timehorizon_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()



