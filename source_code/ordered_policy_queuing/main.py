import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import multiprocessing
import copy
import json
import sys
import os
from itertools import product


import hyperparameters
import our_algo
import ic
import baselines
import feedback_graph
import PPO
import PPO_gym
import PPO_MM1L
import online_convex
import queuing_model
import UCRL2


import gc
from tqdm import tqdm
import datetime
import platform
import time




#notice that in our algo, we set the policy to be integer in line 79 to align with other algos especially the Q-learning ones.
#purchasing cost has to be 0
#the epsilon for optimal/empirical_hindsight are 0.1, fixed in the baselines file.

def run_one_experiment(algo, param_copy, dynamics_copy, S, A, X, H, alpha_, seed_):
    print('===' + algo + '===')
    if algo == 'our_algo':
        exp_env = our_algo.InformationOrderedPolicyElimination(param_copy, dynamics_copy, alpha_=alpha_, seed_ = seed_)
    elif algo == 'random':
        exp_env = baselines.Random(param_copy, dynamics_copy, seed_ = seed_)
    elif algo == 'optimal':
        exp_env = baselines.Optimal(param_copy, dynamics_copy, seed_)
    elif algo == 'feedback_graph':
        S_hat = len(S)
        exp_env = feedback_graph.FeedbackGraph(param_copy, dynamics_copy, X=X, S=S, A=A, H=H, S_hat=S_hat, seed_ = seed_)
    elif algo == 'PPO':
        #exp_env = PPO.FuncPPO(param_copy, dynamics_copy, X, S, A, H, seed_ = seed_)
        exp_env = PPO_gym.PPOTrainer(env_class = PPO_gym.ContinuousEnv, param_ = param_copy, dynamics_ = dynamics_copy, S_box = param_copy.hyper_set['policy_set'], A_box = param_copy.hyper_set['policy_set'], dimension_ = param_copy.state_dimension_, H = H, seed_=seed_)
    elif algo == 'heuristic':
        exp_env = online_convex.ConvexityAlgo(param_copy, dynamics_copy, H, seed_ = seed_)
    elif algo == 'empirical_hindsight':
        exp_env = baselines.EmpiricalHindsight(param_copy, dynamics_copy, seed_ = seed_)
    else:
        raise ValueError(f"Unknown algo {algo}")

    return exp_env.run()

def run_one_experiment_queuing(algo, param_copy, dynamics_copy, seed_):
    print('===' + algo + '===')
    if algo == 'our_algo':
        exp_env = our_algo.InformationOrderedPolicyElimination(param_copy, dynamics_copy, alpha_=1, seed_ = seed_)
    elif algo == 'random':
        exp_env = baselines.Random(param_copy, dynamics_copy, seed_ = seed_)
    elif algo == 'optimal':
        exp_env = baselines.Optimal(param_copy, dynamics_copy, seed_)
    elif algo == 'feedback_graph':
        S_hat = 1
        exp_env = feedback_graph.FeedbackGraph(param_copy, dynamics_copy, X=[a + b for a, b in product([(s,) for s in range(param_copy.hyper_set['S'] + 1)], [(s,) for s in range(param_copy.hyper_set['Amax'] + 1)])], S=[(s,) for s in range(param_copy.hyper_set['S'] + 1)], A=[(s,) for s in range(param_copy.hyper_set['Amax'] + 1)], H=1, S_hat=None, seed_ = seed_)
    elif algo == 'PPO':
        #exp_env = PPO.FuncPPO(param_copy, dynamics_copy, X, S, A, H, seed_ = seed_)
        exp_env = PPO_MM1L.PPOKernel(param_ = param_copy, dynamics_ = dynamics_copy, seed_=seed_)
    elif algo == 'empirical_hindsight':
        exp_env = baselines.EmpiricalHindsight(param_copy, dynamics_copy, seed_ = seed_)
    elif algo == 'UCRL2':
        exp_env = UCRL2.UCRL2Learner(param_copy, dynamics_copy, delta = 0.01, measure = param_copy.discretization_measure, seed_ = seed_)
    else:
        raise ValueError(f"Unknown algo {algo}")

    return exp_env.run()


def main():
    # Parameters
    arg_v = sys.argv[0]
    if platform.system() != 'Darwin':
        multiprocessing.set_start_method('spawn', force = True)
    param_ = hyperparameters.HyperParameters()
    time_horizon_list = param_.time_horizon_list
    k = param_.exp_repeat_times
    algo_names = param_.hyper_algo_set
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not param_.plotting_flag:
    
        results = {algo: {T: [] for T in time_horizon_list} for algo in algo_names}
        
        
        # Main experiment loop
        print(f"\n=== {param_.current_problem} ===")
        for T in time_horizon_list:
            print(f"\n=== Time Horizon: {T} ===")
            param_.hyper_set['time_horizon'] = T
        
            if param_.current_problem in ['inventory_control', 'dual_index']:
                dynamics_ = ic.InventoryControl(
                    param_.hyper_set['policy_set'],
                    param_.hyper_set['time_horizon'],
                    param_.hyper_set['holding_cost'],
                    param_.hyper_set['shortage_penalty'],
                    param_.hyper_set['purchasing_cost'],
                    param_.hyper_set['demand_zero_rate'],
                    param_.hyper_set['L'],
                    param_.hyper_set['l'],
                    param_.hyper_set['maximum_demand'],
                    param_.hyper_set['distribution'],
                    param_.hyper_set['purchasing_cost_expedit']
                )
                if 'feedback_graph' in algo_names:
                    if param_.current_problem == 'inventory_control':
                        S, A, X = dynamics_.construct_state_action_space(
                            b=param_.hyper_set['policy_set'][0][1],
                            L=param_.hyper_set['L']+1,
                            r=param_.hyper_set['discretization_radius_Qlearning']
                        )
                    else:
                        S, A, X = dynamics_.construct_state_action_space(
                            b=param_.hyper_set['policy_set'][1][1],
                            L=param_.hyper_set['L']+1,
                            r=param_.hyper_set['discretization_radius_Qlearning'],
                            l=param_.hyper_set['l']
                        )
                else:
                    S, A, X = None, None, None
                alpha_partial_order = None
                if 'our_algo' in algo_names:
                    if param_.current_problem == 'inventory_control':
                        alpha_partial_order = 1
                    else:
                        alpha_partial_order = (param_.hyper_set['demand_zero_rate']**param_.hyper_set['L'])/2
                if param_.testing_flag:
                    H = param_.hyper_set['H_for_testing']
                else:  
                    H = np.round((1/(param_.hyper_set['demand_zero_rate']))**param_.hyper_set['L'], 0).astype(int) if param_.hyper_set['demand_zero_rate'] > 0 else 100
                if H > 1e7:
                    print('warning: H may be too large:>')

            elif param_.current_problem in ['M1L']:
                dynamics_ = queuing_model.DVFSMDP(
                    param_.hyper_set['S'],
                    param_.hyper_set['Amax'],
                    param_.hyper_set['lambda_rate'],
                    param_.hyper_set['lambda_max'],
                    param_.hyper_set['mu_rate'],
                    param_.hyper_set['mu_max'],
                    param_.hyper_set['C'],
                    param_.hyper_set['time_horizon'])

            else:
                sys.exit('wrong problem setting')
            
            parent_seed = np.random.SeedSequence(int(time.time()))
            n_workers = max(1, multiprocessing.cpu_count() - 2)
            child_seeds = parent_seed.spawn(n_workers) 
        
            pool = multiprocessing.Pool(processes=n_workers)
            task_list = []
            
            for algo in algo_names:
                tasks = []
                for idx in range(k):
                    param_copy = copy.deepcopy(param_)
                    dynamics_copy = copy.deepcopy(dynamics_)
                    if param_.current_problem in ['inventory_control', 'dual_index']:
                        tasks.append((algo, param_copy, dynamics_copy, S, A, X, H, alpha_partial_order, child_seeds[idx]))
                    else:
                        tasks.append((algo, param_copy, dynamics_copy, child_seeds[idx]))
                
                if param_.current_problem in ['inventory_control', 'dual_index']:
                    results_list = pool.starmap(run_one_experiment, tasks)
                else:
                    results_list = pool.starmap(run_one_experiment_queuing, tasks)
                results[algo][T].extend(results_list)  
            
            pool.close()
            pool.join()
    
        if not param_.current_problem == 'M1L':
            results = dynamics_.denormalize_(results, parent_seed.spawn(1)[0])
        print(results)
        
        if param_.saving_flag:
            np.save('results/experiment_results_' + arg_v + '_.npy', results)

    else:
        results = np.load('results/experiment_results_.npy', allow_pickle=True).item()

        v = []

        #with open('output.txt', 'w') as f:
        #    json.dump(results, f, indent=4)

        plt.figure(figsize=(10, 6))
        
        for algo in algo_names:
            means = []
            lowers = []
            uppers = []
            for T in time_horizon_list:
                data = np.array([v[1] for v in results[algo][T]])
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
    
    
        plt.savefig(f'results/performance_vs_timehorizon_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    main()
    
    