import sys

class HyperParameters:
	def __init__(self):
		self.current_problem = 'M1L'
		self.problem_types = ['inventory_control', 'dual_index', 'M1L']
		self._problem_validation_check()
		if self.current_problem == 'inventory_control':
			self.hyper_set = self._inventory_problem_setting()
		if self.current_problem == 'dual_index':
			self.hyper_set = self._dual_index_setting()
		if self.current_problem == 'M1L':
			self.hyper_set = self._M1L()
		#self.hyper_algo_set = ['our_algo', 'random', 'PPO', 'heuristic', 'optimal', 'feedback_graph', 'empirical_hindsight', 'UCRL2']
		self.hyper_algo_set = ['our_algo', 'random', 'PPO', 'optimal', 'feedback_graph', 'empirical_hindsight', 'UCRL2']

		self.time_horizon_list = [100,1000,10000,100000]
		self.testing_horizon = 100000
		self.exp_repeat_times = 20
		self.testing_flag = 1
		self.saving_flag = 1
		# if 0 then running the exp, if 1 then just plotting
		self.plotting_flag = 0

		self.discretization_measure = 1

		self.feedbackgraph_update_fraction = 0.1
		self.bouns_scale_factor = 1e-4
		self.state_dimension_ = self._dimension_detection()

	def _problem_validation_check(self):
		if self.current_problem not in self.problem_types:
			sys.exit('non-valid problem claim')
		else:
			return 0

	def _dimension_detection(self):
		if self.current_problem == 'inventory_control':
			return 1 + self.hyper_set['L']
		if self.current_problem == 'dual_index':
			return 1 + self.hyper_set['L'] + self.hyper_set['l']


	def _inventory_problem_setting(self):
		val_dic = {'policy_set': [[0, 7]],
			'time_horizon': 100000,
			'holding_cost': 1,
			'shortage_penalty': 10, 
			'purchasing_cost': 0, 
			'purchasing_cost_expedit': 0,
			'L': 2,
			'l': 0,
			'distribution': 'exponential',
			'maximum_demand': 2,
			'discretization_radius_Qlearning': 1,
			'demand_zero_rate': 0.3,
			'H_for_testing': 20}
		return val_dic

	def _dual_index_setting(self):
		val_dic = {'policy_set': [[0, 30], [0, 200]], #[[short-term base-stock levels], [long-term base-stock levels]]
			'time_horizon': 10000,
			'holding_cost': 1,
			'shortage_penalty': 10, 
			'purchasing_cost': 0, 
			'purchasing_cost_expedit': 0.5,
			'L': 5,
			'l': 0,
			'distribution': 'exponential',
			'maximum_demand': 30,
			'discretization_radius_Qlearning': 1,
			'demand_zero_rate': 0.3,
			'H_for_testing': 20}
		return val_dic

	def _M1L(self):
		val_dic = {'policy_set': None,
			'time_horizon': 10000,
			'S': 2,
			'Amax': 3,
			'lambda_rate':6,
			'lambda_max': 10,
			'mu_rate': 3,
			'mu_max': 10,
			'C': 100}
		return val_dic


