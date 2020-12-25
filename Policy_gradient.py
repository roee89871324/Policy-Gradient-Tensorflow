r"""
Date: 19.2.2019 
Author: R.E
Version: 1.0

Policy gradient on cartpole environment.

Tensorflow variables: Begin with name of neuralnet it belongs to (P for policy neuralnet or V for value neuralnet) 
		and ends with '_'.
Numpy arrays: end with "_1d" to denote dimension of array.

Written after this guide: http://kvfrans.com/simple-algoritms-for-solving-cartpole/
Recommended Video for backround knowledge about Policy Gradient by Stanford: https://www.youtube.com/watch?v=lvoHnicueoE: 
"""

import gym; gym.logger.set_level(40)
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import pandas
import random
import math

MAX_NUM_STEPS = 200 # maximum number of steps per trajectory (single game execution)

def evaluate_trained_algo(env, trained_algo, iterations=1):
	"""
	Run given trained algo N times and avgs the reward to asses how good the algo is.
	"""

	total_reward_sum = 0
   	
	for _ in range(int(iterations)):
		# init
		observation_1d = env.reset()
		reward_of_cur_iteration = 0

		# run environment until stop
		for _ in range(MAX_NUM_STEPS):
			# set action to 0 or 1 (left or right) based on weights
			action = trained_algo.predict(observation_1d)
			# run a step and get back relevant things
			observation_1d, reward_of_cur_step, done, info = env.step(action)
			# sum reward
			reward_of_cur_iteration += reward_of_cur_step
			if done: break

		total_reward_sum += reward_of_cur_iteration
	
	iterations_reward_avg = float(total_reward_sum) / iterations
	return iterations_reward_avg

def main():
	# configuration
	env = gym.make('CartPole-v0')
	RANDOM_SEED=1; np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); tf.random.set_random_seed(RANDOM_SEED); env.seed(RANDOM_SEED)
	N=1
	TRAIN_ITERATIONS=5e2
	EVAL_ITERATIONS=1e2
	my_algo = Policy_Gradient(env)

	# evaluate policy gradient
	for i in range(N):
		my_algo.train(iterations=TRAIN_ITERATIONS, reset=True)
		print (evaluate_trained_algo(env, my_algo, iterations=EVAL_ITERATIONS))

#######################################################################################
############################# ALGORITHM ##############################################
#######################################################################################
class Algo():
	def __init__(self):
		raise NotImplementedError

	def train(self, *args, **kwargs):
		raise NotImplementedError		

	def predict(self, observation_1d):
		raise NotImplementedError		

class Policy_Gradient(Algo):
	"""
	Policy Gradient Algorithm.
	"""

	def __init__(self, env, discount_factor=0.97):

		# hyperparameters
		self.env = env
		self.discount_factor = discount_factor

		### neuralnetwork - P(observation)=action, optimize(observation, action, advantage).
		with tf.variable_scope("policy_neuralnet"):
			## make prediction by logistic regression
			self.P_observations_input_  = tf.placeholder("float", [None,4])
			P_w1_ = tf.get_variable("P_w1", [4,2])
			# calculate probability for each action for every observation.
			self.P_pred_action_probabilities_ = tf.nn.softmax(tf.matmul(self.P_observations_input_, P_w1_))
			
			## optimize P neuralnet
			# action + advantage, used for optimization.
			self.P_actions_input_, self.P_advantages_input_ = tf.placeholder("float",[None,2]), tf.placeholder("float",[None,1])
			# convert vector of probabilities for every action, to a vector of the probability of the action taken (1d)
			P_prob_of_action_taken_ = tf.reduce_sum(tf.multiply(self.P_pred_action_probabilities_, self.P_actions_input_), reduction_indices=[1])
			# maximize how good your action was (as compared to baseline, aka advantage) times the probability to take it.
			P_loss_ = -tf.reduce_sum(tf.log(P_prob_of_action_taken_) * self.P_advantages_input_)
			self.P_optimizer_ = tf.train.AdamOptimizer(0.01).minimize(P_loss_)

		### neuralnetwork - V(observation)=estimated_future_reward, optimize(observation, true_reward).
		with tf.variable_scope("value_neuralnet"):
			## make prediction by 1 inner layer neuralnet
			self.V_observations_input_  = tf.placeholder("float", [None,4])
			V_w1_ = tf.get_variable("V_w1_", [4,10])
			V_b1_ = tf.get_variable("V_b1_",[10])
			V_h1_ = tf.nn.relu(tf.matmul(self.V_observations_input_, V_w1_) + V_b1_)
			V_w2_ = tf.get_variable("V_w2_",[10,1])
			V_b2_ = tf.get_variable("V_b2_",[1])
			self.V_pred_future_reward_ = tf.matmul(V_h1_, V_w2_) + V_b2_
			
			## optimize 
			self.V_actual_future_reward_input_ = tf.placeholder("float", [None,1])
			# minimize how much wrong neural net was from the actual future reward
			V_loss_ = tf.nn.l2_loss(self.V_pred_future_reward_ - self.V_actual_future_reward_input_)
			self.V_optimizer_ = tf.train.AdamOptimizer(0.1).minimize(V_loss_)

		# define tensorflwo graph
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

	def _compute_trajectory(self):
		"""
		compute single einvronment trajectory (until 'done' is reached).
		Trajectory format: [(1, observation_1, action_1, reward_1), (2, observation_2, action_2, reward_2)...etc]
		"""

		observation_1d = self.env.reset()
		trajectory = []
		for transition_index in range(MAX_NUM_STEPS):
			# compute which action to take based on current pred neural net and observation
			action = self.predict(observation_1d)
			# take the action and record current transition: transition_index, observation received, action taken and reward received.
			new_observation_1d, reward, done, _ = self.env.step(action)
			trajectory.append((transition_index, observation_1d, action, reward))
			# break if done received
			if done: break
			# reinit observation for next iteration
			observation_1d = new_observation_1d

		return trajectory

	def _compute_transitions_advantage(self, trajectory):
		"""
		For every transition, compute the advantage of how the trajectory went vs how our value neuralnet predicted.
		Also return other stats per transition that are required for the neural net training.
		"""

		# make a list of rewards of trajectory, do this prior, in order to make future calculations easier
		rewards = [reward for _, _, _, reward in trajectory]

		advantages, true_future_rewards, observations, actions = [], [], [], []
		for cur_transition_index, observation_1d, action, _ in trajectory:
			# estimate future reward from given current transition observation
			estimated_future_reward = self.sess.run(self.V_pred_future_reward_, feed_dict={self.V_observations_input_: np.expand_dims(observation_1d, axis=0)})[0][0]
			# for each action and observation pair, calculate actual the future reward in the trajectory.
			true_future_reward, multiplier = 0.0, 1.0
			for future_transition_reward in rewards[cur_transition_index:]:
				true_future_reward += future_transition_reward * multiplier
				multiplier = multiplier * self.discount_factor

			# make lists for the neural networks optimization
			advantages.append(true_future_reward - estimated_future_reward)
			true_future_rewards.append(true_future_reward)
			observations.append(observation_1d)
			action_onehot = np.zeros(2); action_onehot[action] = 1; actions.append(action_onehot)

		return advantages, true_future_rewards, observations, actions

	def train(self, iterations=1e3, reset=False):
		"""
		train the value and policy

		param iterations (float): number of training iterations
		param reset (boolean): whether to erase past training of the model.
		"""

		# retrain the model.
		if reset: self.sess.run(tf.global_variables_initializer())

		for i in range(int(iterations)):
			# compute cur iteration Prediction neuralnet trajectory.
			trajectory = self._compute_trajectory()
			# computer advantages by comparing trajectory true rewards to estimated rewards from value neuralnet.
			advantages, true_future_rewards, observations, actions = self._compute_transitions_advantage(trajectory)
			
			# optimize both neuralnets
			self.sess.run(self.P_optimizer_, feed_dict={self.P_observations_input_: observations, self.P_actions_input_: actions, self.P_advantages_input_: np.expand_dims(advantages, axis=1)})
			self.sess.run(self.V_optimizer_, feed_dict={self.V_observations_input_: observations, self.V_actual_future_reward_input_: np.expand_dims(true_future_rewards, axis=1)})

	def predict(self, observation_1d):
		"""
		Make prediction given an observation, return action to perform in environment.
		"""

		# compute probability to take every possible action given observation
		action_probabilities_2d = self.sess.run(self.P_pred_action_probabilities_, feed_dict={self.P_observations_input_: np.expand_dims(observation_1d, axis=0)})
		firstaction_probability = action_probabilities_2d[0][0]
		# choose action random in relation to the probabilities returned
		action = 0 if random.uniform(0,1) < firstaction_probability else 1

		return action

if __name__ == "__main__":
	start = time()
	try:
		main()
	finally:
		print ("\nTook: %.3f s" % (time() - start))