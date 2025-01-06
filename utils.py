import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_data):
		observation, capacities , video_quality, audio_quality = batch_data
		
		self.state = observation
		self.action = capacities
		
		batch_size, seq_length, state_size = observation.shape
		self.next_state = torch.zeros(batch_size, seq_length, state_size)
		self.next_state[:,:-1,:] = observation[:,1:,:]
		
		last_recv_rate = observation[:,:,1*10-10]
		last_loss = observation[:,:,11*10-10]
		last_delay = observation[:,:,4*10-10]
		
		self.reward =  0.5* last_recv_rate/(8*1e6) + video_quality/5.0 + audio_quality/5.0 - 0.5* last_loss  - 0.3 * (last_delay / 500.0) 
		# + video_quality/5.0 + audio_quality/5.0    

		self.not_done = torch.zeros(batch_size, seq_length, 1) + 1
		self.not_done[:,-1,:] = self.not_done[:,-1,:] - 1

		return (
			torch.FloatTensor(self.state).to(self.device),
			torch.FloatTensor(self.action).to(self.device),
			torch.FloatTensor(self.next_state).to(self.device),
			torch.FloatTensor(self.reward).to(self.device),
			torch.FloatTensor(self.not_done).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std