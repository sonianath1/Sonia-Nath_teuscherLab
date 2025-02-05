import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



def create_dataset():

	inputs = []
	labels = []
	num_comb = 50
	grid = 5

	for i in range(grid):
		for j in range(num_comb): # creating 20 samples of each combination
		#horizontal = torch.zeros(num_steps,grid, grid) # feeding 50 training samples of each 5 x 5 grid
			horizontal = torch.zeros(grid, grid)
			horizontal[i, :] = 1 #selected row is set to 1
			inputs.append(horizontal)
			labels.append(0) # 0 for horizintal

	for i in range(grid):
		for j in range(num_comb): # creating 20 samples of each combination
			vertical = torch.zeros(grid, grid)
			vertical[:, i] = 1
			inputs.append(vertical)
			labels.append(1) # 1 for vertical	

#	print(f"Inputs: {inputs}")
#	print(f"Inputs torch stack: {torch.stack(inputs)}")
	# torch.stack(inputs) turns into (10, 5, 5)
	return torch.stack(inputs), torch.tensor(labels)
#	return inputs, torch.tensor(labels)



class Reservior:
	def __init__(self, input_neurons,res_size, threshold, beta, mem_pot, spectral_radius, lr):
		self.input_neurons = input_neurons # number of input neurons 
		self.res_size = res_size # reservior size
		self.threshold = threshold # treshold
		self.beta = beta # leak rate
		self.mem_pot = mem_pot # inital membrane potential
		self.spectral_radius = spectral_radius
		self.lr = lr # leanring rate
		
		
		#initalzing state and spike 
		self.res_state = np.zeros(self.res_size)
		self.spk = np.zeros(self.res_size)

		#initalzing weights
		self.W_in = np.random.rand(self.res_size, self.input_neurons)
		self.W = np.random.rand(self.res_size, self.res_size)
		self.W_out = np.random.rand(1, self.res_size)
	
		# altering res weights once with spectral radius
		eigvals = np.linalg.eigvals(self.W)
		self.W = self.W / np.max(eigvals) * self.spectral_radius
	

	
	def update(self, inputs):
		total_input = np.dot(self.W, self.spk) + np.dot(self.W_in,inputs.flatten())
		self.res_state = (1 - self.beta) * self.res_state + total_input
		self.spk = self.res_state > self.threshold
		self.res_state[self.spk] = self.mem_pot
#		print(f"spk shape: {self.spk.shape}")

	def predict(self):
		return np.dot(self.W_out, self.spk)

	def error(self, targets):
		return targets - self.predict()
#		return 1 / (1 + np.exp(-x)) # applying sigmoid function



	def train_output(self, error):
#		print(f"error shape: {error.shape}, sike shape: {self.spk.shape}")
		self.W_out += self.lr * np.outer(error, self.spk)



# creating input / labels for training and testing
inputs, labels = create_dataset()
#inputs = inputs.view(inputs.shape[0], -1)

train_input, test_input, train_label, test_label = train_test_split(inputs, labels, test_size=0.2, random_state=42)


#train_dataset = TensorDataset(train_input, train_label)
#test_dataset = TensorDataset(test_input, test_label)

#train = DataLoader(train_dataset, batch_size=16, shuffle=True)
#test = DataLoader(test_dataset, batch_size=16, shuffle=False)


# intalizing variables
input_neurons = inputs.shape[1] * inputs.shape[2]
res_size = 1000
threshold = 1
beta = 0.7
mem_pot = 0
spec_radius = 0.5
lr = 1e-4


# = Reservior(input_neurons,res_size, threshold, beta, mem_pot, spec_radius, lr).to(device)



#creating reservior 
res = Reservior(input_neurons,res_size, threshold, beta, mem_pot, spec_radius, lr)

	

# training 
for i in range(train_input.shape[0]):
	res.update(train_input[i])
	errors = res.error(train_label[i])
	print(f"Epoch {i + 1}/{train_input.shape[0]}")
	print(f"errors: {errors.item()}")
	train = res.train_output(errors)



correct = 0
for i in range(test_input.shape[0]):
	res.update(test_input[i])
	predict = res.predict()[0] # extracts only value from np array
	if round(predict) == test_label[i].item():
		correct += 1



accuracy = correct / test_input.shape[0]
print(f"Test accuracy: {accuracy:.2%}")
