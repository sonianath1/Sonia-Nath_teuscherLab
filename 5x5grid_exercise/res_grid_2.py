# Sonia Nath - Teuscher Lab 

# Spiking Reservior made to solve simple classification
#problem of 5 x 5 grid. Guessing either horizontal line or 
# vertical line.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import optuna

def create_dataset():

	inputs = []
	labels = []
	num_comb = 1000
	grid_size = 5
	
	for _ in range(num_comb):
		grid = np.zeros((grid_size, grid_size))
		if np.random.rand() < 0.5:
			row = np.random.randint(0, grid_size)
			grid[row, :] = 1
			labels.append(1)
		else:
			col = np.random.randint(0, grid_size)
			grid[:, col] = 1
			labels.append(0)
		inputs.append(grid)
	
	return np.array(inputs), np.array(labels)



class Reservior:
	def __init__(self, input_neurons,res_size, threshold, beta, mem_pot, spectral_radius, lr):
		self.input_neurons = input_neurons # number of input neurons
		self.res_size = res_size # reservior size
		self.threshold = threshold # treshold
		self.beta = beta # leak rate
		self.mem_pot = mem_pot # inital membrane potential
		self.spectral_radius = spectral_radius
		self.lr = lr # leanring rate


		sparse = 0.2

		#initalzing weights
		self.W_in = np.random.randn(res_size, input_neurons)
		self.W = np.random.randn(res_size, res_size)
		self.W_out = np.random.randn(1, res_size)
	
		self.W_in *= (np.random.rand(res_size, input_neurons) < sparse)
		self.W *= (np.random.rand(res_size, res_size) < sparse)
		
		# scaling res weights by spec radius 
		eigvals = np.linalg.eigvals(self.W)
		self.W = self.W / np.max(np.abs(eigvals)) * spectral_radius

		# intialzie res state 
		self.res_state = np.zeros(res_size)

#		print(f"Input weights: {self.W_in}")
#		print(f"Res weights: {self.W}")
#		print(f"Output weights: {self.W_out}")


	def update(self, inputs):
		total_input = np.dot(self.W, self.res_state) + np.dot(self.W_in,inputs.flatten())
		self.res_state = (1 - self.beta) * self.res_state + total_input
		

		#spikes 
		spk = self.res_state > self.threshold
		self.res_state[spk] = self.mem_pot # return to membrane potential 

#		print(f"spk shape: {self.spk.shape}")
		return spk

	def predict(self, spk):
		x = np.dot(self.W_out, spk)
		return 1/ (1 + np.exp(-x)) # applying sigmoid
#		return x

#	def error(self, targets):
		#return targets - self.predict()


	def train_output(self, inputs, labels, error_list):
		epochs = 100
		error_list = []

		for epoch in range(epochs):
			total_error = 0
			length = len(inputs)
			for i in range(length):
				
				#getting spikes 
				spk = self.update(inputs[i])
				
				#output
				output = self.predict(spk)
				
				# error computation
				error = labels[i] - output 
				total_error += np.abs(error)
				error_list.append(total_error / length)

				# updating output weights
				self.W_out += self.lr * np.outer(error, spk)
			
			#computing average error 
			avg = total_error / length
			print(f"Epoch {epoch + 1} / {epochs}, Error: {avg.item():.4}")
		
		return error_list


#using optuna to find best parameters
#def objective(trial):
#res_size = trial.suggest_int("res_size", 500, 2000)
#threshold = trial.suggest_float("threshold", 0.5, 2.0)
#beta = trial.suggest_float("beta", 0.8, 1.0)
#spectral_radius = trial.suggest_float("spectral_radius", 0.8, 1.5)
#lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)

#Best is trial 21 with value: 1.0.
#Best Parameters: {'res_size': 1188, 'threshold': 1.0393349533404501, 'beta': 0.8790635677507052, 'spectral_radius': 0.8008810010820975, 'lr': 0.05359473671289511}

# initalize reserverior
input_size = 5*5
res_size = 1188
threshold = 1.0393349533404501
beta = 0.8790635677507052
mem_pot = 0 
spectral_radius = 0.8008810010820975
lr = 0.05359473671289511
error = []



inputs, labels = create_dataset()
 
res = Reservior(input_size,res_size, threshold, beta, mem_pot, spectral_radius, lr)


#splitting data 800 (80%) for train 200 (20%) for test
train_inputs, test_inputs = inputs[:800], inputs[800:]
train_labels, test_labels = labels[:800], labels[800:]




#train
error = res.train_output(train_inputs, train_labels, error)

#test
correct = 0

for i in range(len(test_inputs)):
	spk = res.update(test_inputs[i])
	output = res.predict(spk)
	if output >= 0.5:
		prediction = 1
	else:
		prediction = 0
	print(f"Prediction: {prediction}, Actual: {test_labels[i]}")	
	if prediction == test_labels[i]:
		correct += 1



accuracy = correct / len(test_inputs)
print(f"Accuracy: {accuracy* 100:.2f}")
#return accuracy

#study = optuna.create_study(direction="maximize")  # We negate accuracy, so we minimize
#study.optimize(objective, n_trials=50)
#print("Best Parameters:", study.best_params)
#print("Best Accuracy:", study.best_value) 


# Plot training error over epochs
plt.figure(figsize=(5, 5))
plt.plot(error, label="Training Error", linestyle='-', marker='o')
#plt.plot(test_errors, label="Testing Error", linestyle='-', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title("Training & Test Error Over Time")
plt.legend()
plt.grid(True)
plt.show()
