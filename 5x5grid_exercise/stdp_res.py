#Sonia Nath - Teuscher Lab 

# Spiking Reservior made to solve simple classification
#problem of 5 x 5 grid. Guessing either horizontal line or 
# vertical line.

import numpy as np
import matplotlib.pyplot as plt
import optuna
import time

def create_dataset():

	inputs = []
	labels = []
	num_comb = 1000
	grid_size = 5
	noise = 0.1

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

		# adding noise
#		flip_bits = np.random.rand(grid_size, grid_size) < noise
#		grid[flip_bits] = 1 - grid[flip_bits]

		inputs.append(grid)
	
	print(f"torch.stack(inputs): {np.array(inputs).shape}")
	return np.array(inputs), np.array(labels)



class Reservior:
	def __init__(self, input_neurons,res_size, threshold, beta, mem_pot, spectral_radius, lr, A_plus, A_minus, T_plus, T_minus):
		self.input_neurons = input_neurons # number of input neurons
		self.res_size = res_size # reservior size
		self.threshold = threshold # treshold
		self.beta = beta # leak rate
		self.mem_pot = mem_pot # inital membrane potential
		self.spectral_radius = spectral_radius
		self.lr = lr

		# STDP variables
		self.A_plus = A_plus # LTP
		self.A_minus = A_minus # LTD
		self.T_plus = T_plus # LTP
		self.T_minus = T_minus # LTD

		sparse = 0.2

		#initalzing weights
		self.W_in = np.random.rand(res_size, input_neurons)
		self.W = np.random.rand(res_size, res_size)
		self.W_out = np.random.rand(1, res_size)
	
		self.W_in *= (np.random.rand(res_size, input_neurons) < sparse)
		self.W *= (np.random.rand(res_size, res_size) < sparse)
		
		# scaling res weights by spec radius 
		eigvals = np.linalg.eigvals(self.W)
		self.W = self.W / np.max(np.abs(eigvals)) * spectral_radius

		# intialzie res state 
		self.res_state = np.zeros(res_size)
		self.spk = np.zeros(res_size)
		self.last_spk_time = np.full(res_size, -np.inf) # keeping track of pre spks


	#update reservior neruon states & using STDP for weights	
	def update(self, inputs, current_t):
		total_input = np.dot(self.W, self.spk) + np.dot(self.W_in,inputs.flatten())
		self.res_state = (1 - self.beta) * self.res_state + total_input
		
		#spikes
		new_spk = self.res_state > self.threshold # seeing which neruon spiked.
		self.res_state[new_spk] = self.mem_pot # return to membrane potential 
	
		
		# updating weights with stdp every 5 timesteps 
		if current_t % 5 == 0:	
			for i in range(self.res_size):
				if new_spk[i]: # post syn spike
					for j in range(self.res_size):
						if i != j: # avoiding self connections
							if self.spk[j]: # if pre syn spiked then LTP
								delta_t = self.last_spk_time[i] - self.last_spk_time[j]
							# LTP
								if delta_t > 0:
									self.W[i, j] += self.A_plus * np.exp(-delta_t / self.T_plus)
						#LTD
								if delta_t < 0:
									self.W[i, j] -= self.A_minus * np.exp(delta_t / self.T_minus)

		self.last_spk_time[new_spk] = current_t
		self.spk = new_spk

		return self.spk
	
	# predict the output
	def predict(self):
		x = np.dot(self.W_out, self.spk)
		return 1/ (1 + np.exp(-x)) # applying sigmoid
#		return x

	
	# train the output layer
	def train_output(self, inputs, labels, error_list):
		epochs =  150
		error_list = []

		for epoch in range(epochs):
			total_error = 0
			length = len(inputs)
			for i in range(length):
				
				current_t = i
				#getting spikes 
				self.spk = self.update(inputs[i], current_t)
				
				#output
				output = self.predict()
				
				# error computation
				error = labels[i] - output 
				total_error += np.abs(error)
				
				self.W_out += self.lr * np.outer(error, self.spk)

			error_list.append(total_error / length)
			
			#computing average error 
			avg = total_error / length
			print(f"Epoch {epoch + 1} / {epochs}, Error: {avg.item():.4}\n\n")
		
		return error_list


'''
[I 2025-03-11 16:07:39,215] Trial 49 finished with value: 0.46 and parameters: {'res_size': 718, 'threshold': 1.1634769140472343, 'beta': 0.837245851284336, 'spectral_radius': 1.2546890479330655, 'lr': 0.00010259623475177071, 'A_plus': 0.18429262555721815, 'A_minus': 0.13405708668943517, 'T_plus': 16.884430662847457, 'T_minus': 12.743395822332317}. Best is trial 24 with value: 0.805.
Best Parameters: {'res_size': 669, 'threshold': 1.2455829862297048, 'beta': 0.8344897387753338, 'spectral_radius': 0.8960878996969639, 'lr': 0.0011192398817055941, 'A_plus': 0.19128149966298552, 'A_minus': 0.010955583093427243, 'T_plus': 12.384842002150416, 'T_minus': 16.30839758879683}
Best Accuracy: 0.805
'''


'''
Best is trial 44 with value: 0.88.
Best Parameters: {'res_size': 555, 'threshold': 1.756225049143497, 'beta': 0.9093417385101635, 'spectral_radius': 1.4557156621825478, 'lr': 0.008205985657245275, 'A_plus': 0.2968296602795653, 'A_minus': 0.328861813551042, 'T_plus': 16.27242424863669, 'T_minus': 9.710864724717801}
Best Accuracy: 0.88
'''



# initalize reserverior
input_size = 5*5
res_size = 555
threshold = 1.756225049143497
beta = 0.9093417385101635
mem_pot = 0 
spectral_radius = 1.4557156621825478
lr = 0.008205985657245275
A_plus = 0.2968296602795653
A_minus = 0.328861813551042
T_plus = 16.27242424863669
T_minus = 9.710864724717801
error = []
accuracy_list = []


inputs, labels = create_dataset()
 

res = Reservior(input_size,res_size, threshold, beta, mem_pot, spectral_radius, lr, A_plus, A_minus, T_plus, T_minus)


#splitting data 800 (80%) for train 200 (20%) for test
train_inputs, test_inputs = inputs[:800], inputs[800:]
train_labels, test_labels = labels[:800], labels[800:]


print("Lets GO!")
start = time.time()

#train
error = res.train_output(train_inputs, train_labels, error)

#test
correct = 0

for i in range(len(test_inputs)):
	spk = res.update(test_inputs[i], i)
	output = res.predict()
	if output > 0.5:
		prediction = 1
	else:
		prediction = 0
	print(f"Prediction: {prediction}, Actual: {test_labels[i]}")	
	if prediction == test_labels[i]:
		correct += 1



accuracy = correct / len(test_inputs)
print(f"Accuracy: {accuracy* 100:.2f}")

end = time.time()

print("All done!")
print(f"Total Time: {end - start}")


'''

#using optuna to find best parameters
def objective(trial):
	res_size = trial.suggest_int("res_size", 500, 800)
	threshold = trial.suggest_float("threshold", 0.5, 2.0)
	beta = trial.suggest_float("beta", 0.8, 1.0)
	spectral_radius = trial.suggest_float("spectral_radius", 0.8, 1.5)
	lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
	A_plus = trial.suggest_float("A_plus", 0.01, 0.5)
	A_minus = trial.suggest_float("A_minus", 0.01, 0.5)
	T_plus = trial.suggest_float("T_plus", 5, 20)
	T_minus = trial.suggest_float("T_minus", 5, 20)
	error = []
	

#	inputs, labels = create_dataset()
 
	res = Reservior(input_size,res_size, threshold, beta, mem_pot, spectral_radius, lr, A_plus, A_minus, T_plus, T_minus)


#splitting data 800 (80%) for train 200 (20%) for test
	train_inputs, test_inputs = inputs[:800], inputs[800:]
	train_labels, test_labels = labels[:800], labels[800:]


	print('Lets GO!')
	start = time.time()

#train
	error = res.train_output(train_inputs, train_labels, error)

#test
	correct = 0

	for i in range(len(test_inputs)):
		spk = res.update(test_inputs[i], i)
		output = res.predict()
		if output > 0.5:
			prediction = 1
		else:
			prediction = 0
		print(f"Prediction: {prediction}, Actual: {test_labels[i]}")	
		if prediction == test_labels[i]:
			correct += 1



	accuracy = correct / len(test_inputs)
	print(f"Accuracy: {accuracy* 100:.2f}")

	end = time.time()

	print("All done!")
	print(f"Started {start}")
	print(f"Finished {end}")
	print(f"Total Time: {start - end}")
	return accuracy

	

#Best is trial 21 with value: 1.0.
#Best Parameters: {'res_size': 1188, 'threshold': 1.0393349533404501, 'beta': 0.8790635677507052, 'spectral_radius': 0.8008810010820975, 'lr': 0.05359473671289511}
study = optuna.create_study(direction="maximize")  # We negate accuracy, so we minimize
study.optimize(objective, n_trials=50)
print("Best Parameters:", study.best_params)
print("Best Accuracy:", study.best_value) 
'''

# Plot training error over epochs
plt.figure(figsize=(5, 5))
plt.plot(error, label="Training Loss") #, linestyle='-', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()
