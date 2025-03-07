# Sonia Nath - Teuscher Lab 

#spiking reservior using the mnist dataset to 
# explore and compare perforamce with SNN
# as well as in the future using STDP rule.


import numpy as np
import matplotlib.pyplot as plt
import optuna

def create_dataset(file_name, subset_size):
	# load the mnist training data CSV file into a list
	training_data_file = open(file_name, 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	if subset_size is not None:
		training_data_list = training_data_list[:subset_size]


	input_data = []
	label_data = []
	output_nodes = 10

	# go through all records in the training data set
	for record in training_data_list:
		all_values = record.split(',') # split the record by the ',' commas
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs
		targets = np.zeros(output_nodes) + 0.01 # create the target output values (all 0.01, except the desired label which is 0.99)
		# all_values[0] is the target label for this record
		targets[int(all_values[0])] = 0.99

		input_data.append(inputs)
		label_data.append(targets)

	return np.array(input_data), np.array(label_data)


class Reservior:
	def __init__(self, input_neurons,res_size, threshold, beta, mem_pot, spectral_radius, lr):
		self.input_neurons = input_neurons # number of input neurons
		self.res_size = res_size # reservior size
		self.threshold = threshold # treshold
		self.beta = beta # leak rate
		self.mem_pot = mem_pot # inital membrane potential
		self.spectral_radius = spectral_radius
		self.lr = lr # leanring rate


		sparse = 0.6

		#initalzing weights
		self.W_in = np.random.rand(res_size, input_neurons)
		self.W = np.random.rand(res_size, res_size)
		self.W_out = np.random.rand(10, res_size)

		self.W_in *= (np.random.rand(res_size, input_neurons) < sparse)
		self.W *= (np.random.rand(res_size, res_size) < sparse)

		# scaling res weights by spec radius 
		eigvals = np.linalg.eigvals(self.W)
		self.W = self.W / np.max(np.abs(eigvals)) * spectral_radius

		# intialzie res state 
		self.res_state = np.zeros(res_size)
		self.spk = np.zeros(res_size)

#		print(f"Input weights: {self.W_in}")
#		print(f"Res weights: {self.W}")
#		print(f"Output weights: {self.W_out}")



	def STDP_update(self, pre_syn, post_syn):
		A_plus = 0.1 # potentation rate
		A_minus = 0.1 # depression rate
		delta_t = pre_syn - post_syn # confused on

		# determine the ranges of pre-to-postsynaptic interspike intervals over which synaptic strengthening or weakening occurs
		T_plus = 10 
		T_minus = 10 

		# self.W += A_plus * (np.exp(delta_t) / T_plus) # LTP
		# self.W += -A_minus * (np.exp(delta_t) / T_minus) # LTD

		# if pre syn preceeds post syn spike then weights are strengthed (LTP)
		# if post syn preceeds pre syn spike then weights are weakened (LTD)




	def update(self, inputs):
		total_input = np.dot(self.W, self.spk) + np.dot(self.W_in,inputs.flatten())
		self.res_state = (1 - self.beta) * self.res_state + total_input


		#spikes 
		self.spk = self.res_state > self.threshold
		self.res_state[self.spk] = self.mem_pot # return to membrane potential 

		return self.spk

	def predict(self):
		x = np.dot(self.W_out, self.spk)
		return x


	def train_output(self, inputs, labels, error_list):
		epochs = 50
		error_list = []
		count = 0

		for epoch in range(epochs):
			total_error = 0
			length = len(inputs)
			for i in range(length):

				#getting spikes 
				self.spk = self.update(inputs[i])

				#output
				output = self.predict()

				# error computation
				error = labels[i] - output 
			#	print(f"Iteration: {count}, error: {np.abs(error).sum()}")
				count += 1
				total_error += np.abs(error).sum()

				# updating output weights
				self.W_out += self.lr * np.outer(error, self.spk)


			error_list.append(total_error / length)

			#computing average error 
			avg = total_error / length
			print(f"Epoch {epoch + 1} / {epochs}, Error: {avg.item():.4}")

		return error_list




#creating training and testing data
train_inputs, train_labels = create_dataset("/Users/wahhbeh/Documents/mnist_train.csv", subset_size=1000)
test_inputs, test_labels = create_dataset("/Users/wahhbeh/Documents/mnist_test.csv", subset_size=200)

# initalize reserverior
input_size = 28*28
res_size = 500
threshold = 1
beta = 0.8790635677507052
spectral_radius = 0.8008810010820975
lr = 0.01
error = []
accuracy_list = []
mem_pot = 0

res = Reservior(input_size,res_size, threshold, beta, mem_pot, spectral_radius, lr)


#train
error = res.train_output(train_inputs, train_labels, error)
	#test
correct = 0

for i in range(len(test_inputs)):
	spk = res.update(test_inputs[i])
	output = res.predict()

	predicted_label = np.argmax(output)
	actual_label = np.argmax(test_labels[i])

	if predicted_label == actual_label:
		correct += 1

#	print(f"Prediction: {predicted_label}, Actual: {actual_label}")	


accuracy = correct / len(test_inputs)
print(f"Accuracy: {accuracy* 100:.2f}")




'''
#using optuna to find best parameters
def objective(trial):
	res_size = trial.suggest_int("res_size", 500, 1000)
	threshold = trial.suggest_float("threshold", 0.5, 2.0)
	beta = trial.suggest_float("beta", 0.8, 1.0)
	spectral_radius = trial.suggest_float("spectral_radius", 0.8, 1.5)
	lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)
	error = []

	#initalzing reservoir
	res = Reservior(input_size,res_size, threshold, beta, mem_pot, spectral_radius, lr)


	#train
	error = res.train_output(train_inputs, train_labels, error)
	#test
	correct = 0

	for i in range(len(test_inputs)):
		spk = res.update(test_inputs[i])
		output = res.predict()

		predicted_label = np.argmax(output)
		actual_label = np.argmax(test_labels[i])

		if predicted_label == actual_label:
			correct += 1

		print(f"Prediction: {predicted_label}, Actual: {actual_label}")	

	accuracy = correct / len(test_inputs)
	return accuracy

#print(f"Accuracy: {accuracy* 100:.2f}")



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
#plt.show()

