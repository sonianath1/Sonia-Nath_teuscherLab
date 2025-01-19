import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")



def create_dataset():
	inputs = []
	labels = []
	num_steps = 50
	grid = 5

	for i in range(grid):
		#horizontal = torch.zeros(num_steps,grid, grid) # feeding 50 training samples of each 5 x 5 grid
		horizontal = torch.zeros(grid, grid)
		horizontal[i, :] = 1 #selected row is set to 1
		inputs.append(horizontal)
		labels.append(0) # 0 for horizintal

	for i in range(grid):
		vertical = torch.zeros(grid, grid)
		vertical[:, i] = 1
		inputs.append(vertical)
		labels.append(1) # 1 for vertical
	
	# torch.stack(inputs) turns into (10, 5, 5)
	return torch.stack(inputs), torch.tensor(labels)


num_inputs = 5*5
num_hidden = 20
num_outputs = 2
grid = 5

# temporal dynamics
num_steps = 50
beta = 0.9 #trying different decays



# defining network
class Net(nn.Module):
	def __init__(self):
		super().__init__()
		#initalize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs)
		self.lif2 = snn.Leaky(beta=beta)

	def forward(self, x):

		#initalize hidden states at t =0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()

		#record final layer
		spk2_rec = []
		mem2_rec = []

		for step in range(num_steps):
			cur1 = self.fc1(x.flatten(1)) # flatten to 1d
			spk1, mem1 = self.lif1(cur1, mem1) # procoess through first lif layer
			cur2 = self.fc2(spk1) # passing spike output from first to second
			spk2, mem2 = self.lif2(cur2, mem2) # process trhough second lif

			#store
			spk2_rec.append(spk2)
			mem2_rec.append(mem2)
	
		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)



net = Net().to(device)
dtype = torch.float
# creating dataset
inputs, labels = create_dataset()
train_input, test_input, train_label, test_label = train_test_split(inputs, labels, test_size=0.2, random_state=42)

#print(inputs.shape, labels.shape)
#print(train_input.shape, test_input.shape)

train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)

train = DataLoader(train_dataset, batch_size=16, shuffle=True)
test = DataLoader(test_dataset, batch_size=16, shuffle=False)

loss = nn.CrossEntropyLoss() # calcualting error
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


num_epochs = 10
counter = 0
repeat = 2# passing each sample 50 times for spike encoding

for epoch in range(num_epochs):
	print(f"Epoch {epoch + 1}/{num_epochs}")
	train_batch = iter(train)

	#training minibatches
	for data, targets in train_batch:
		data = data.to(device)
		targets = targets.to(device)

		for _ in range(repeat): # spike encoding

			#forward pass
			net.train()
			spk_rec, _ = net(data)


			#initalize the loss and sum over time
			loss_val = torch.zeros((1), dtype=dtype, device=device)
			loss_val += loss(spk_rec.sum(0), targets)

	
			#gradient calculation + weight update
			optimizer.zero_grad()
			loss_val.backward()
			optimizer.step()

	
			#print train.test loss/accruary
			print(f"Iteration: {counter} \t Train loss: {loss_val.item()}")
			counter += 1

		if counter == 100:
			break

def measure_accuracy(model, dataloader):

	with torch.no_grad():
		model.eval()
		running_length = 0
		running_accuracy = 0

		for data, targets in iter(dataloader):
			data = data.to(device)
			targets = targets.to(device)
			
			#forward pass
			spk_rec, _ = model(data)
			spike_count = spk_rec.sum(0) # batch x num_outputs
			_, max_spike = spike_count.max(1)

			#correct classes for one batcg
			num_correct = (max_spike == targets).sum()

			#total accuracy
			running_length += len(targets)
			running_accuracy += num_correct

		accuracy = (running_accuracy / running_length)

		return accuracy.item()

print(f"Test set accuracy: {measure_accuracy(net, test)}")
