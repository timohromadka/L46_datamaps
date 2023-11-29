from turtle import forward
from regex import P
import torch
import torch.nn as nn


class LearnableSparsityVector(nn.Module):
	"""
	Learn vector v such which will be multiplied with the first layer W to obtian W_sparse = W * sigmoid(v)
	"""
	def __init__(self, args) -> None:
		super().__init__()
		self.weights = nn.Parameter(torch.ones(args.num_features))
		self.sigmoid = nn.Sigmoid()

	def forward(self):
		return self.sigmoid(self.weights)


class SparsityNetwork(nn.Module):
	"""
	Sparsity network

	Architecture
	- same 4 hidden layers of 100 neurons as the DietNetwork (for simplicity)
	- output layer: 1 neuron, sigmoid activation function
	- note: the gating network in LSNN used 3 hidden layers of 100 neurons

	Input
	- gene_embedding: gene embedding (batch_size, embedding_size)
	Output
	- sigmoid value (which will get multiplied by the weights associated with the gene)
	"""
	def __init__(self, args, embedding_matrix):
		"""
		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		
		print(f"Initializing SparsityNetwork with embedding_matrix of size {embedding_matrix.size()}")
		
		self.args = args
		self.register_buffer('embedding_matrix', embedding_matrix) # store the static embedding_matrix

		layers = []
		if args.sparsity_type == 'local':						# input for instance-wise sparsity: gene embedding + input of D dimensions
			dim_prev = args.sparsity_gene_embedding_size + args.num_features
		else:													# input for global sparsity: gene embedding
			dim_prev = args.sparsity_gene_embedding_size

		for _, dim in enumerate(args.diet_network_dims):
			layers.append(nn.Linear(dim_prev, dim))
			layers.append(nn.LeakyReLU())
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.Dropout(args.dropout_rate))

			dim_prev = dim
		
		layers.append(nn.Linear(dim, 1))
		self.network = nn.Sequential(*layers)

		if args.mixing_layer_size:
			mixing_layers = []

			layer1 = nn.Linear(args.num_features, args.mixing_layer_size, bias=False)
			nn.init.uniform_(layer1.weight, -0.005, 0.005)
			mixing_layers.append(layer1)

			mixing_layers.append(nn.LeakyReLU())

			if args.mixing_layer_dropout:
				mixing_layers.append(nn.Dropout(args.mixing_layer_dropout))
			
			layer2 = nn.Linear(args.mixing_layer_size, args.num_features, bias=False)
			nn.init.uniform_(layer2.weight, -0.005, 0.005)
			mixing_layers.append(layer2)

			self.mixing_layers = nn.Sequential(*mixing_layers)
		else:
			self.mixing_layers = None

	def forward(self, input):
		"""
		Input:
		- input: Tensor of patients (B, D)

		Returns:
		if args.sparsity_type == 'global':
			- Tensor of sigmoid values (D)
		if args.sparsity_type == 'local':
			- Tensor of sigmoid values (B, D)
		"""
		if self.args.sparsity_type == 'global':
			out = self.network(self.embedding_matrix) # (D, 1)]

			if self.mixing_layers:
				out = self.mixing_layers(out.T).T # input of size (1, D) to the linear layer

			out = torch.sigmoid(out)
			return torch.squeeze(out, dim=1) 		  # (D)

		elif self.args.sparsity_type == 'local':
			B = input.shape[0]
			D = self.args.num_features
			K = self.args.sparsity_gene_embedding_size

			# duplicate embedding matrix B times ( (D, K) -> (B, D, K))
			embedding_matrix_expanded = self.embedding_matrix.unsqueeze(dim=0).expand(B, D, K)

			# duplicate the input D times ( (B, D) -> (B, D, D) )
			input_expanded = input.unsqueeze(dim=2).expand(B, D, D)

			# concatenate the two ( (B, D, K) + (B, D, D) -> (B, D, K + D) )
			concatenated = torch.concat([embedding_matrix_expanded, input_expanded], 2)
			
			# reshape to ( (B, D, K + D) -> (BxD, K + D))
			concatenated = torch.reshape(concatenated, (B * D, K + D))

			weights = self.network(concatenated) 	# (BxD, 1)
			return torch.reshape(weights, (B, D)) 	# (B, D)