from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def stratified_split_dataset(X, Y, valid_size=0.1, test_size=0.2, random_state=42, standardize=True):
	"""
	Split a dataset. Return even the indices of samples, and masks for train, valid, and test.
	"""
	indices = np.arange(X.shape[0])

	X_train_valid, X_test, y_train_valid, y_test, indices_train_valid, indices_test \
		= train_test_split(X, Y, indices, test_size=test_size, random_state=random_state)

	X_train, X_valid, y_train, y_valid, indices_train, indices_valid \
		= train_test_split(X_train_valid, y_train_valid, indices_train_valid, test_size=valid_size, random_state=random_state)

	# create masks
	train_mask, val_mask, test_mask = np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0])
	train_mask[indices_train] = 1
	val_mask[indices_valid] = 1
	test_mask[indices_test] = 1

	if standardize:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_valid = scaler.transform(X_valid)
		X_test = scaler.transform(X_test)


	return X_train, y_train, indices_train, train_mask, \
		   X_valid, y_valid, indices_valid, val_mask, \
		   X_test, y_test, indices_test, test_mask


def compute_gene_graph_edges(X_standardized, gene_id, threshold):
	"""
	Create the edge list for a gene graph.

	Assumption: the data is standardized using z-score.

	- X is the data matrix
	- gene_id (int) is the index of the gene in the data matrix
	- threshold (float): multiple of the standard deviation.
			all nodes with an expression within the threshold will be connected
	"""
	edge_list = []
	for i in range(X_standardized.shape[0]):			# for each patient
		for j in range(X_standardized.shape[0]): 		# for each patient that could be connected to it
			if i!=j:
				if abs(X_standardized[i][gene_id] - X_standardized[j][gene_id]) < threshold:
					edge_list.append([i, j])
					edge_list.append([j, i])
					
	return np.array(edge_list).T


