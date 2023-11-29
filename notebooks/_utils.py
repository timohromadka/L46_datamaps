import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fitter import Fitter
from scipy import stats

#### SVD-based data augmentation

def compute_relative_Fnorm_of_SVD_noise(S):
	"""
	Define noise as the low-rank matrix associated with the smallest singular values.

	Returns a list where
	- X[i] = (A - A(rank i))_F / A_F (e.g., 100, 8, 7.5, 7.3, ...)
	"""
	# compute Frobenius norm of A
	frob_A = torch.sqrt(torch.sum(torch.square(S))).item()

	# compute Frobenous norm of noise (= smallest singular values)
	frob_residual = []
	for i in range(-1, len(S)-1):
		frob_residual.append(torch.sqrt(torch.sum(torch.square(S[i+1:]))).item() / frob_A * 100)

	return frob_residual


def compute_SVD_noise(X, cutoff_percentage):
	"""
	Compute the noise (based on SVD's smallest singular values) at `cutoff_percentage` relative Frobenius
	
	Arguments
	- X (np.ndarray) of size N x D
	- cutoff_percentage (float)

	Return
	- noise (tensor) of size N x D

	Return the noises for all datapoints.
	"""
	assert(type(X) == np.ndarray)

	X_tensor = torch.tensor(X, dtype=torch.float32)
	U, S, Vh = torch.linalg.svd(X_tensor, full_matrices=False)

	### compute noise
	noise_relative_Frob_norm = compute_relative_Fnorm_of_SVD_noise(S)
	index_singular_value_cutoff = np.argmax(np.array(noise_relative_Frob_norm) < cutoff_percentage)
	if index_singular_value_cutoff==0:
		index_singular_value_cutoff=-1

	S[:index_singular_value_cutoff] = 0 		# zero out all non-noise singular value noise
	noise = U @ torch.diag(S) @ Vh 

	return noise


def resample_and_compute_SVD_noise(X, ratio_sampling, cutoff_percentage, resampling_times=50):
	"""
	Resample `X` without replacement, each time keeping `ratio_sampling` of the samples.
	For each sampled X_sample, compute the SVD noise.

	Arguments
	- X (np.ndarray) of size N x D
	- ratio_sampling (float): ratio of samples selected to compute the noise
	- cutoff_percentage: used to determine how many singular values represent the noise
	- resampling_times (int): number of time to perform resampling

	Return
	- np.ndarray (resampling_times * X.shape[0] * ratio_sampling), where the index value
		attributes the noise to a specific patient
	"""

	noises = []
	for _ in range(resampling_times):
		### sample datapoints
		perm = np.random.permutation(np.arange(X.shape[0]))
		index_sampled = perm[:int(X.shape[0] * ratio_sampling)]
		X_sampled = X[index_sampled]

		noise = compute_SVD_noise(X_sampled, cutoff_percentage)

		# check the noise is removed correctly	
		# print(torch.linalg.norm(noise, 'fro') / torch.linalg.norm(X_sampled_tensor, 'fro'))
		
		noises.append(pd.DataFrame(noise.cpu().detach().numpy(), index=index_sampled))
	
	return pd.concat(noises, axis=0)


def plot_svd_global_noise_analysis(noise_samples, suptitle):
	"""
	Plot multiple statistics for a given noise samples
	
	Plot
	- distribution of means of noise per feature
	- distribution of standard deviations of noise per feature
	- histogram comparing number of times Normal vs Student t fitted better each noise distribution
	- the noise distribution of two randomly selected genes

	Argument
	- noise_samples: np.ndarray with noise samples
	"""
	fig, axis = plt.subplots(1, 5, squeeze=False,  figsize=(12,3))

	means, stds = [], []
	for col_id in range(noise_samples.shape[1]):
		means.append(np.mean(noise_samples[col_id]))
		stds.append(np.std(noise_samples[col_id]))
	

	axis[0][0].set_title("Distribution of MEANs")
	axis[0][0].hist(means, bins=100)
	
	axis[0][1].set_title("Distribution of STDs")
	axis[0][1].hist(stds, bins=100)


	### Comparison where Gaussian or Student t are best
	count_best_t, count_best_gaussian, count_best_cauchy = 0, 0, 0
	for col_id in np.random.permutation(range(noise_samples.shape[1])[:100]): # compute the Gaussian vs Normal fit on 100 random features
		f = Fitter(noise_samples[col_id], distributions=['norm', 't', 'cauchy'])
		f.fit()
		res = f.get_best()
		if 't' in res:
			count_best_t += 1
		elif 'norm' in res:
			count_best_gaussian += 1
		elif 'cauchy' in res:
			count_best_cauchy += 1
	axis[0][2].set_title("Best distribution fit")
	axis[0][2].bar(['Cauchy', 'T', 'Norm'], [count_best_cauchy, count_best_t, count_best_gaussian])


	### plot three random distributionm (alongside fitten Gaussian and Student t)
	indices = np.random.randint(0, noise_samples.shape[1], 2)
	for i in range(2):
		data = noise_samples[indices[i]]

		axis[0][3+i].hist(data, bins=100, density=True)

		### plot fitted Student t
		f = Fitter(data, distributions=['t'])
		f.fit()
		params = f.get_best()['t']
		print(params)
		loc, scale = params['loc'], params['scale']
		x = np.linspace(loc - 3*scale, loc + 3*scale, 100)
		axis[0][3+i].plot(x, stats.t.pdf(x, df=len(data)-1, loc=loc, scale=scale), label='Student t')

		### plot fitted Normal
		f = Fitter(data, distributions=['norm'])
		f.fit()
		params = f.get_best()['norm']
		loc, scale = params['loc'], params['scale']
		x = np.linspace(loc - 3*scale, loc + 3*scale, 100)
		axis[0][3+i].plot(x, stats.norm.pdf(x, loc, scale), label='Gaussian')

		### plot fitted Cauchy
		f = Fitter(data, distributions=['cauchy'])
		f.fit()
		params = f.get_best()['cauchy']
		loc, scale = params['loc'], params['scale']
		x = np.linspace(loc - 3*scale, loc + 3*scale, 100)
		axis[0][3+i].plot(x, stats.cauchy.pdf(x, loc=loc, scale=scale), label='Cauchy')

		axis[0][3+i].set_title('Noise distribution \n for random gene')

	axis[0][4].legend(loc='upper right')


	fig.suptitle(f"Analysis for the noise distributions for multiple genes \n{suptitle} ", y=1.15, fontsize=13)


class DictObj:
	"""
	Transform a dictionary into an object
	"""
	def __init__(self, in_dict:dict):
		assert isinstance(in_dict, dict)
		for key, val in in_dict.items():
			if isinstance(val, (list, tuple)):
				setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
			else:
				setattr(self, key, DictObj(val) if isinstance(val, dict) else val)



#### DietDNN investigate embeddings

def plot_svd_spectrum_embedding_range(S, gene_embedding, emb_sizes, suptitle, file_name=None):
	"""
	Plot a grid with
	- singular value spectrum
	- relative Frobenius norm of the noise
	- embeddings dsitribution for different embedding sizes
	
	Arguments
	- S: array of len=rank with singular values
	- gene_embedding (D x rank): each row is one gene embedding
	- emb_sizes (list of int): for each value plot the range of the embeddings of length ranges_to_show[i]
	"""
	fig, axis = plt.subplots(1, 2 + len(emb_sizes), squeeze=True, figsize=(15,4))
	
	# plot singular values
	axis[0].plot(S)
	axis[0].set_title("Singular values")
	axis[0].set_xlabel("Id singular value")
	axis[0].set_yscale('log')
	axis[0].set_ylim(bottom=1e0)
	
	# plot relative Frobenius norm of different rank approximation

	frob_residual = compute_relative_Fnorm_of_SVD_noise(S)	

	axis[1].plot(frob_residual)
	axis[1].set_title("Relative Frobenius norm of \n low-rank residual $||X - \hat{X}||_F / ||X||_F$")
	axis[1].set_xlabel("Rank of the approximation $rank(\hat{X})$")
	# axis[1].set_ylabel("$||X - \hat{X}||_F / ||X||_F$")

	for i, emd_size in enumerate(emb_sizes):
		axis[2+i].hist(gene_embedding[:, :emd_size].numpy().reshape(-1), bins=100)
		axis[2+i].get_yaxis().set_ticks([])
		axis[2+i].set_title(f"Range emb size {emd_size}")

	fig.suptitle(suptitle, y=1.08, fontsize=17)
	# fig.autoscale()

	if file_name:
		fig.savefig(f"/home/am2770/Github/cancer-low-data/plots/{file_name}", bbox_inches='tight')

	fig.show()


def make_SVD_embeddings_range_plots(X_raw, dataset_name):
	"""
	Plot SVD singular values, relative noise Frobenius norm, and embedding range
	for raw, min-max and z-score scaled data
	"""
	assert type(X_raw)==np.ndarray
	
	print(torch.tensor(X_raw, dtype=torch.float32).shape)

	### Raw data
	U, S, Vh = torch.linalg.svd(torch.tensor(X_raw, dtype=torch.float32), full_matrices=False) # Vh.T (4160 x rank) contains the gene embeddings on each row

	plot_svd_spectrum_embedding_range(S, Vh.T, [20, 50, 100], 
		f"SVD embeddings - {dataset_name} 200 raw data",
		f"SVD_embeddings_{dataset_name}_raw")
	
	### min-max scaling
	X_min_max = MinMaxScaler().fit_transform(X_raw)
	U, S, Vh = torch.linalg.svd(torch.tensor(X_min_max, dtype=torch.float32), full_matrices=False) # Vh.T (4160 x rank) contains the gene embeddings on each row

	plot_svd_spectrum_embedding_range(S, Vh.T, [20, 50, 100], 
		f"SVD embeddings - {dataset_name} 200 min-max scaling",
		f"SVD_embeddings_{dataset_name}_minmax")

	### standard scaling
	X_standard = StandardScaler().fit_transform(X_raw)
	U, S, Vh = torch.linalg.svd(torch.tensor(X_standard, dtype=torch.float32), full_matrices=False) # Vh.T (4160 x rank) contains the gene embeddings on each row

	plot_svd_spectrum_embedding_range(S, Vh.T, [20, 50, 100], 
		f"SVD embeddings - {dataset_name} 200 $\mu=0, \sigma=1$",
		f"SVD_embeddings_{dataset_name}_standard")


def make_subplots(functions, suptitle=None, figsize_col=4, figsize_row=4, xlabel=None, ylabel=None, **kwargs):
	"""
	Auxiliary function which creates a subplot grid and plots the functions in the grid

	functions: 2D list of functions to plot
	"""
	nrows = len(functions)
	ncols = len(functions[0])

	fig, axis = plt.subplots(nrows, ncols, figsize=(ncols * figsize_col, nrows * figsize_row), squeeze=False, **kwargs)

	for i in range(nrows):		
		for j in range(ncols):
			functions[i][j](axis[i][j])

	if suptitle is not None:
		fig.suptitle(suptitle)

	fig.supxlabel(xlabel, y=0.1)
	fig.supylabel(ylabel, x=0.04)

	fig.tight_layout()
	return fig, axis


def filter_df_on_conditions(df, conditions):
	"""
	Filter based on equality to a dictionary of conditions

	conditions: dictionary
	"""
	return df.loc[(df[list(conditions)] == pd.Series(conditions)).all(axis=1)].copy()