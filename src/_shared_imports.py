import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import random
import math
import re
import datetime
from joblib import dump, load
from typing import DefaultDict
from tqdm.auto import tqdm
import argparse
from ast import literal_eval, parse

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import gpytorch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.utils import shuffle

from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def pearson_distance(x, y):
	return 1 - pearsonr(x, y)[0]

def spearman_distance(x, y):
	return 1 - spearmanr(x, y)[0]

def cosine_similarity(x, y):
	return 1 - cosine_distance(x, y)

def get_PCA_projected_data(X, y, n_components, test_size, split_seed, PCA_seed, whiten = False):
    """
    Run PCA of a split of size (1-`test_size`) of X

    Args:
    - X: data
    - n_components (int): number of PCA components
    - train_test_seed (int): seed for spliting into train/test

    Returns:
    - X_train_pca, X_test_pca: PCA reduced data
    - explained_variance_ratio (float)
    """
    scaler = StandardScaler()
    
    X_train, X_test, _, _ = train_test_split(X, y, stratify=y, test_size=test_size, random_state=split_seed)


    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components = n_components, whiten=whiten, random_state=PCA_seed)
    X_train_pca = pca.fit_transform(X_train)    
    X_test_pca = pca.transform(X_test)

    explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))
    
    return X_train_pca, X_test_pca, explained_variance_ratio