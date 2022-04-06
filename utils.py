import time
import os
import sys
import cloudpickle
from matplotlib import pyplot as plt

from math import sqrt
import numpy as np
import torch

def generate_points(random_state, **kwargs):
    np.random.seed(random_state)

    n = kwargs['n_samples']
    d = kwargs['n_features']
    k = kwargs['centers']
    (lo, hi) = kwargs['center_box']

    assert n % k == 0 #ensure that each cluster will have same number of points
    #assert lo > 0 #binomial and poisson have support >= 0, exponential has mean (center) > 0

    # if kwargs['data_dist'] == 'multinomial': #ensures n and p are valid for the binomial (n is approximate since it has to be integer)
    #     assert kwargs['desired_variance'] < lo

    if 'center_coordinates' in kwargs.keys():
        assert kwargs['center_coordinates'].shape == (k,d)
        #assert kwargs['center_coordinates'].shape[1] == d or kwargs['center_coordinates'].shape[1] == 1
        centers = kwargs['center_coordinates']

        # if kawargs['center_coordinates'].shape[1] != d: #when d > 1 and centers in R^1, perform d independent samples for the coordinates
        #     centers =
        # else:
        #     centers = kwargs['center_coordinates']
    else:
        centers = np.random.rand(k, d) * (hi - lo) + lo #generate centers uniformly from (lo, hi)^d

    #centers = np.sort(centers, axis=0)

    #generate points
    X = np.zeros([n, d], dtype=np.float64)
    y_true = np.zeros([n], dtype=np.int64)
    cluster_size = n // k

    for i in range(k):
        if kwargs['data_dist'] == 'gamma':
            X[i*cluster_size : (i+1)*cluster_size, :] = random_sample(dist_name=kwargs['data_dist'], center=centers[i], desired_variance=kwargs['desired_variance'], n=cluster_size, d=d, shape=kwargs['shape'])
        else:
            X[i*cluster_size : (i+1)*cluster_size, :] = random_sample(dist_name=kwargs['data_dist'], center=centers[i], desired_variance=kwargs['desired_variance'], n=cluster_size, d=d)

        y_true[i*cluster_size : (i+1)*cluster_size] = i

    X, y_true = parallel_shuffle(X, y_true) #randomly shuffle points and labels
    return X, y_true, centers


def random_sample(dist_name, center, desired_variance, n, d, shape=None):
    dist_dict = {
        'gaussian': lambda center: np.random.normal(loc=center, scale=sqrt(desired_variance), size=(n,d)),
        'multinomial': lambda center: np.random.binomial(n=50, p=center[0]/50, size=(n,d)), 
        'exponential': lambda center: np.random.exponential(scale=center, size=(n,d)),
        'poisson': lambda center: np.random.poisson(lam=center, size=(n,d)),
        'gamma': lambda center, shape: np.random.gamma(shape=shape, scale=center/shape, size=(n,d))
    }

    sample_func = dist_dict[dist_name]

    if shape:
        return sample_func(center, shape).astype(np.float64)
    else:
        return sample_func(center).astype(np.float64)















