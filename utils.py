import time
import os
import sys
import cloudpickle
from matplotlib import pyplot as plt

from math import sqrt
import numpy as np

#center = mean, n = num points, d = dimension
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

'''
data_params =  {
                'n_samples': 99, #they did 100 in robust bregman clustering, but 99 is divisible by 3
                'n_features': 2,
                'center_box': (1, 40), #they did 10,20,40 in robust bregman clustering: https://arxiv.org/pdf/1812.04356.pdf
                'centers': 3,
                'center_coordinates': np.array([[10,10], [20,20], [40,40]]),
                'data_dist': 'gamma',
                'desired_variance': None, #isn't used with gamma
                'shape': 3.0,
            }
'''












