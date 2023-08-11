import scipy
import numpy as np
from math import sqrt
from scipy import stats
from numpy import cov, linspace
from statistics import mean
from numpy.random import normal, exponential
import matplotlib.pyplot as plt
from random import random
%matplotlib inline


def lift(A, B):
    return mean(B) - mean(A)

def p_value(A, B):
    return stats.ttest_ind(A, B)[1]

def get_AB_samples(mu, sigma, treatment_lift, N):
    A = list(normal(loc=mu                 , scale=sigma, size=N))
    B = list(normal(loc=mu + treatment_lift, scale=sigma, size=N))
    return A, B

N = 1000
N_multiplier = 4
mu = 100
sigma = 10
treatment_lift = 2
num_simulations = 1000

print('Simulating %s A/B tests, true treatment lift is %d...' % (num_simulations, treatment_lift))

n1_lifts, n4_lifts = [], []
for i in range(num_simulations):
    print('%d/%d' % (i, num_simulations), end='\r')
    A, B = get_AB_samples(mu, sigma, treatment_lift, N)
    n1_lifts.append(lift(A, B))
    A, B = get_AB_samples(mu, sigma, treatment_lift, N_multiplier*N)
    n4_lifts.append(lift(A, B))

print('N samples  A/B testing, mean lift = %.2f, variance of lift = %.2f' % (mean(n1_lifts), cov(n1_lifts)))
print('4N samples A/B testing, mean lift = %.2f, variance of lift = %.2f' % (mean(n4_lifts), cov(n4_lifts)))
print('Raio of lift variance = %.2f (expected = %.2f)' % (cov(n4_lifts)/cov(n1_lifts), 1/N_multiplier))

bins = linspace(-2, 6, 100)
plt.figure(figsize=(14, 7))
plt.hist(n1_lifts, bins, alpha=0.5, label='N samples')
plt.hist(n4_lifts, bins, alpha=0.5, label=f'{N_multiplier}N samples')
plt.xlabel('lift')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.title('lift histogram')
plt.show()


