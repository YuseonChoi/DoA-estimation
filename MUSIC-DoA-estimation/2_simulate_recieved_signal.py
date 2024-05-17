import numpy as np
import matplotlib.pyplot as plt
from cmath import e, pi, sin, cos


N = 5      # Number of src
M = 10     # Number of mic
p = 100    # Number of time samples in each src signal
fc = 1e6   # 1MHz
fs = 1e7   # 10MHz, sampling frequency


## source data ##
s = np.load('source_signal_data.npy')
print("Source Signal s : ", s.shape)

## storing DoAs in radians ##
doa = np.array([20, 50, 85, 110, 145]) * pi / 180
print("Original Directions of Arrival (degrees): \n", doa * 180 / pi)


c = 3e8    # Speed of sound
d = 150    # Distance between mic elements


## Steering Vector as a function of theta ##
def a(theta):
    a1 = np.exp(-1j * 2 * pi * fc * d * (np.cos(theta) / c) * np.arange(M))
    return a1.reshape((M, 1))

A = np.zeros((M, N), dtype=complex)
for i in range(N):
    A[:, i] = a(doa[i])[:, 0]
print("Steering Matrix A: ", A.shape)

## Generating Recieved Signal ##
noise = np.random.multivariate_normal(mean=np.zeros(M), cov=np.diag(np.ones(M)), size=p).T
X = (A @ s + noise)
print("Recieved Signal X: ", X.shape)

np.save('recieved_signal_data.npy', X)