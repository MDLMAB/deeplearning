import numpy as np                                              # The sofmax calculation returns math values 
import random                                                   # as probabilities sigma = {value1, value2, value3 ... , value n}
import torch                                                    # the summatory of each value1 + value2 + ... + valuen  = 1
from torch import nn as nn
from matplotlib import pyplot as plt

# numpy
                                                                # Define de 'z' function manually
z = [1,2,3]                                                     # First, define a list of values for z

num = np.exp(z)                                                 # Evaluate the numerator formula
den = np.sum( num )                                             # Evaluate the denominator formula
sigma = num / den                                               # Then evaluate sigma
print(sigma), print(' ')
print(np.sum(sigma))                                            # This calculation must be 1 


# Repeating with random integers
zz = np.random.randint(-5, high = 15, size =25)                 # First point = -5 / Last Point = 15 / Maximum elements = 25
print(zz)
num_zz = np.exp(zz) 
den_zz = np.sum( num_zz )
sigma_zz = num_zz / den_zz
print(sigma_zz), print(' ')
print(np.sum(sigma_zz))

# Compare  
plt.plot(zz, sigma_zz, 'ko')
plt.xlabel(' Original number (z) ')
plt.ylabel(' Softmax $\sigma$ ')
plt.yscale('log')                                               # Change the yscale for see the linear transformation in log space
plt.title(' $\sum\sigma$ = %g ' %np.sum(sigma_zz))
plt.show()


# torch
softfun = nn.Softmax(dim = 0)                                   # Instance of the Softmax class
sigmaT = softfun(torch.Tensor(z))                               # Converting the list to tensor data type
print(sigmaT)
plt.plot(sigma, sigmaT, 'ko')
plt.xlabel(' Manual Softmax ')
plt.ylabel(' Torch nn.Softmax ')
plt.yscale('log') 
r = np.corrcoef(sigma,sigmaT)[0,1]
plt.title(f' The two methods correlate at r = {r} ')
plt.show()