# Deeplearning models learn by examples. 
# Non - random sampling can introduce systematic biases in DL models.
# Non - representative sampling causes overfitting and limits generalizability. 

import numpy as np                                      # Import the libraries
import random
import matplotlib as plt
                                                        # create a list of numbers to compute the mean and variance of
x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
xlen = len(x)
popmean = np.mean(x)                                    # compute the population mean
sample = np.random.choice(x,size=5,replace=True)        # compute a sample mean
sampmean = np.mean(sample)

print(popmean)                                          # print them
print(sampmean)
                                                        # number of experiments to run
nExpers = 10000
sampleMeans = np.zeros(nExpers)
for i in range(nExpers):
    sample = np.random.choice(x,size=15,replace=True)   # draw a sample
    sampleMeans[i] = np.mean(sample)                    # compute its mean
                                                        # show the results as a histogram
plt.hist(sampleMeans,bins=40,density=True)
plt.plot([popmean,popmean],[0,.3],'m--')
plt.ylabel('Count')
plt.xlabel('Sample mean')
plt.show()