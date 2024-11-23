import numpy as np                              # import libraries
x = [1,2,4,6,5,4,0]                             # create a list of numbers to compute the mean and variance of
xlen = len(x)
                                                # compute the mean
mean_1 = np.mean(x)
mean_2 = np.sum(x) / xlen
print(mean_1)                                   # print them both
print(mean_2)
                                                # variance
var_1 = np.var(x, ddof=0)                       # Degrees of freedom default equals 0
var_2 = (1/(xlen-1)) * np.sum( (x-mean_1)**2 )  # unbiased - diference is key in small dataset
# var3 = (1/(xlen)) * np.sum( (x-mean_1)**2 )   # biased 
print(var_1)
print(var_2)
                                                # Degrees of freedom equals 1
var_3 = np.var(x,ddof=1)
print(var_3)
print(var_2)
                                                # does it matter for large N? 
N = 10000               
y = np.random.randint(0,high=20,size=N)

var0 = np.var(y,ddof=0)                         # default
var1 = np.var(y,ddof=1)                         # unbiased

print(var0)                                     # print them both
print(var1)