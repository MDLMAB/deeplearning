# Gradient descent implemented in python. 
# Parameters learning rate and training epochs are implemented. 
#   Goal: 
#   appreciate why gradient descent is not guaranteed to give the correct answer.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

                                        # function (as a function)
def fx(x):
    return 3*x**2 - 3*x + 4

                                        # derivative function
def deriv(x):
    return 6*x - 3

                                        # plot the function and its derivative
                                        # define a range for x
x = np.linspace(-2,2,2001)
                                        # plotting
plt.plot(x,fx(x), x,deriv(x))
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['y','dy'])
plt.show()
                                        # Learning algorithm
                                        # random starting point
localmin = np.random.choice(x,1)
print(localmin)

                                        # learning parameters
learning_rate = .01                     # This param can be tune down or tune up
training_epochs = 100
                                        # run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate*grad

localmin

                                        # plot the results
plt.plot(x,fx(x), x,deriv(x))
plt.plot(localmin,deriv(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')

plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)','df','f(x) min'])
plt.title('Empirical local minimum: %s'%localmin[0])
plt.show()

                                        # random starting point
localmin = np.random.choice(x,1)
                                        # learning parameters
learning_rate = .01
training_epochs = 100
                                        # run through training and store all the results
modelparams = np.zeros((training_epochs,2))
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate*grad
    modelparams[i,0] = localmin         # Warning: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
    modelparams[i,1] = grad             # Warning: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
                                        # plot the gradient over iterations
fig,ax = plt.subplots(1,2,figsize=(12,4))

for i in range(2):
    ax[i].plot(modelparams[:,i],'o-')
    ax[i].set_xlabel('Iteration')
    ax[i].set_title(f'Final estimated minimum: {localmin[0]:.5f}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Derivative')          # the goal is to achive the result approch zero.

plt.show()

# Most often in DL, the model trains for a set number of iterations, which is what we do here. But there are other ways
# of defining how long the training lasts. Modify the code so that training ends when the derivative is smaller than 
# some threshold, e.g., 0.1. Make sure your code is robust for negative derivatives.
# 
# Does this change to the code produce a more accurate result? What if you change the stopping threshold?
# 
# Can you think of any potential problems that might arise when the stopping criterion is based on the derivative 
# instead of a specified number of training epochs?