# How DL models learn:
# 1. Guess a solution
# 2. Compute the error. 
# 3. Learn from mistakes and modify the parameters.

# To do that we need a mathematical description of the error "landscape" of the problem
# and we need a way to find the minimun of the landscape.

# The core idea of gradient descent: To get the smallest errors in our system. 

# A numerical example:

                                            # import libraries
import numpy as np
import sympy as sym
                                            # make the equations look prettier
from IPython.display import display
import matplotlib.pyplot as plt
import sympy.plotting.plot as symplot
                                            # create symbolic variables in sympy
x = sym.symbols('x')
fx = 3*x**2 -3*x +4
                                            # compute their individual derivatives
df = sym.diff(fx)

# print everything
print('The functions:')
display(fx)
print(' ')

print('Their derivatives:')
display(df)
print(' ')

# plot them
p = symplot(fx, (x, -3, 3), label='The function', show=False, line_color='magenta')  
p.extend(symplot(df, (x, -3, 3), label='The derivative', show=False, line_color='blue')) 
p.title = 'The function and its derivative'
p.xlabel = 'x'
p.ylabel = 'y'
p.legend = True
p.show()

# Now, for finding out the minimum point we must use the Gradient descent algorithm
# 1. Initizalize random guess of minimum
# 2. Loop over training iterations (for loop)
#   2.1 Compute derivative at guess min
#   2.2 Updated guess min is itself minus derivative scaled by learning rate