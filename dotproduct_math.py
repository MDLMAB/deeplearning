import numpy as np
import torch 
                                                # La definición matemática del DOT PRODUCT suele ser 
                                                # alpha = a . b = <a,b> = a bT = SUM{_i=1}{^n} (a_i b_i)
                                                # Para realizar el DOT PRODUCT necesitamos mismas dimensiones en los vectores
v = np.array([[1,2,3,4]])                       # Define vector1
w = np.array([[0,1,0,-1]])                      # Define vector2
wT = w.transpose()                              # Realiza traspuesta vector2
result = np.dot(v,wT)                           # dot(vector1, traspuesta del vector2) haría el cálculo deseado  
print(result), print(' ')                       # Importante, debe dar un escalar

                                                # El DOT PRODUCT refleja las características comunes entre dos vectores, matrices...