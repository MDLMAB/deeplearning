import random 
import numpy
import torch
                                                            # Importante: No todas las matrices pueden multiplicarse unas con otras.
                                                            # Es importante respetar las dimensiones [M x N] * [N * K] = [M x K]
# numpy
a =  numpy.random.randn(3,4)                                # Define matriz random de 3 filas x 4 columnas
b =  numpy.random.randn(4,5)                                # Define matriz random de 4 filas x 5 columnas
c =  numpy.random.randn(5,7)                                # Define matriz random de 3 filas x 7 columnas

b_x_c = b@c                                                 # Realiza la operación de multiplicar dos matrices. Método abreviado.
a_x_b = numpy.matmul(a,b)                                   # Realiza la operación de multiplicar dos matrices. Método completo.
print(numpy.round(a_x_b,3))                                 # Imprime el producto de la matriz a x b y muestra 3 dígitos 

# torch
d =  torch.randn(3,4)                                       # Define matriz random de 3 filas x 4 columnas
e =  torch.randn(4,5)                                       # Define matriz random de 4 filas x 5 columnas
f =  torch.randn(5,7)                                       # Define matriz random de 3 filas x 7 columnas

                                                            # Las matrices creadas con numpy se llevan bien con las creadas con torch
                                                            # Muchas veces es necesario convertir datos numpy a datos torch
d_x_e = d@e
print(numpy.round(d_x_e,2))                                 # Imprime el producto de la matriz d x e y muestra 2 dígitos 