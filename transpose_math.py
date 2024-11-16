import numpy as np
import torch
                                                                    # En este archivo .py se aprendizaje para conseguir una matriz transpuesta
# numpy
nv = np.array([ [1,2,3,4],
                [4,5,6,7] ])                                        # Organiza bien la estructura para visualización clara                                           
print(nv), print(' ')                                               # Imprime matrix 2x4                                           
nvT = nv.transpose()                                                # Con el método transpose() de numpy conseguimos la traspuesta                                          
print(nvT), print(' ')                                              # Imprimimos la traspuesta 4x2 de la matriz nv

print(f" El tipo de variable nv usando numpy  es {type(nv)}")
print(f" El tipo de variable nvT usando numpy  es {type(nvT)}")

# torch
mv = torch.tensor([ [1,2,3,4] ])                                    # Define vector
print(mv), print(' ')                                               # Imprime vector
mvT = mv.T                                                          # Traspuesta vector
print(mvT), print(' ')                                              # Imprime traspuesta vector

print(f" El tipo de variable mv usando torch  es {type(mv)}")
print(f" El tipo de variable mvT usando torch  es {type(mvT)}")
