# Exercise:
# Step 1: Repeat 1D gradient descent to find minimum of the following function.
import numpy as np
import sympy as sym
from IPython.display import display
import matplotlib.pyplot as plt
import sympy.plotting.plot as symplot
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') 



# Defino las funciones dx y df puramente simbolicas
def fx_sim():
    x = sym.symbols('x')
    return sym.cos(2*sym.pi*x) +x**2

def df_sim():
    x = sym.symbols('x')
    f = fx_sim()
    df = sym.diff(f,x)
    return  df                          
# Muestro las funciones simbolicas en terminal
display(fx_sim())
display(df_sim())

# Convierto las funciones simbolicas a numericas
x_num= np.linspace(-2,2,2001)

def fx_num():
    x = sym.symbols('x')                # Variable simbólica
    fx_simb = fx_sim()                  # Función simbólica
    return sym.lambdify(x, fx_simb, 'numpy')

def df_num():
    x = sym.symbols('x')                # Variable simbólica
    df_simb = df_sim()                  # Derivada simbólica
    return sym.lambdify(x, df_simb, 'numpy') 

fx_values = fx_num()(x_num)  # Función numérica evaluada en x_num
df_values = df_num()(x_num)  # Derivada numérica evaluada en x_num

plt.figure(figsize=(8, 5))
plt.plot(x_num, fx_values, label='$f(x)$')
plt.plot(x_num, df_values, label="$f'(x)$", linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend(['fx', 'dx'])
plt.title('Función f y su derivada df')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(alpha=0.3)
plt.show()

# Definimos el Local Minimun
localmin = np.random.choice(x_num,1) # np.array([0])
print(localmin), print(' ')
learning_rate = .01
training_epochs = 100

for i in range(training_epochs):
    grad = df_num()(localmin)
    localmin = localmin - learning_rate*grad

localmin

plt.plot(x_num,fx_values, x_num,df_values)
plt.plot(localmin,df_num()(localmin),'go')
plt.plot(localmin,fx_num()(localmin),'bo')

plt.xlim(x_num[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)','df','f(x) min'])
plt.title('Empirical local minimum: %s'%localmin[0])
plt.show()