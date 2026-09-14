
import numpy as np
import sympy as sp
import matplotlib as mp
import matplotlib.pyplot as plt


V0=3
a=3
h_bar= 1.0545718e-34 # J·s
# r = np.sqrt( (2*m_eff*V0*a**2) / h_bar**2 )
z0 = 10.0
r = z0

# Despejamos m_eff (m*):   
print('\nAPARTADO 2: Massa efectiva')
m_eff = (z0**2 * h_bar**2) / (2 * V0 * a**2)
print('La masa efecitva es: ', m_eff)
print('                                            ')

x = sp.symbols('x', real=True, positive=True) # define la variable simbólica x

#Paridad definida para las funciones de onda:

# Función Par
def f_par_sym(x, r):
    return x * sp.tan(x) - sp.sqrt(r**2 - x**2)

# Función Impar
# cot(x) = 1/tan(x)
def f_impar_sym(x, r):
    return x / sp.tan(x) + sp.sqrt(r**2 - x**2)


f_par = f_par_sym(x, r)
f_impar = f_impar_sym(x, r)
f_d_par = sp.diff(f_par, x)
f_d_impar = sp.diff(f_impar, x)


f = sp.lambdify(x, f_par)
f_d = sp.lambdify(x, f_d_par)
f_imp = sp.lambdify(x, f_impar)
f_d_imp = sp.lambdify(x, f_d_impar)


# Newton-Rapson PAR
N_max = 20
tol = 1.e-9
raices_totales = []

# Calculamos cuántos intervalos de pi/2 hay hasta z0
n_intervalos = int(np.ceil(z0 / (np.pi/2)))

def seed_par(inf, sup, N=1000):
    for _ in range(N):
        x_random = np.random.uniform(inf + 0.05, sup - 0.05)
        if np.abs(f(x_random)) < 5.0:
            return x_random
    return (inf + sup) / 2

# Recorremos solo los intervalos pares
for n in range(0, n_intervalos, 2):
    limite_inf = n * np.pi / 2
    limite_sup = min((n + 1) * np.pi / 2, z0)
    
    if limite_inf >= limite_sup: break
        
    x_aprox = [seed_par(limite_inf, limite_sup)]

    for k in range(1, N_max):
        df = f_d(x_aprox[k-1])
        if np.abs(df) < 1e-14:
            break
        x_new = x_aprox[k-1] - f(x_aprox[k-1]) / df
        x_aprox.append(x_new)
        if x_new >= z0:
            break

        if np.abs(x_new - x_aprox[k-1]) / np.abs(x_new) < tol:
            if limite_inf <= x_new <= limite_sup:
                raices_totales.append(x_new)
                print('Raíz PAR: ', x_aprox[-1])
            break

def seed_impar(inf, sup, N=1000):
    for _ in range(N):
        x_random = np.random.uniform(inf + 0.05, sup - 0.05)
        if np.abs(f_imp(x_random)) < 5.0:
            return x_random
    return (inf + sup) / 2

# Recorremos solo los intervalos impares
for n in range(1, n_intervalos, 2):
    limite_inf = n * np.pi / 2
    limite_sup = min((n + 1) * np.pi / 2, z0)
    
    if limite_inf >= limite_sup: break
        
    x_aprox_impar = [seed_impar(limite_inf, limite_sup)]
    
    for k in range(1, N_max):
        df = f_d_imp(x_aprox_impar[k-1])
        if np.abs(df) < 1e-14:
            break
        x_new = x_aprox_impar[k-1] - f_imp(x_aprox_impar[k-1]) / df
        x_aprox_impar.append(x_new)
        if x_new >= z0:
            break
        
        if np.abs(x_new - x_aprox_impar[k-1]) / np.abs(x_new) < tol:
            if limite_inf <= x_new <= limite_sup:
                raices_totales.append(x_new)
                print('Raíz IMPAR: ', x_aprox_impar[-1])
            break

# Niveles de E
print('\nAPARTADO 1: Niveles de energía dentro del pozo')
raices_totales.sort() # Ordenar las raices
for i, chi in enumerate(raices_totales):
    # E = V0 * (chi / z0)^2
    energia_ev = V0 * (chi / z0)**2 #Chi = raices
    print(f'Nivel {i+1}: E = {energia_ev:.4f} eV')


mp.__version__

xx = np.linspace(0.01, z0, 2000)

yy_tan  = np.tan(xx)
yy_cot  = -1/np.tan(xx)

yy_circ = np.zeros_like(xx)
mask = xx <= r
yy_circ[mask] = np.sqrt((r/xx[mask])**2 - 1)
yy_circ[~mask] = np.nan

fig = plt.figure(figsize = (10,8))
plt.ylim(0, 10)
plt.xlim(0, z0)

plt.plot(xx, yy_circ, c='r', lw=2, label='f(x) = $\\sqrt{(z_0/x)^2 - 1}$')
plt.plot(xx, yy_tan,  c='b', lw=1.5, label='g(x) = $\\tan(x)$')
plt.plot(xx, yy_cot,  c='g', lw=1.5, label='g(x) = $-1/\\tan(x)$')

# Dibujamos las intersecciones de la lista que juntó ambos bloques
for raiz in raices_totales:
    y_raiz = np.sqrt((r/raiz)**2 - 1)
    plt.plot(raiz, y_raiz, 'ko', markersize=8, zorder=5)

plt.xlabel('z (eV)', fontsize=12)
plt.ylabel('Functions', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.show()
