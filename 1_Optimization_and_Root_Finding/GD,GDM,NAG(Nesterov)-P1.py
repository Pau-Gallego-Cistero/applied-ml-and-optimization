#GD,GDM, NAG algorithms 

#PART 1

#--------------------------------------------------------------------------------
#GD
#--------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


μ = 88.39
λ = 0.129
L_min = -200
L_max = 200
N = 1000
x = np.linspace(L_min, L_max, N) #Discretizar
dx = x[1] - x[0]
learn_rate=0.00001
phi_actual = 100
h = 1e-8 #Slope

def V(phi, μ, λ):
    return -(μ*μ)*(phi*phi)+ λ*(phi*phi*phi*phi)
    

history = []
for i in range(1000):
    # Cálculo de la derivada numérica
    dphi = (V(phi_actual + h, μ, λ) - V(phi_actual - h, μ, λ)) / (2 * h)
    phi = phi_actual - (learn_rate * dphi)
    phi_actual = phi #Update phi
    history.append(phi_actual) #Keep track  
    if abs(phi_actual - 174.2) < 1: #abs per que surti 1 bé
        print(f"{i} Iteration is the one that reaches")
        break

v_calculado = phi_actual * np.sqrt(2)
masa_higgs = np.sqrt(2 * λ * v_calculado**2)

print(f"The minimum value is: {phi_actual:.2f} GeV")
print(f"The Higgs masss calculated is: {masa_higgs:.2f} GeV")

plt.figure(figsize=(10, 6))
plt.plot(history, label='$\phi$', color='blue', linewidth=2)

plt.title('GD algorithm')
plt.xlabel('Iteration number')
plt.ylabel('Phi value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#--------------------------------------------------------------------------------
#GDM
#--------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


μ = 88.39
λ = 0.129
γ = 0.9
L_min = -200
L_max = 200
N = 1000
x = np.linspace(L_min, L_max, N) #Discretizar
dx = x[1] - x[0]
learn_rate=0.00001 #η
phi_actual = 100
h = 1e-8 #Slope
v = 0

def V(phi, μ, λ):
    return -(μ*μ)*(phi*phi)+ λ*(phi*phi*phi*phi)
    
history = []
for i in range(1000):
    # Cálculo de la derivada numérica
    dphi = (V(phi_actual + h, μ, λ) - V(phi_actual - h, μ, λ)) / (2 * h)
    v = (γ*v) + (learn_rate *dphi) #Update and calc. v
    phi = phi_actual - v
    phi_actual = phi #Update phi
    history.append(phi_actual) #Keep track  
    if abs(phi_actual - 174.2) < 1: #abs per que surti 1 bé
        print(f"{i} Iteration is the one that reaches")
        break
    
v_calculado = phi_actual * np.sqrt(2)
masa_higgs = np.sqrt(2 * λ * v_calculado**2)

print(f"The minimum value is: {phi_actual:.2f} GeV")
print(f"The Higgs masss calculated is: {masa_higgs:.2f} GeV")


plt.figure(figsize=(10, 6))
plt.plot(history, label='$\phi$', color='blue', linewidth=2)

plt.title('GDM algorithm')
plt.xlabel('Iteration number')
plt.ylabel('Phi value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#--------------------------------------------------------------------------------
#NAG
#--------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

μ = 88.39
λ = 0.129
γ = 0.9
L_min = -200
L_max = 200
N = 1000
x = np.linspace(L_min, L_max, N) #Discretizar
dx = x[1] - x[0]
learn_rate = 0.00001
phi_actual = 100
h = 1e-8 #Slope
v = 0

def V(phi, μ, λ):
    return -(μ*μ)*(phi*phi)+ λ*(phi*phi*phi*phi)

history = []
for i in range(1000):
# Punto de anticipacion (NAG)
    phi_ahead = phi_actual - (γ * v)
    dphi_ahead = (V(phi_ahead + h, μ, λ) - V(phi_ahead - h, μ, λ)) / (2 * h)
    v = (γ*v) + (learn_rate * dphi_ahead)
    phi = phi_actual - v
    phi_actual = phi #Update phi
    history.append(phi_actual) #Keep track

v_calculado = phi_actual * np.sqrt(2)
masa_higgs = np.sqrt(2 * λ * v_calculado**2)

print(f"The minimum value is: {phi_actual:.2f} GeV")
print(f"The Higgs masss calculated is: {masa_higgs:.2f} GeV")

plt.figure(figsize=(10, 6))
plt.plot(history, label='$\phi$', color='blue', linewidth=2, marker='o')

plt.title('NAG algorithm')
plt.xlabel('Iteration number')
plt.ylabel('Phi value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# --------------------------------------------------------------------------------
# COMBINED PLOT
# --------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot each history array with a unique color and label
plt.plot(history_gd, label='Basic GD', color='blue', linewidth=2)
plt.plot(history_gdm, label='GDM', color='orange', linewidth=2)
plt.plot(history_nag, label='NAG', color='green', linewidth=2)

plt.title('Comparison of GD, GDM, and NAG algorithms (1D Higgs SM)')
plt.xlabel('Iteration number')
plt.ylabel('Phi value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()