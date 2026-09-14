#GD,GDM, NAG algorithms 

#PART 2


#--------------------------------------------------------------------------------
#GD - P2
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
learn_rate=0.00001 # η
phi_actual = 100
h = 1e-8 #Slope
α1 = 2.5e4
α2 = 2.0e4
β1 = 1
β2 = 1
β3 = 0.5

def V(p1, p2, α1, α2, β1, β2, β3):
    return -α1*(p1**2) - α2*(p2**2) + (β1/2)*(p1**4) + (β2/2)*(p2**4) + β3*(p1**2)*(p2**2)
    
phi_actual1 = 10.0 # Valor inicial sugerido
phi_actual2 = 10.0 # Valor inicial sugerido

history1 = []
history2 = []

for i in range(1000):
    # Cálculo de la derivada numérica
    # DPhi 1 (Phi 2 cte)
    dphi1 = (V(phi_actual1 + h, phi_actual2, α1, α2, β1, β2, β3) - V(phi_actual1 - h, phi_actual2, α1, α2, β1, β2, β3)) / (2 * h)
    # DPhi 2 (Phi 1 cte)
    dphi2 = (V(phi_actual1, phi_actual2 + h, α1, α2, β1, β2, β3) - V(phi_actual1, phi_actual2 - h, α1, α2, β1, β2, β3)) / (2 * h)
    
    phi_actual1 = phi_actual1 - (learn_rate * dphi1)
    phi_actual2 = phi_actual2 - (learn_rate * dphi2)
    
    #Keep track  
    history1.append(phi_actual1)
    history2.append(phi_actual2)


suma = β1 * phi_actual1**2 + β2 * phi_actual2**2
vabs = abs(β1 * phi_actual1**2 - β2 * phi_actual2**2)

m_H0 = np.sqrt(suma + vabs)
m_h0 = np.sqrt(suma - vabs)
m_charged = np.sqrt(β3 * (phi_actual1**2 + phi_actual2**2))


print(f"Minimum found: v1 = {phi_actual1:.2f} GeV, v2 = {phi_actual2:.2f} GeV")
print(f"Mass H0: {m_H0:.2f} GeV")
print(f"Mass h0: {m_h0:.2f} GeV")
print(f"Mass H+/-: {m_charged:.2f} GeV")

plt.figure(figsize=(10, 6))
plt.plot(history1, label='$\Phi_1$')
plt.plot(history2, label='$\Phi_2$')
plt.title('GD algorithm - Pt2')
plt.xscale('log')
plt.xlabel('Iteration number')
plt.ylabel('Field Value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



#--------------------------------------------------------------------------------
#GDM - P2
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
v1 = 0
v2 = 0
α1 = 2.5e4
α2 = 2.0e4
β1 = 1
β2 = 1
β3 = 0.5  

def V(p1, p2, α1, α2, β1, β2, β3):
    return -α1*(p1**2) - α2*(p2**2) + (β1/2)*(p1**4) + (β2/2)*(p2**4) + β3*(p1**2)*(p2**2)
    
phi_actual1 = 10.0 # Valor inicial sugerido
phi_actual2 = 10.0 # Valor inicial sugerido

history1 = []
history2 = []

for i in range(1000):
    # Cálculo de la derivada numérica
    # DPhi 1 (Phi 2 cte)
    dphi1 = (V(phi_actual1 + h, phi_actual2, α1, α2, β1, β2, β3) - V(phi_actual1 - h, phi_actual2, α1, α2, β1, β2, β3)) / (2 * h)
    # DPhi 2 (Phi 1 cte)
    dphi2 = (V(phi_actual1, phi_actual2 + h, α1, α2, β1, β2, β3) - V(phi_actual1, phi_actual2 - h, α1, α2, β1, β2, β3)) / (2 * h)
    
    v1 = (γ*v1) + (learn_rate *dphi1)
    v2 = (γ*v2) + (learn_rate *dphi2)
    
    phi_actual1 = phi_actual1 - v1
    phi_actual2 = phi_actual2 - v2
    
    #Keep track  
    history1.append(phi_actual1)
    history2.append(phi_actual2)


suma = β1 * phi_actual1**2 + β2 * phi_actual2**2
vabs = abs(β1 * phi_actual1**2 - β2 * phi_actual2**2)

m_H0 = np.sqrt(suma + vabs)
m_h0 = np.sqrt(suma - vabs)
m_charged = np.sqrt(β3 * (phi_actual1**2 + phi_actual2**2))


print(f"Minimum found: v1 = {phi_actual1:.2f} GeV, v2 = {phi_actual2:.2f} GeV")
print(f"Mass H0: {m_H0:.2f} GeV")
print(f"Mass h0: {m_h0:.2f} GeV")
print(f"Mass H+/-: {m_charged:.2f} GeV")

plt.figure(figsize=(10, 6))
plt.plot(history1, label='$\Phi_1$')
plt.plot(history2, label='$\Phi_2$')
plt.title('GDM algorithm - Pt2')
plt.xlabel('Iteration number')
plt.ylabel('Field Value (GeV)')
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
learn_rate=0.00001 #η
phi_actual = 100
h = 1e-8 #Slope
v1 = 0
v2 = 0
α1 = 2.5e4
α2 = 2.0e4
β1 = 1
β2 = 1
β3 = 0.5  

def V(p1, p2, α1, α2, β1, β2, β3):
    return -α1*(p1**2) - α2*(p2**2) + (β1/2)*(p1**4) + (β2/2)*(p2**4) + β3*(p1**2)*(p2**2)
    
phi_actual1 = 10.0 # Valor inicial sugerido
phi_actual2 = 10.0 # Valor inicial sugerido

history1 = []
history2 = []

for i in range(1000):
    # ANTICIPACION (Primero esto): Calculamos donde nos llevaria la inercia
    phi1_ahead = phi_actual1 - (γ * v1)
    phi2_ahead = phi_actual2 - (γ * v2)

    # DERIVADA FUTURA (Usamos los "ahead" aqui):
    # DPhi 1 (Phi 2 cte) -> Evaluamos en phi1_ahead
    dphi1 = (V(phi1_ahead + h, phi2_ahead, α1, α2, β1, β2, β3) - V(phi1_ahead - h, phi2_ahead, α1, α2, β1, β2, β3)) / (2 * h)
    # DPhi 2 (Phi 1 cte) -> Evaluamos en phi2_ahead
    dphi2 = (V(phi1_ahead, phi2_ahead + h, α1, α2, β1, β2, β3) - V(phi1_ahead, phi2_ahead - h, α1, α2, β1, β2, β3)) / (2 * h)
    
    # ACTUALIZAR VELOCIDAD
    v1 = (γ*v1) + (learn_rate * dphi1)
    v2 = (γ*v2) + (learn_rate * dphi2)
    
    # ACTUALIZAR POSICION REAL
    phi_actual1 = phi_actual1 - v1
    phi_actual2 = phi_actual2 - v2
    
    # Guardar historial
    history1.append(phi_actual1)
    history2.append(phi_actual2)


suma = β1 * phi_actual1**2 + β2 * phi_actual2**2
vabs = abs(β1 * phi_actual1**2 - β2 * phi_actual2**2)

m_H0 = np.sqrt(suma + vabs)
m_h0 = np.sqrt(suma - vabs)
m_charged = np.sqrt(β3 * (phi_actual1**2 + phi_actual2**2))


print(f"Minimum found: v1 = {phi_actual1:.2f} GeV, v2 = {phi_actual2:.2f} GeV")
print(f"Mass H0: {m_H0:.2f} GeV")
print(f"Mass h0: {m_h0:.2f} GeV")
print(f"Mass H+/-: {m_charged:.2f} GeV")

plt.figure(figsize=(10, 6))
plt.plot(history1, label='$\Phi_1$')
plt.plot(history2, label='$\Phi_2$')
plt.title('NAG algorithm - Pt2')
plt.xscale('log')
plt.xlabel('Iteration number')
plt.ylabel('Field Value (GeV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
