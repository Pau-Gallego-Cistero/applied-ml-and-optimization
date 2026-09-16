import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#----------------------------------------------------
# OSL
#----------------------------------------------------


g = -9.81
sigma = 1
N = 100

t = np.linspace(0, 10, 100)
noise = np.random.normal(0, sigma, N)
x = t*t
X = x.reshape(-1, 1) #Use .reshape(-1, 1) to turn the flat list into a 2D Matrix column

def y(g, x, noise):
    return (0.5 * g * x) + noise

y_calculada = y(g, x, noise)

Y = y_calculada.reshape(-1, 1)

w = np.linalg.inv(X.T @ X)
w_ols = w @ X.T @ Y  # We use '@' for matrix multiplicatio
# Multiply inputs (X) by the trained slope (w_ols) to get 100 predicted points
Y_pred = X @ w_ols

def calculate_r_squared(y_true, y_pred):
    mean_y_true = sum(y_true) / len(y_true)
    
    ss_tot = 0
    for y in y_true:
        ss_tot += (y - mean_y_true) ** 2
        
    ss_res = 0
    for actual, predicted in zip(y_true, y_pred):
        # zip() takes two or more lists and pairs up their elements based on their position
        ss_res += (actual - predicted) ** 2
        
    if ss_tot == 0:
        return 0.0 
        
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

r2_score = calculate_r_squared(Y.flatten(), Y_pred.flatten())


print("Calculated slope (w_ols):", w_ols)
print(f"R-squared: {r2_score:.4f}")


plt.figure(figsize=(8, 6))
plt.scatter(x, Y, color='blue', label='Actual Data (with noise)', alpha=0.5)
plt.plot(x, Y_pred, color='red', linewidth=2, label=f'OLS Fit (R²={r2_score:.4f})')
plt.title('OLS Fit')
plt.xlabel('Time squared (t^2)')
plt.ylabel('Distance (y)')
plt.legend()
plt.grid(True)

plt.show()



#----------------------------------------------------
# Ridge
#----------------------------------------------------

g = -9.81
sigma = 1
N = 100
λ = 1

I = np.identity(1)

t = np.linspace(0, 10, 100)
noise = np.random.normal(0, sigma, N)
x = t*t
X = x.reshape(-1, 1) #Use .reshape(-1, 1) to turn the flat list into a 2D Matrix column

def y(g, x, noise):
    return (0.5 * g * x) + noise

y_calculada = y(g, x, noise)

Y = y_calculada.reshape(-1, 1)

w = np.linalg.inv(X.T @ X + λ * I)
w_ridge = w @ X.T @ Y  # We use '@' for matrix multiplicatio
# Multiply inputs (X) by the trained slope (w_ols) to get 100 predicted points
Y_pred = X @ w_ridge

def calculate_r_squared(y_true, y_pred):
    mean_y_true = sum(y_true) / len(y_true)
    
    ss_tot = 0
    for y in y_true:
        ss_tot += (y - mean_y_true) ** 2
        
    ss_res = 0
    for actual, predicted in zip(y_true, y_pred):
        # zip() takes two or more lists and pairs up their elements based on their position
        ss_res += (actual - predicted) ** 2
        
    if ss_tot == 0:
        return 0.0 
        
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

r2_score = calculate_r_squared(Y.flatten(), Y_pred.flatten())


print("Calculated slope (w_ridge):", w_ridge)
print(f"R-squared: {r2_score:.4f}")


plt.figure(figsize=(8, 6))
plt.scatter(x, Y, color='blue', label='Actual Data (with noise)', alpha=0.5)
plt.plot(x, Y_pred, color='red', linewidth=2, label=f'Ridge Fit (R²={r2_score:.4f})')
plt.title('Ridge Fit')
plt.xlabel('Time squared (t^2)')
plt.ylabel('Distance (y)')
plt.legend()
plt.grid(True)

plt.show()



#----------------------------------------------------
# LASSO
#----------------------------------------------------


g = -9.81
sigma = 20
N = 100
alpha = 10

t = np.linspace(0, 10, N)
noise = np.random.normal(0, sigma, N)

x = t**2
X = x.reshape(-1, 1)
Y = (0.5 * g * x + noise).reshape(-1, 1)

modelo = Lasso(alpha)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

Y_pred_test = modelo.predict(X_test) 
# Evaluamos la calidad del modelo con los datos de TEST
Y_pred_plot = modelo.predict(X)
# predecimos sobre toda la columna X original.

w_LASSO = modelo.coef_[0] #guardamos la pendiente

# la función oficial para R-cuadrado:
R2 = r2_score(y_test, Y_pred_test)

print(f"Pendiente (w) encontrada por LASSO: {w_LASSO:.4f}")
print(f"R-squared en el set de TEST: {R2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(x, Y, color='blue', label='Datos reales (Ruido)', alpha=0.5)
plt.plot(x, Y_pred_plot, color='red', linewidth=2, label=f'LASSO (R²={R2:.4f})')

plt.title(f'LASSO Regression (λ={alpha})')
plt.xlabel('t^2')
plt.ylabel('Distancia')
plt.legend()
plt.grid(True)
plt.show()


