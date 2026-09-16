import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


#----------------------------------------------------
# OLS SKLEARN
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
Y = (0.5 * g * x + noise).reshape(-1, 1)


modelo = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

Y_pred_test = modelo.predict(X_test) 
# Evaluamos la calidad del modelo con los datos de TEST
Y_pred_plot = modelo.predict(X)
# predecimos sobre toda la columna X original.

w_OLS = modelo.coef_[0] #guardamos la pendiente

# la función oficial para R-cuadrado:
R2 = r2_score(y_test, Y_pred_test)

print("Calculated slope (w_OLS):", w_OLS)
print(f"R-squared en el set de TEST: {R2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(x, Y, color='blue', label='Datos reales (Ruido)', alpha=0.5)
plt.plot(x, Y_pred_plot, color='red', linewidth=2, label=f'OLS (R²={R2:.4f})')

plt.title('OLS Regression')
plt.xlabel('t^2')
plt.ylabel('Distancia')
plt.legend()
plt.grid(True)
plt.show()




#----------------------------------------------------
# Ridge SKLEARN
#----------------------------------------------------

g = -9.81
sigma = 100
N = 100
alpha = 10

t = np.linspace(0, 10, 100)
noise = np.random.normal(0, sigma, N)
x = t*t
X = x.reshape(-1, 1) #Use .reshape(-1, 1) to turn the flat list into a 2D Matrix column

def y(g, x, noise):
    return (0.5 * g * x) + noise
Y = (0.5 * g * x + noise).reshape(-1, 1)


modelo = Ridge(alpha)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

Y_pred_test = modelo.predict(X_test) 
# Evaluamos la calidad del modelo con los datos de TEST
Y_pred_plot = modelo.predict(X)
# predecimos sobre toda la columna X original.

w_Ridge = modelo.coef_[0] #guardamos la pendiente

# la función oficial para R-cuadrado:
R2 = r2_score(y_test, Y_pred_test)

print(f"Pendiente (w) encontrada por Ridge: {w_Ridge:.4f}")
print(f"R-squared en el set de TEST: {R2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(x, Y, color='blue', label='Datos reales (Ruido)', alpha=0.5)
plt.plot(x, Y_pred_plot, color='red', linewidth=2, label=f'Ridge (R²={R2:.4f})')

plt.title(f'Ridge Regression (λ={alpha})')
plt.xlabel('t^2')
plt.ylabel('Distancia')
plt.legend()
plt.grid(True)
plt.show()



#----------------------------------------------------
# MIX
#----------------------------------------------------

g = -9.81
sigma = 10
N = 100
alpha = 20

t = np.linspace(0, 10, 100)
noise = np.random.normal(0, sigma, N)
x = t*t
X = x.reshape(-1, 1) #Use .reshape(-1, 1) to turn the flat list into a 2D Matrix column


def y(g, x, noise):
    return (0.5 * g * x) + noise
Y = (0.5 * g * x + noise).reshape(-1, 1)


OLS = linear_model.LinearRegression()
Ridge = linear_model.Ridge(alpha)
LASSO = linear_model.Lasso(alpha)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
OLS.fit(X_train, y_train)
Ridge.fit(X_train, y_train)
LASSO.fit(X_train, y_train)

Y_OLS_test = OLS.predict(X_test)
Y_Ridge_test = Ridge.predict(X_test)
Y_LASSO_test = LASSO.predict(X_test)
# Evaluamos la calidad del modelo con los datos de TEST
Y_OLS_plot = OLS.predict(X)
Y_Ridge_plot = Ridge.predict(X)
Y_LASSO_plot = LASSO.predict(X)
# predecimos sobre toda la columna X original.

w_OLS = OLS.coef_.flatten()[0]
w_Ridge = Ridge.coef_.flatten()[0]
w_LASSO = LASSO.coef_.flatten()[0]

# la función oficial para R-cuadrado:
R2_OLS = r2_score(y_test, Y_OLS_test)
R2_Ridge = r2_score(y_test, Y_Ridge_test)
R2_LASSO = r2_score(y_test, Y_LASSO_test)

mse_ols = mean_squared_error(y_test, Y_OLS_test)
mse_ridge = mean_squared_error(y_test, Y_Ridge_test)
mse_lasso = mean_squared_error(y_test, Y_LASSO_test)

print("Calculated slope (OLS):", w_OLS)
print("Calculated slope (Ridge):", w_Ridge)
print("Calculated slope (LASSO):", w_LASSO)

print(f"R-squared en el set de TEST para OLS -> R²: {R2_OLS:.4f} | MSE: {mse_ols:.2f}")
print(f"R-squared en el set de TEST para Ridge -> R²: {R2_Ridge:.4f} | MSE: {mse_ridge:.2f}")
print(f"R-squared en el set de TEST para LASSO -> R²: {R2_LASSO:.4f} | MSE: {mse_lasso:.2f}")


# --------------------------------------------------------------------------------
# COMBINED PLOT
# --------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.scatter(x, Y, color='royalblue', label='Datos reales (Ruido)', alpha=0.4)

# Plot each history array with a unique color and label
plt.plot(x, Y_OLS_plot, color='red', linewidth=2, label=f'OLS (R²={R2_OLS:.4f})')
plt.plot(x, Y_Ridge_plot, color='orange', linewidth=2, label=f'Ridge (R²={R2_Ridge:.4f})')
plt.plot(x, Y_LASSO_plot, color='green', linewidth=2, label=f'LASSO (R²={R2_LASSO:.4f})')

plt.title(f'Comparison: OLS vs Ridge vs LASSO (lambda={alpha})')
plt.xlabel('t^2 (Time squared)')
plt.ylabel('Distancia (y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
