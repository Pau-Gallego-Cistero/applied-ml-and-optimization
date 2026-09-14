import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Parameters
k = 5
sigma = 1
N = 100

# Generate data
x = np.linspace(0, 10, N)
noise = np.random.normal(0, sigma, N)
F = -k * x + noise

# Split the data
x_train, x_test, F_train, F_test = train_test_split(x, F, test_size=0.1 , random_state=42, shuffle=True)

# Linear model (Degree 2)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
x_test_poly = poly.transform(x_test.reshape(-1, 1))

model = LinearRegression()
model.fit(x_train_poly, F_train)

# Predictions
F_train_pred = model.predict(x_train_poly)
F_test_pred = model.predict(x_test_poly)

# Calculate errors
E_in = mean_squared_error(F_train, F_train_pred)
E_out = mean_squared_error(F_test, F_test_pred)

# Calculus of the R^2 error .score() R^2 de 0 a 1
r2_train = model.score(x_train_poly, F_train)
r2_test = model.score(x_test_poly, F_test)

print(f"E_in: {E_in}")
print(f"E_out: {E_out}")
print(f"Precision R^2 Train: {r2_train*100:.2f}%")
print(f"Precision R^2 Test: {r2_test*100:.2f}%")

# Plot data
plt.scatter(x_train, F_train, label="Training data", color="blue", alpha=0.7)
plt.scatter(x_test, F_test, label="Test data", color="green", alpha=0.7)

x_range_plot = np.linspace(0, 10, 100).reshape(-1, 1)
plt.plot(x_range_plot, model.predict(poly.transform(x_range_plot)), 
         label=f"Fitted model ($R^2 = {r2_test:.4f}$)", 
         color="red")
plt.xlabel("Displacement (x)")
plt.ylabel("Force (F)")
plt.title("Model fit visualization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Comparativa de 3 grados
plt.figure(figsize=(10, 6))
plt.scatter(x_train, F_train, label="Training data", color="blue", alpha=0.5)
plt.scatter(x_test, F_test, label="Test data", color="green", alpha=0.7)

degrees = [1, 2, 3]
colors = ['red', 'orange', 'purple']

for d, c in zip(degrees, colors):
    poly = PolynomialFeatures(degree=d)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly, F_train)

    # R^2 calculus (0 to 1)
    r2_test = model.score(x_test_poly, F_test)
    E_out = mean_squared_error(F_test, model.predict(x_test_poly))
    
    print(f"Grado {d} -> R^2 Test: {r2_test:.4f}, MSE: {E_out:.2f}")

    x_range = np.linspace(0, 10, 100).reshape(-1, 1)
    y_plot = model.predict(poly.transform(x_range))
    plt.plot(x_range, y_plot, label=f"Degree {d} (R^2 = {r2_test:.3f})", color=c, linewidth=2)

plt.xlabel("Displacement (x)")
plt.ylabel("Force (F)")
plt.title("Model Fit Comparison: Polynomial Degrees 1, 2, and 3")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()