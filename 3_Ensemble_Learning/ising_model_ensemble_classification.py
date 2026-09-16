
#--------------------------------------------------------------------------------
# Bagging Classifier
#--------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

latice = 20
cold_temp = 1.5 # temp (Ordered)
hot_temp = 4.0  # temp (Disordered)
samples = 200
steps = 15000 

def get_neighbors_sum(grid, r, c, n):

    above = grid[(r - 1) % n, c]
    below = grid[(r + 1) % n, c]
    left  = grid[r, (c - 1) % n]
    right = grid[r, (c + 1) % n]
    
    return above + below + left + right

def should_we_flip(current_spin, neighbors_sum, temperature):

    # how much energy change this flip would cause
    # If the spin matches neighbors, energy is low. If it flips away, energy rises.
    energy_change = 2 * current_spin * neighbors_sum
    
    # Rule A: If flipping REDUCES energy (de <= 0), the system loves it. Always flip.
    if energy_change <= 0:
        return True
    
    # Rule B: If flipping INCREASES energy, we usually don't do it.
    # But heat allows for 'unstable' moves. We roll a dice.
    else:
        # Probability of flipping even though it's high energy:
        chance_of_flipping = np.exp(-energy_change / temperature)
        
        # Roll a random number between 0 and 1
        roll = np.random.random()
        
        if roll < chance_of_flipping:
            return True # Heat was high enough to cause the flip!
        else:
            return False # Stay as you are.

def create_single_grid(n, temp, steps):
    # Start with a random mix of +1 and -1
    grid = np.random.choice([1, -1], size=(n, n))
    
    for _ in range(steps):
        # Pick a random spot in the grid
        r, c = np.random.randint(0, n, size=2)
        
        # Get context
        myself = grid[r, c]
        others = get_neighbors_sum(grid, r, c, n)
        
        # Make the decision
        if should_we_flip(myself, others, temp):
            grid[r, c] *= -1 # Flip from 1 to -1 or vice versa
            
    return grid

# --- DATA GENERATION ---
X = [] # This will hold the flattened grids (the 'pictures')
y = [] # This will hold the labels (0 for Cold, 1 for Hot)

for _ in range(samples):
    # Make a cold grid
    cold_grid = create_single_grid(latice, cold_temp, steps)
    X.append(cold_grid.flatten()) # Flatten 20x20 into a list of 400
    y.append(0) 

    # Make a hot grid
    hot_grid = create_single_grid(latice, hot_temp, steps)
    X.append(hot_grid.flatten())
    y.append(1)

X = np.array(X)
y = np.array(y)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(),
    n_estimators=100, random_state=42)

modelo_bagging.fit(X_train, y_train)

predictions = modelo_bagging.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {score * 100}%")

# --- HEATMAP ---
# Promedio importancia arboles
importances = np.mean([tree.feature_importances_ for tree in modelo_bagging.estimators_],
                      axis=0).reshape(latice, latice)

plt.figure(figsize=(6, 5))
plt.title("Importance Heatmap: Where is the AI looking?")
plt.imshow(importances, cmap='hot')
plt.colorbar(label='Importance Level')
plt.show()