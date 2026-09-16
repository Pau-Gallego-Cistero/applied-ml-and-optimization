#--------------------------------------------------------------------------------
# a) Dataset preparation
#--------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import pandas as pd


wine_data = load_wine() #cargar data

X = wine_data.data
y = wine_data.target
    
# Splitting into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Dframe = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Add the target variable (0 = class_0, 1 = class_1, 2 = class_2) 
Dframe['target'] = wine_data.target

scaler = StandardScaler() #Llamamos al estandarizador
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(Dframe.head())

print(f"Target names: {wine_data.target_names}")
print(f"Media (debe ser 0): {X_train_scaled.mean():.2f}")
print(f"Desviación (debe ser 1): {X_train_scaled.std():.2f}")


#--------------------------------------------------------------------------------
# b) 
#--------------------------------------------------------------------------------


model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate accuracy
acc_Log_train = accuracy_score(y_train, y_train_pred)
acc_Log_test  = accuracy_score(y_test,  y_test_pred)

print(f"Precision Train: {acc_Log_train*100:.2f}%")
print(f"Precision Test: {acc_Log_test*100:.2f}%")

#--------------------------------------------------------------------------------
# c) Gradient Boosting and AdaBoost with 50 estimators
#--------------------------------------------------------------------------------

gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb.fit(X_train_scaled, y_train)

acc_gb_train = accuracy_score(y_train, gb.predict(X_train_scaled))
acc_gb_test  = accuracy_score(y_test,  gb.predict(X_test_scaled))

ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train_scaled, y_train)

acc_ada_train = accuracy_score(y_train, ada.predict(X_train_scaled))
acc_ada_test  = accuracy_score(y_test,  ada.predict(X_test_scaled))

print(f"Gradient Boosting — Train: {acc_gb_train*100:.2f}%  |  Test: {acc_gb_test*100:.2f}%")
print(f"AdaBoost          — Train: {acc_ada_train*100:.2f}%  |  Test: {acc_ada_test*100:.2f}%")



#--------------------------------------------------------------------------------
# d) Bar chart comparing training and test accuracies
#--------------------------------------------------------------------------------


models      = ['Gradient Boosting', 'AdaBoost']
train_accs  = [acc_gb_train*100, acc_ada_train*100]
test_accs   = [acc_gb_test*100,  acc_ada_test*100]

x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_train = ax.bar(x - width/2, train_accs, width, label='Train', color='steelblue',  edgecolor='navy')
bars_test  = ax.bar(x + width/2, test_accs,  width, label='Test',  color='lightsalmon', edgecolor='darkred')

ax.set_title('Training vs Test Accuracy — Wine dataset', fontsize=13)
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(True, alpha=0.3)

for bar in bars_train:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar in bars_test:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
	
#--------------------------------------------------------------------------------
# e) Discussion
#--------------------------------------------------------------------------------

print("""
--- Discussion ---

Gradient Boosting trains trees sequentially, each correcting the residual errors
of the previous one. It tends to reach near-perfect training accuracy (low bias)
but can overfit more than Random Forest with a small number of estimators,
which may lower test performance slightly.

AdaBoost also works sequentially but focuses on misclassified samples by
reweighting them. On a 3-class problem like Wine it is not as sensitive to
noisy or ambiguous samples as it would be on a 10-class problem like Digits, 
which can hurt generalisation compared to the other two ensemble methods. However,
since it is a 3-class, as mentioned, it behaves more similarly to GBoosting.

Moreover, Logistic Regression has now been implemented properly with scaled data.
Being a powerful linear model that handles multiclass classification natively, it 
achieves highly competitive precision, matching or exceeding the Boosting models 
without the risk of overfitting the training set.

In summary: Gradient Boosting ≈ AdaBoost on this dataset,
with the gap explained by variance control (bagging) vs sequential correction
(boosting) trade-offs.

""")