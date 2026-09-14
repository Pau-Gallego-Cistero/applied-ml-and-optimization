
# Sirve para limpiar los datos de nuevos datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from IPython.display import display


data = pd.read_csv("oec.csv")
display(data)

# Split de Datos y Target
X = np.array([  ])
X = X.T
display(X)
    
features = [
    'PlanetaryMassJpt', 
    'RadiusJpt', 
    'PeriodDays', 
    'DistFromSunParsec', 
    'HostStarMassSlrMass', 
    'HostStarTempK'
]

# Seleccionar mis datos relevantes
df = data[features + ['DiscoveryMethod']].copy()

# Eliminamos filas donde falte al menos un dato para el RF
df_clean = df.dropna()

X = df_clean[features]
y = df_clean['DiscoveryMethod']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler() #Llamamos al estandarizador
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
precisión = accuracy_score(y_test, y_pred)


print(df_clean.head())

print(f"Target names: {y.unique()}") 
print(f"Media (debe ser 0): {X_train_scaled.mean():.2f}")
print(f"Desviación (debe ser 1): {X_train_scaled.std():.2f}")
print(f"Precisión del Random Forest: {precisión * 100:.2f}%")



# Creamos un dataset corrupto para poner en practica el modelo
df_corrupt = pd.read_csv("oec.csv")

df_corrupt.loc[::10, 'PlanetaryMassJpt'] *= 10  
df_corrupt.loc[::15, 'HostStarTempK'] = 99999

df_corrupt.to_csv("oec_corrupto.csv", index=False)

# Aplicamos el modelo
df_erroneo = pd.read_csv("oec_corrupto.csv")
df_erroneo = df_erroneo.dropna(subset=features)

X_err = df_erroneo[features]
X_err_scaled = scaler.transform(X_err)
probs = clf.predict_proba(X_err_scaled)
max_probs = np.max(probs, axis=1)

umbral_confianza = 0.5
df_erroneo['confianza'] = max_probs
df_erroneo['es_anomalo'] = max_probs < umbral_confianza # Creamos una nueva columna (True/False)

anomalias = df_erroneo[df_erroneo['es_anomalo'] == True] # Contiene solo las filas donde error
print(f"\nSe han detectado {len(anomalias)} filas potencialmente erróneas")
display(anomalias.head())

# Las filas con baja confianza suelen ser aquellas donde las
# características (Masa/Temperatura) no coinciden con los patrones del método de descubrimiento.

