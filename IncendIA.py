import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import folium
from sklearn.tree import DecisionTreeClassifier
from folium.plugins import HeatMap

# Leer el archivo de datos
dtype = {'X': float, 'Y': float, 'MUNICIPIO': str}
path = 'C:/Users/hugom/Downloads/IncendIA/Data_full.xlsx'
data = pd.read_excel(path, dtype=dtype)

# Limpiar los valores nulos
data.dropna(inplace=True)

# Seleccionar características relevantes
features = ['X', 'Y']
X = data[features]
Y = data['MUNICIPIO']

X.columns = [''] * len(X.columns)

# Estandarizar y normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Convertir los datos de entrenamiento y prueba a matrices NumPy
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)

# Entrenar un modelo de árbol de decisiones
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, Y_train)

# Realizar predicciones
Y_pred = tree_classifier.predict(X_test)

from sklearn.model_selection import GridSearchCV

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializar el clasificador de árbol de decisiones
tree_clf = DecisionTreeClassifier(random_state=42)

# Inicializar el objeto de búsqueda en cuadrícula
grid_search = GridSearchCV(estimator=tree_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Realizar la búsqueda en cuadrícula en los datos de entrenamiento
grid_search.fit(X_train, Y_train)

# Obtener los mejores hiperparámetros encontrados
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Realizar predicciones en los datos de prueba
Y_pred = best_model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
print("Classification report:")
print(classification_report(Y_test, Y_pred))

# Crear DataFrame con datos de prueba y predicciones
data_filtered_test = pd.DataFrame(X_test, columns=['X', 'Y'])
data_filtered_test['Predicted_Municipality'] = Y_pred

plt.figure(figsize=(12, 8))
sns.scatterplot(x='X', y='Y', hue='Predicted_Municipality', data=data_filtered_test, palette='viridis', legend='full')
plt.title('Diagrama de dispersión de zonas de riesgo de incendio')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

plt.figure(figsize=(12, 8))
sns.kdeplot(data=data_filtered_test[['X', 'Y']], fill=True, thresh=0, levels=100)
plt.title('Mapa de calor de zonas de riesgo de incendio')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

#--------------------MAPA----------------------------

mapa = folium.Map(location=[24.027, -104.653], zoom_start=7)

#función para asignar colores según las predicciones del modelo
def get_color(prediction):
        return 'red'  

# Iterar sobre cada fila del DataFrame y agregar un marcador con el color correspondiente
for _, row in data.iterrows():
    prediction = tree_classifier.predict([[row['X'], row['Y']]])[0]
    color = get_color(prediction)
    folium.CircleMarker([row['Y'], row['X']], radius=5, color=color, fill=True, fill_color=color).add_to(mapa)

# Guardar el mapa como un archivo HTML
mapa.save('Mapa_de_predicciones.html')