<h1>Modelo de Análisis Periódico de Contratos.</h1>

<h2> Funcionamiento </h2>

* Preprocesamiento de los datos:

Se considera el monto y la duración de los contratos para que estuvieran en un rango comparable.

Los datos fueron escalados para que los valores numéricos no fueran dominados por las diferencias en magnitud (por ejemplo, un contrato muy costoso frente a uno económico).

* Definición de patrones:

Se definieron dos parámetros clave: distancia máxima entre puntos para considerarlos similares(vecinos) y la cantidad mínima de puntos necesarios para formar un grupo.

Se agrupan los datos que compartían características similares de monto y duración en diferentes grupos.

* Detección de anomalías:

Los contratos que no formaron parte de ningún grupo fueron clasificados como anomalías.

Estas anomalías representan contratos inusuales, ya sea porque tenían montos o duraciones significativamente diferentes a la mayoría.

<h2> Código anotado </h2>

```python
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo CSV
file_path = '/content/drive/MyDrive/dataton/archivo_sin_duplicados.csv'
data = pd.read_csv(file_path)

# Selección de columnas relevantes
columns = ['formatted_amount', 'startDate', 'endDate']
df = data[columns].copy()

# Limpieza de datos y cálculo de duración del contrato
df['formatted_amount'] = df['formatted_amount'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')
df['contract_duration'] = (df['endDate'] - df['startDate']).dt.days

# Eliminar filas con valores faltantes
df_clean = df[['formatted_amount', 'contract_duration']].dropna()

# Mostrar estadísticas básicas
print("Número de filas después del preprocesamiento:", len(df_clean))
if len(df_clean) < 3:
    print("Advertencia: Hay menos de 3 filas después del preprocesamiento. Considera revisar los datos.")
    print(df_clean)
else:
    print("Datos procesados exitosamente:")
    print(df_clean.head())

# Aplicar DBSCAN si hay suficientes datos
if len(df_clean) > 1:
    # Estandarización de datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)

    # Gráfico de k-distancias para determinar eps
    k = max(1, min(3, len(df_clean) - 1))
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)

    # Ordenar distancias y graficar
    distances = np.sort(distances[:, k - 1], axis=0)
    plt.plot(distances)
    plt.title('Gráfico de k-distancias')
    plt.xlabel('Puntos ordenados')
    plt.ylabel(f'Distancia al {k}-ésimo vecino más cercano')
    plt.show()

    # Solicitar valor de eps
    eps_value = float(input("Introduce el valor de eps basado en el gráfico de k-distancias: "))

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps_value, min_samples=k)
    labels = dbscan.fit_predict(data_scaled)

    # Agregar etiquetas de clusters al DataFrame original
    df_clean['Cluster'] = labels

    # Identificar y mostrar anomalías
    anomalies = df_clean[df_clean['Cluster'] == -1]
    print("Anomalías detectadas:")
    print(anomalies)

    # Gráfica en escala original
    plt.scatter(df_clean['formatted_amount'], df_clean['contract_duration'], c=labels, cmap='viridis', marker='o')
    plt.title('Análisis Periódico de Contratos')
    plt.xlabel('Monto pagado por el contrato (original)')
    plt.ylabel('Duración del contrato (días)')
    plt.show()

    # Estadísticas de clusters
    unique_labels = set(labels)
    print(f"Clusters encontrados: {len(unique_labels) - (1 if -1 in labels else 0)}")
    print(f"Anomalías detectadas: {list(labels).count(-1)}")
else:
    print("No hay suficientes datos para realizar clustering.")

```
<h2>Salida </h2>

