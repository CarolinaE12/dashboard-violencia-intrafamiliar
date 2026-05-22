import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── 1. Cargar y filtrar datos ──────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv('DataSet/vif_final.csv', encoding='utf-8', sep=None, engine='python')

codigos = ['25754', '25175', '25126', '25307', '25320', '25513', '25214']
df = df[df['Código Dane Municipio'].astype(str).isin(codigos)].copy()

df = df.reset_index(drop=True)
print(f"Registros: {len(df)}")

# ── 2. Columnas para clustering ────────────────────────────────────────────────
cols_cluster = [
    'Factor Desencadenante de la Agresión',
    'Escenario del Hecho',
    'Zona del Hecho',
    'Sexo de la victima',
    'Sexo del Agresor',
    'Ciclo Vital',
    'Municipio del hecho DANE'
]

df_cluster = df[cols_cluster].fillna('Sin información')

# Codificar categorías a números
encoders = {}
df_encoded = pd.DataFrame()
for col in cols_cluster:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_cluster[col].astype(str))
    encoders[col] = le

# K-Means con 3 clusters
print("Aplicando K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster = df_cluster.copy()
df_cluster['Cluster'] = kmeans.fit_predict(df_encoded)
df_cluster['Cluster'] = df_cluster['Cluster'].apply(lambda x: f'Cluster {x+1}')

df_cluster.to_csv('DataSet/clusters_limpio.csv', index=False, encoding='utf-8-sig')
print("Clusters guardados:")
print(df_cluster['Cluster'].value_counts())

# ── 3. Reglas de asociación ────────────────────────────────────────────────────
print("\nGenerando reglas de asociación...")

cols_reglas = [
    'Factor Desencadenante de la Agresión',
    'Escenario del Hecho',
    'Zona del Hecho',
    'Sexo de la victima',
    'Sexo del Agresor'
]

df_reglas = df[cols_reglas].fillna('Sin información')

# Convertir a transacciones
transacciones = []
for _, row in df_reglas.iterrows():
    transaccion = [f"{col}={val}" for col, val in row.items()]
    transacciones.append(transaccion)

te = TransactionEncoder()
te_array = te.fit_transform(transacciones)
df_te = pd.DataFrame(te_array, columns=te.columns_)

# Apriori
itemsets = apriori(df_te, min_support=0.1, use_colnames=True)
reglas = association_rules(itemsets, metric='confidence', min_threshold=0.6)
reglas = reglas.sort_values('confidence', ascending=False)

# Limpiar para exportar
reglas['antecedents'] = reglas['antecedents'].apply(lambda x: ', '.join(list(x)))
reglas['consequents'] = reglas['consequents'].apply(lambda x: ', '.join(list(x)))
reglas = reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
reglas['support'] = reglas['support'].round(3)
reglas['confidence'] = reglas['confidence'].round(3)
reglas['lift'] = reglas['lift'].round(3)

reglas.to_csv('DataSet/reglas_limpio.csv', index=False, encoding='utf-8-sig')
print(f"Reglas guardadas: {len(reglas)}")
print(reglas.head(5).to_string())

print("\n✓ Archivos generados:")
print("  DataSet/clusters_limpio.csv")
print("  DataSet/reglas_limpio.csv")