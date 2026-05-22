from flask import Flask, render_template, jsonify
import pandas as pd
import os
import glob
import json

app = Flask(__name__)

POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiNDgwNWJmZTItNjRjOS00NThhLWIyNzEtMGQxYWViNWY2ZjgzIiwidCI6IjA3ZGE2N2EwLTFmNDMtNGU4Yy05NzdmLTVmODhiNjQ3MGVlNiIsImMiOjR9"
RUTA = 'modelo_estrella'

def leer(tabla):
    patron = os.path.join(RUTA, tabla, 'part-*.csv')
    archivos = glob.glob(patron)
    if not archivos:
        return pd.DataFrame()
    return pd.read_csv(archivos[0], encoding='utf-8')

# Cargar tablas del modelo estrella
fact   = leer('fact_casos')
tiempo = leer('dim_tiempo')
muni   = leer('dim_municipio')
victi  = leer('dim_victima')
agres  = leer('dim_agresor')
hecho  = leer('dim_hecho')

df = (fact
      .merge(tiempo, on='id_tiempo',    how='left')
      .merge(muni,   on='id_municipio', how='left')
      .merge(victi,  on='id_victima',   how='left')
      .merge(agres,  on='id_agresor',   how='left')
      .merge(hecho,  on='id_hecho',     how='left'))

# Cargar datos de minería
clusters = pd.read_csv('DataSet/clusters_limpio.csv', encoding='utf-8-sig')
reglas   = pd.read_csv('DataSet/reglas_limpio.csv',   encoding='utf-8-sig')

# Cargar resultados ML
with open('DataSet/resultados_ml.json', encoding='utf-8') as f:
    resultados_ml = json.load(f)

# ── Rutas ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', powerbi_url=POWERBI_URL)

@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

@app.route('/mineria')
def mineria():
    return render_template('mineria.html')

@app.route('/profundo')
def profundo():
    return render_template('profundo.html')

# ── API modelo estrella ────────────────────────────────────────────────────────
@app.route('/api/resumen_modelo')
def api_resumen_modelo():
    return jsonify({
        'tablas': [
            {'nombre': 'Casos registrados',         'registros': len(fact),  'descripcion': 'Tabla de hechos principal'},
            {'nombre': 'Tiempo',                    'registros': len(tiempo),'descripcion': 'Dimensión temporal'},
            {'nombre': 'Municipios',                'registros': len(muni),  'descripcion': 'Dimensión geográfica'},
            {'nombre': 'Perfiles de víctimas',      'registros': len(victi), 'descripcion': 'Características de las víctimas'},
            {'nombre': 'Perfiles de agresores',     'registros': len(agres), 'descripcion': 'Características de los agresores'},
            {'nombre': 'Características del hecho', 'registros': len(hecho), 'descripcion': 'Contexto y circunstancias'},
        ]
    })

@app.route('/api/top_combinaciones')
def api_top_combinaciones():
    top = (df.groupby(['factor_desencadenante', 'escenario', 'presunto_agresor'])['cantidad']
             .sum().reset_index()
             .sort_values('cantidad', ascending=False).head(10))
    return jsonify(top.to_dict(orient='records'))

@app.route('/api/ciclovital_genero')
def api_ciclovital_genero():
    data = (df.groupby(['ciclo_vital', 'identidad_genero'])['cantidad']
              .sum().reset_index()
              .sort_values('cantidad', ascending=False))
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/calor_hora_dia')
def api_calor_hora_dia():
    data = (df.groupby(['dia', 'rango_hora'])['cantidad']
              .sum().reset_index())
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/casos_municipio_año')
def api_casos_municipio_año():
    data = (df.groupby(['municipio', 'año'])['cantidad']
              .sum().reset_index()
              .sort_values(['municipio', 'año']))
    return jsonify(data.to_dict(orient='records'))

# ── API minería ────────────────────────────────────────────────────────────────
@app.route('/api/clusters')
def api_clusters():
    conteo = clusters['Cluster'].value_counts().reset_index()
    conteo.columns = ['cluster', 'casos']
    return jsonify(conteo.to_dict(orient='records'))

@app.route('/api/clusters_perfil')
def api_clusters_perfil():
    perfil = (clusters.groupby(['Cluster', 'Factor Desencadenante de la Agresión'])
              .size().reset_index(name='casos')
              .sort_values(['Cluster', 'casos'], ascending=[True, False]))
    top = perfil.groupby('Cluster').head(3)
    return jsonify(top.to_dict(orient='records'))

@app.route('/api/clusters_escenario')
def api_clusters_escenario():
    data = (clusters.groupby(['Cluster', 'Escenario del Hecho'])
            .size().reset_index(name='casos')
            .sort_values(['Cluster', 'casos'], ascending=[True, False]))
    top = data.groupby('Cluster').head(3)
    return jsonify(top.to_dict(orient='records'))

@app.route('/api/reglas')
def api_reglas():
    return jsonify(reglas.head(15).to_dict(orient='records'))

# ── API conocimiento profundo ──────────────────────────────────────────────────
@app.route('/api/modelos_ml')
def api_modelos_ml():
    return jsonify(resultados_ml['modelos'])

@app.route('/api/tiempos_ml')
def api_tiempos_ml():
    return jsonify(resultados_ml['tiempos'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)