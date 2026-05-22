from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time
import os

os.environ['HADOOP_HOME'] = r'D:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + r';D:\hadoop\bin'

# ── Función para medir tiempo ──────────────────────────────────────────────────
def medir_tiempo(nombre, funcion):
    inicio = time.time()
    resultado = funcion()
    fin = time.time()
    segundos = round(fin - inicio, 3)
    print(f"  {nombre}: {segundos}s")
    return resultado, segundos

tiempos = {}

# ── 1. Iniciar Spark (modo local = master solo) ────────────────────────────────
print("\n=== CONFIGURACIÓN 1: Solo master (local[1]) ===")
spark1 = SparkSession.builder \
    .appName("VIF_ML_Master") \
    .master("local[1]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark1.sparkContext.setLogLevel("ERROR")

RUTA = r'D:\U_CUN\9_Semestre\MINERIA_DE_DATOS\Dashboard_VIF\DataSet\vif_final.csv'
MUNICIPIOS = ['25754', '25175', '25126', '25307', '25320', '25513', '25214']

# Cargar datos
def cargar_datos(spark):
    df = spark.read.csv(RUTA, header=True, inferSchema=False, encoding='utf-8')
    df = df.filter(F.col('Código Dane Municipio').isin(MUNICIPIOS))
    df = df.fillna('Sin información')
    return df

# ── Features compartidas ───────────────────────────────────────────────────────
FEATURES = [
    'Zona del Hecho',
    'Sexo de la victima',
    'Sexo del Agresor',
    'Ciclo Vital',
    'Municipio del hecho DANE',
    'Mes del hecho',
    'Dia del hecho'
]

# ── Pipeline Modelo 1: Predecir escenario (Vivienda vs No Vivienda) ────────────
def construir_pipeline_escenario(df):
    df = df.withColumn('escenario_bin',
        F.when(F.col('Escenario del Hecho') == 'Vivienda', 'Vivienda')
         .otherwise('Otro'))

    cols_index = FEATURES + ['Factor Desencadenante de la Agresión']
    indexers = [StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid='keep')
                for c in cols_index]
    target_indexer = StringIndexer(inputCol='escenario_bin', outputCol='label',
                                   handleInvalid='keep')
    assembler = VectorAssembler(
        inputCols=[c+'_idx' for c in cols_index],
        outputCol='features'
    )
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    pipeline = Pipeline(stages=indexers + [target_indexer, assembler, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    modelo = pipeline.fit(train)
    predicciones = modelo.transform(test)
    return predicciones

# ── Pipeline Modelo 2: Predecir Factor Desencadenante (multiclase) ─────────────
def construir_pipeline_factor(df):
    indexers = [StringIndexer(inputCol=c, outputCol=c+'_idx', handleInvalid='keep')
                for c in FEATURES]
    target_indexer = StringIndexer(
        inputCol='Factor Desencadenante de la Agresión',
        outputCol='label', handleInvalid='keep'
    )
    assembler = VectorAssembler(
        inputCols=[c+'_idx' for c in FEATURES],
        outputCol='features'
    )
    lr = LogisticRegression(maxIter=10, regParam=0.01,
                            family='multinomial')
    pipeline = Pipeline(stages=indexers + [target_indexer, assembler, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    modelo = pipeline.fit(train)
    predicciones = modelo.transform(test)
    return predicciones

evaluador_bin   = BinaryClassificationEvaluator(metricName='areaUnderROC')
evaluador_multi = MulticlassClassificationEvaluator(metricName='accuracy')

# ── Ejecutar con master solo ───────────────────────────────────────────────────
df1 = cargar_datos(spark1)
df1.cache()

print("\n--- Modelo 1: Predicción de Escenario (Vivienda vs Otro) ---")
pred_esc1, t_esc1 = medir_tiempo("Master solo", lambda: construir_pipeline_escenario(df1))
auc1 = round(evaluador_multi.evaluate(pred_esc1), 4)
print(f"  Accuracy: {auc1}")
tiempos['escenario_master1'] = t_esc1

print("\n--- Modelo 2: Predicción de Factor Desencadenante ---")
pred_fac1, t_fac1 = medir_tiempo("Master solo", lambda: construir_pipeline_factor(df1))
acc1 = round(evaluador_multi.evaluate(pred_fac1), 4)
print(f"  Accuracy: {acc1}")
tiempos['factor_master1'] = t_fac1

spark1.stop()

# ── 2. Spark con 2 threads (master + 1 worker simulado) ───────────────────────
print("\n=== CONFIGURACIÓN 2: Master + 1 worker (local[2]) ===")
spark2 = SparkSession.builder \
    .appName("VIF_ML_Worker1") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark2.sparkContext.setLogLevel("ERROR")

df2 = cargar_datos(spark2)
df2.cache()

pred_esc2, t_esc2 = medir_tiempo("Master + 1 worker", lambda: construir_pipeline_escenario(df2))
tiempos['escenario_master2'] = t_esc2

pred_fac2, t_fac2 = medir_tiempo("Master + 1 worker", lambda: construir_pipeline_factor(df2))
tiempos['factor_master2'] = t_fac2

spark2.stop()

# ── 3. Spark con 4 threads (master + 2 workers simulado) ──────────────────────
print("\n=== CONFIGURACIÓN 3: Master + 2 workers (local[4]) ===")
spark3 = SparkSession.builder \
    .appName("VIF_ML_Worker2") \
    .master("local[4]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark3.sparkContext.setLogLevel("ERROR")

df3 = cargar_datos(spark3)
df3.cache()

pred_esc3, t_esc3 = medir_tiempo("Master + 2 workers", lambda: construir_pipeline_escenario(df3))
tiempos['escenario_master3'] = t_esc3

pred_fac3, t_fac3 = medir_tiempo("Master + 2 workers", lambda: construir_pipeline_factor(df3))
tiempos['factor_master3'] = t_fac3

spark3.stop()

# ── 4. Guardar resultados ──────────────────────────────────────────────────────
import pandas as pd
import json

resultados = {
    'modelos': [
        {
            'nombre': 'Predicción de Escenario',
            'tipo': 'Regresión Logística Binaria',
            'descripcion': 'Predice si el hecho ocurre en Vivienda o en otro lugar',
            'metrica': 'Accuracy',
            'valor': auc1,
            'interpretacion': 'Vivienda vs Otro escenario'
        },
        {
            'nombre': 'Predicción de Factor Desencadenante',
            'tipo': 'Regresión Logística Multiclase',
            'descripcion': 'Predice el factor que desencadena la violencia',
            'metrica': 'Accuracy',
            'valor': acc1,
            'interpretacion': '12 categorías de factores desencadenantes'
        }
    ],
    'tiempos': [
        {'configuracion': 'Solo master (1 thread)',    'escenario': tiempos['escenario_master1'], 'factor': tiempos['factor_master1']},
        {'configuracion': 'Master + 1 worker (2 threads)', 'escenario': tiempos['escenario_master2'], 'factor': tiempos['factor_master2']},
        {'configuracion': 'Master + 2 workers (4 threads)', 'escenario': tiempos['escenario_master3'], 'factor': tiempos['factor_master3']},
    ]
}

with open('DataSet/resultados_ml.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n Resultados guardados en DataSet/resultados_ml.json")
print("\n=== RESUMEN DE TIEMPOS ===")
for t in resultados['tiempos']:
    print(f"  {t['configuracion']}: escenario={t['escenario']}s  factor={t['factor']}s")
print(f"\n=== MÉTRICAS ===")
print(f"  Modelo Escenario  Accuracy: {auc1}")
print(f"  Modelo Factor     Accuracy: {acc1}")
