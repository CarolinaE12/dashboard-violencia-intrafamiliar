from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

os.environ['HADOOP_HOME'] = r'D:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + r';D:\hadoop\bin'

spark = SparkSession.builder \
    .appName("Modelo Estrella VIF") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

RUTA_VIF    = r'D:\U_CUN\9_Semestre\MINERIA_DE_DATOS\Dashboard_VIF\DataSet\vif_final.csv'
RUTA_SALIDA = r'D:\U_CUN\9_Semestre\MINERIA_DE_DATOS\Dashboard_VIF\modelo_estrella'

MUNICIPIOS = ['25754', '25175', '25126', '25307', '25320', '25513', '25214']

os.makedirs(RUTA_SALIDA, exist_ok=True)

print("Cargando datos con Spark...")

# ── 1. Cargar VIF ──────────────────────────────────────────────────────────────
df = spark.read.csv(RUTA_VIF, header=True, inferSchema=False, encoding='utf-8')
df = df.filter(F.col('Código Dane Municipio').isin(MUNICIPIOS))
df = df.withColumn('id_caso', F.monotonically_increasing_id() + 1)
print(f"VIF cargado: {df.count()} registros")

# ── 2. dim_tiempo ──────────────────────────────────────────────────────────────
dim_tiempo = df.select(
    'Año del hecho', 'Mes del hecho',
    'Dia del hecho', 'Rango de Hora del Hecho X 3 Horas'
).distinct()
dim_tiempo = dim_tiempo.withColumn('id_tiempo', F.monotonically_increasing_id() + 1)
dim_tiempo = dim_tiempo \
    .withColumnRenamed('Año del hecho', 'año') \
    .withColumnRenamed('Mes del hecho', 'mes') \
    .withColumnRenamed('Dia del hecho', 'dia') \
    .withColumnRenamed('Rango de Hora del Hecho X 3 Horas', 'rango_hora')

df = df.join(dim_tiempo,
    (df['Año del hecho'] == dim_tiempo['año']) &
    (df['Mes del hecho'] == dim_tiempo['mes']) &
    (df['Dia del hecho'] == dim_tiempo['dia']) &
    (df['Rango de Hora del Hecho X 3 Horas'] == dim_tiempo['rango_hora']),
    how='left')
print(f"dim_tiempo: {dim_tiempo.count()} registros")

# ── 3. dim_municipio ───────────────────────────────────────────────────────────
dim_municipio = df.select(
    'Código Dane Municipio', 'Municipio del hecho DANE'
).distinct()
dim_municipio = dim_municipio.withColumn('id_municipio', F.monotonically_increasing_id() + 1)
dim_municipio = dim_municipio \
    .withColumnRenamed('Código Dane Municipio', 'codigo_dane') \
    .withColumnRenamed('Municipio del hecho DANE', 'municipio')

df = df.join(dim_municipio.select('codigo_dane', 'id_municipio'),
    df['Código Dane Municipio'] == dim_municipio['codigo_dane'],
    how='left')
print(f"dim_municipio: {dim_municipio.count()} registros")

# ── 4. dim_victima ─────────────────────────────────────────────────────────────
dim_victima = df.select(
    'Sexo de la victima', 'Ciclo Vital', 'Escolaridad',
    'Estado Civil', 'Identidad de Género', 'Transgénero',
    'Tipo de Discapacidad'
).distinct()
dim_victima = dim_victima.withColumn('id_victima', F.monotonically_increasing_id() + 1)
dim_victima = dim_victima \
    .withColumnRenamed('Sexo de la victima', 'sexo') \
    .withColumnRenamed('Ciclo Vital', 'ciclo_vital') \
    .withColumnRenamed('Escolaridad', 'escolaridad') \
    .withColumnRenamed('Estado Civil', 'estado_civil') \
    .withColumnRenamed('Identidad de Género', 'identidad_genero') \
    .withColumnRenamed('Transgénero', 'transgenero') \
    .withColumnRenamed('Tipo de Discapacidad', 'tipo_discapacidad')

df = df.join(
    dim_victima.select('sexo', 'ciclo_vital', 'escolaridad', 'estado_civil',
                       'identidad_genero', 'transgenero', 'tipo_discapacidad',
                       'id_victima'),
    (df['Sexo de la victima']   == dim_victima['sexo']) &
    (df['Ciclo Vital']          == dim_victima['ciclo_vital']) &
    (df['Escolaridad']          == dim_victima['escolaridad']) &
    (df['Estado Civil']         == dim_victima['estado_civil']) &
    (df['Identidad de Género']  == dim_victima['identidad_genero']) &
    (df['Transgénero']          == dim_victima['transgenero']) &
    (df['Tipo de Discapacidad'] == dim_victima['tipo_discapacidad']),
    how='left')
print(f"dim_victima: {dim_victima.count()} registros")

# ── 5. dim_agresor ─────────────────────────────────────────────────────────────
dim_agresor = df.select(
    'Sexo del Agresor', 'Presunto Agresor Detallado'
).distinct()
dim_agresor = dim_agresor.withColumn('id_agresor', F.monotonically_increasing_id() + 1)
dim_agresor = dim_agresor \
    .withColumnRenamed('Sexo del Agresor', 'sexo_agresor') \
    .withColumnRenamed('Presunto Agresor Detallado', 'presunto_agresor')

df = df.join(
    dim_agresor.select('sexo_agresor', 'presunto_agresor', 'id_agresor'),
    (df['Sexo del Agresor']           == dim_agresor['sexo_agresor']) &
    (df['Presunto Agresor Detallado'] == dim_agresor['presunto_agresor']),
    how='left')
print(f"dim_agresor: {dim_agresor.count()} registros")

# ── 6. dim_hecho ───────────────────────────────────────────────────────────────
dim_hecho = df.select(
    'Zona del Hecho', 'Escenario del Hecho', 'Actividad Durante el Hecho',
    'Circunstancia del Hecho Detallada', 'Contexto del Hecho',
    'Mecanismo Causal de la Lesión no Fatal',
    'Diagnostico Topográfico de la Lesión no Fatal',
    'Factor Desencadenante de la Agresión',
    'Localidad del Hecho'
).distinct()
dim_hecho = dim_hecho.withColumn('id_hecho', F.monotonically_increasing_id() + 1)
dim_hecho = dim_hecho \
    .withColumnRenamed('Zona del Hecho', 'zona') \
    .withColumnRenamed('Escenario del Hecho', 'escenario') \
    .withColumnRenamed('Actividad Durante el Hecho', 'actividad') \
    .withColumnRenamed('Circunstancia del Hecho Detallada', 'circunstancia') \
    .withColumnRenamed('Contexto del Hecho', 'contexto') \
    .withColumnRenamed('Mecanismo Causal de la Lesión no Fatal', 'mecanismo_causal') \
    .withColumnRenamed('Diagnostico Topográfico de la Lesión no Fatal', 'diagnostico_topografico') \
    .withColumnRenamed('Factor Desencadenante de la Agresión', 'factor_desencadenante') \
    .withColumnRenamed('Localidad del Hecho', 'localidad')

df = df.join(
    dim_hecho.select('zona', 'escenario', 'actividad', 'circunstancia',
                     'contexto', 'mecanismo_causal', 'diagnostico_topografico',
                     'factor_desencadenante', 'localidad', 'id_hecho'),
    (df['Zona del Hecho']       == dim_hecho['zona']) &
    (df['Escenario del Hecho']  == dim_hecho['escenario']) &
    (df['Actividad Durante el Hecho'] == dim_hecho['actividad']) &
    (df['Circunstancia del Hecho Detallada'] == dim_hecho['circunstancia']) &
    (df['Contexto del Hecho']   == dim_hecho['contexto']) &
    (df['Mecanismo Causal de la Lesión no Fatal'] == dim_hecho['mecanismo_causal']) &
    (df['Diagnostico Topográfico de la Lesión no Fatal'] == dim_hecho['diagnostico_topografico']) &
    (df['Factor Desencadenante de la Agresión'] == dim_hecho['factor_desencadenante']) &
    (df['Localidad del Hecho']  == dim_hecho['localidad']),
    how='left')
print(f"dim_hecho: {dim_hecho.count()} registros")

# ── 7. fact_casos ──────────────────────────────────────────────────────────────
fact_casos = df.select('id_caso', 'id_tiempo', 'id_municipio',
                        'id_victima', 'id_agresor', 'id_hecho')
fact_casos = fact_casos.withColumn('cantidad', F.lit(1))
print(f"fact_casos: {fact_casos.count()} registros")

# ── 8. Exportar ────────────────────────────────────────────────────────────────
print("\nExportando tablas...")

def guardar(df_spark, nombre):
    path = f'{RUTA_SALIDA}/{nombre}'
    df_spark.coalesce(1).write.csv(path, header=True, mode='overwrite', encoding='utf-8')
    print(f"  {nombre} guardado")

guardar(dim_tiempo,    'dim_tiempo')
guardar(dim_municipio, 'dim_municipio')
guardar(dim_victima,   'dim_victima')
guardar(dim_agresor,   'dim_agresor')
guardar(dim_hecho,     'dim_hecho')
guardar(fact_casos,    'fact_casos')

print("\n✓ Modelo estrella creado exitosamente")
spark.stop()