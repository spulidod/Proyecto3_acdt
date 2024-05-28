import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
import mlflow.tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Cargar y preparar los datos
df = pd.read_csv('data_limpieza.csv')
#df = df.head(50000)
df = df.dropna()
df = df.iloc[50000:]

# Variables de entrada (solo las columnas especificadas)
columnas_X = ['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
       'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
       'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA',
       'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'ESTU_GENERO',
       'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
       'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
       'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
       'FAMI_TIENELAVADORA']#, 'DESEMP_INGLES', 'PUNT_INGLES','PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA'

# Variables de entrada (todas menos 'PUNT_GLOBAL')
X = df.drop(columns=['PUNT_GLOBAL'])
# Variable de salida
y = df['PUNT_GLOBAL']

# Columnas categóricas y numéricas
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

# Preprocesamiento: OneHotEncoding para variables categóricas, escalado para numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Preprocesar los datos
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Convertir y_train y y_test a números flotantes
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Configuración de MLflow
mlflow.set_tracking_uri('http://44.203.41.178:8050//')
experiment = mlflow.set_experiment("Prueba2")

# Definir y entrenar el modelo
with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    neurons = 32
    activation = "linear"
    optimizer="Adam"
    epocas=5
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer="he_normal"),
    tf.keras.layers.Dense(neurons // 2, activation=activation, kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1)
    ])

    # Compilar el modelo
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_absolute_error", "mean_absolute_percentage_error"])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    
    # Registrar parámetros y métricas en MLflow
    mlflow.log_param("Neuronas", neurons)
    mlflow.log_param("Activacion", activation)
    mlflow.log_param("epocas", epocas)
    
    mlflow.log_param("Optimizador", optimizer)   
    mlflow.keras.log_model(model, "Clas-RedesNeuronales")


    # Probar el modelo
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

    # Registrar métricas adicionales
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)