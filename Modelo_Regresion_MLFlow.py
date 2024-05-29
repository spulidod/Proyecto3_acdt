import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
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
df = df.head(70000)
df = df.dropna()
# df = df.iloc[50000:]

# Variables de entrada (solo las columnas especificadas)
columnas_X = ['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
       'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
       'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA',
       'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'ESTU_GENERO',
       'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
       'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
       'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
       'FAMI_TIENELAVADORA']  # , 'DESEMP_INGLES', 'PUNT_INGLES','PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA'

X = df[columnas_X]
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
y_train = y_train.to_numpy().astype(float)
y_test = y_test.to_numpy().astype(float)

# Configuración de MLflow
mlflow.set_tracking_uri('http://18.233.216.162:8050//')

# Lista de optimizadores con learning rate
# optimizadores = {
#     'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.001),
#     'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
#     'Momentum': tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
#     'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
#     'Nadam': tf.keras.optimizers.Nadam(learning_rate=0.001),
#     'Adamax': tf.keras.optimizers.Adamax(learning_rate=0.001),
#     'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=0.001),
#     'FTRL': tf.keras.optimizers.Ftrl(learning_rate=0.001)
# }
learning_rate = 0.001

# Construir el modelo
def build_model(activation, neurons, input_shape):
    tf.random.set_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer="he_normal"),
        tf.keras.layers.Dense(neurons // 2, activation=activation, kernel_initializer="he_normal"),
        tf.keras.layers.Dense(1)
    ])

def build_and_train_model(activation, neurons, epochs, learning_rate):
    input_shape = (X_train.shape[1],)  # Asignar forma de entrada
    model = build_model(activation, neurons, input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Crear un nuevo optimizador
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_absolute_error", "mean_absolute_percentage_error"])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))  # , verbose=0)
    return model, history

# Configurar experimento en MLflow
mlflow.set_experiment("Numero de neuronas")

# Lista de funciones de activación
# activaciones = {
#     'ReLU': 'relu',
#     'Sigmoid': 'sigmoid',
#     'Tanh': 'tanh',
#     'ELU': 'elu',
#     'SELU': 'selu',
#     'Mish': 'mish',
#     'Linear': 'linear'
# }
num_neurons_list = [2 ** i for i in range(3, 9)]

# Parámetros
epocas = 5
# neuronas = 16
learning_rate = 0.001
activacion = 'linear'

# Entrenar el modelo y registrar con MLflow
for neuronas in num_neurons_list:
    run_name = f"Número de Neuronas: {neuronas}"
    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.tensorflow.autolog()
            model, history = build_and_train_model(activacion, neuronas, epocas, learning_rate)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("activation", activacion)
            mlflow.log_param("neurons", neuronas)

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

        finally:
            mlflow.end_run()

# Para visualizar los experimentos, ejecuta en la terminal:
# mlflow ui


# Para visualizar los experimentos, ejecuta en la terminal:
# mlflow ui




    