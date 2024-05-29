import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.tensorflow

# Configurar la instancia de MLflow en EC2
mlflow.set_tracking_uri("http://18.233.216.162:8050")

# Suponiendo que tienes tu DataFrame 'df' cargado
# Cargar los datos
df = pd.read_csv('data_limpieza.csv')
df = df.head(100000)

# Variable de respuesta (última columna)
y = df['PUNT_GLOBAL']

# Variables de entrada (solo las columnas especificadas)
columnas_X = ['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
       'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
       'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA',
       'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'ESTU_GENERO',
       'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
       'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
       'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
       'FAMI_TIENELAVADORA']

X= df[columnas_X]
# Definir rangos de puntajes y crear etiquetas de clasificación
bins = [0, 250, 350, float('inf')]
labels = [0, 1, 2]  # 0: Bajo, 1: Medio, 2: Alto
df['PUNT_GLOBAL_BIN'] = pd.cut(df['PUNT_GLOBAL'], bins=bins, labels=labels, right=False)

# Variables de entrada (todas menos 'PUNT_GLOBAL' y 'PUNT_GLOBAL_BIN')
#X = df.drop(columns=['PUNT_GLOBAL', 'PUNT_GLOBAL_BIN'])
# Variable de salida
y = df['PUNT_GLOBAL_BIN']

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

# Convertir y_train y y_test a números enteros
y_train = y_train.astype(int)
y_test = y_test.astype(int)


mlflow.set_experiment("Clasificacion_activacion_MGD")
cambio = "elu"
# Definir y entrenar el modelo
with mlflow.start_run(run_name=cambio):
    
    neurons = 32
    activation = cambio
    optimizer = "Adam"
    epocas = 6
    input_shape = (X_train.shape[1],)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer="he_normal"),
        tf.keras.layers.Dense(neurons // 2, activation=activation, kernel_initializer="he_normal"),
        tf.keras.layers.Dense(3, activation='softmax')  # Cambiar a softmax para clasificación multiclase
    ])

    # Compilar el modelo
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=epocas, validation_data=(X_test, y_test))

    # Predicciones
    y_pred_train = model.predict(X_train).argmax(axis=1)
    y_pred_test = model.predict(X_test).argmax(axis=1)

    # Calcular métricas adicionales
    precision = precision_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    f1 = f1_score(y_test, y_pred_test, average='macro')
    auc_score = roc_auc_score(y_test, model.predict(X_test), multi_class='ovr')

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc_score)

    mlflow.log_param("epocas", epocas)
    mlflow.log_param("optimizador", optimizer)
    mlflow.log_param("neuronas", neurons)
    mlflow.log_param("activacion", activation)
