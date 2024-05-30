import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import keras
import numpy as np
import psycopg2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

conn = psycopg2.connect(
        dbname="colegios",
        user="postgres",
        password="santimaquina",
        host="fernando9.chksas62m4f6.us-east-1.rds.amazonaws.com",
        port='5432'
    )

def get_data(query):
    #conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

# Consulta para obtener los datos de la tabla
query_data = "SELECT * FROM tablename"

# Obtener los datos
data = pd.DataFrame(get_data(query_data), columns=['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
       'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
       'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA',
       'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'ESTU_GENERO',
       'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
       'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
       'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
       'FAMI_TIENELAVADORA', 'DESEMP_INGLES', 'PUNT_INGLES',
       'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES',
       'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL'])

# Definir el estilo externo
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Cargar el modelo serializado
model = keras.models.load_model('Modelo_Clasificacion_Proy3.keras')

# Convertir el campo PERIODO a string y luego a un tipo categórico con un orden específico
data['PERIODO'] = data['PERIODO'].astype(str)
ordered_periods = ["20191", "20192", "20201", "20211", "20221", "20222"]
data['PERIODO'] = pd.Categorical(data['PERIODO'], categories=ordered_periods, ordered=True)

y = data['PUNT_GLOBAL']

# Variables de entrada (solo las columnas especificadas)
columnas_X = ['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
              'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
              'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA',
              'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'ESTU_GENERO',
              'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
              'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
              'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
              'FAMI_TIENELAVADORA']

X = data[columnas_X]

# Definir rangos de puntajes y crear etiquetas de clasificación
bins = [0, 250, 350, float('inf')]
labels = [0, 1, 2]  # 0: Bajo, 1: Medio, 2: Alto
data['PUNT_GLOBAL_BIN'] = pd.cut(data['PUNT_GLOBAL'], bins=bins, labels=labels, right=False)

# Variable de salida
y = data['PUNT_GLOBAL_BIN']

# Columnas categóricas y numéricas
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

# Preprocesamiento: OneHotEncoding para variables categóricas, escalado para numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])

# Crear pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
pipeline.fit(X)
# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Layout de la aplicación
app.layout = html.Div([
    html.H1("Análisis de los Resultados de las Pruebas Saber 11 en Santander y Norte de Santander"),
    
    html.H2("Efecto del Entorno Familiar en el Puntaje Global"),
    html.P("Esta visualización permite analizar cómo diferentes características familiares afectan el puntaje global promedio de los estudiantes en las pruebas Saber 11."),
    html.Label("Seleccione el Departamento:"),
    dcc.Dropdown(
        id='department-dropdown',
        options=[
            {'label': 'Santander', 'value': 'SANTANDER'},
            {'label': 'Norte de Santander', 'value': 'NORTE SANTANDER'}
        ],
        value='SANTANDER'
    ),
    
    html.Label("Seleccione una Variable Familiar:"),
    dcc.Dropdown(
        id='fami-dropdown',
        options=[{'label': col, 'value': col} for col in [
            'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE',
            'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_TIENEAUTOMOVIL',
            'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA'
        ]],
        value='FAMI_EDUCACIONMADRE'
    ),
    
    dcc.Graph(id='bar-chart'),
    
    html.H2("Análisis de Características del Colegio en el Puntaje Global"),
    html.P("Esta visualización permite analizar cómo diferentes características del colegio afectan el puntaje global promedio de los estudiantes en las pruebas Saber 11."),
    
    html.Label("Seleccione el Departamento:"),
    dcc.Dropdown(
        id='policy-department-dropdown',
        options=[
            {'label': 'Santander', 'value': 'SANTANDER'},
            {'label': 'Norte de Santander', 'value': 'NORTE SANTANDER'}
        ],
        value='SANTANDER'
    ),
    
    html.Label("Seleccione una Característica del Colegio:"),
    dcc.Dropdown(
        id='policy-dropdown',
        options=[{'label': 'Género del Colegio', 'value': 'COLE_GENERO'},
                 {'label': 'Colegio Bilingüe', 'value': 'COLE_BILINGUE'},
                 {'label': 'Naturaleza del Colegio', 'value': 'COLE_NATURALEZA'},
                 {'label': 'Jornada del Colegio', 'value': 'COLE_JORNADA'},
                 {'label': 'Calendario del Colegio', 'value': 'COLE_CALENDARIO'}],
        value='COLE_GENERO'
    ),
    
    dcc.Graph(id='policy-bar-chart'),

    html.H2("Comparación Temporal de Puntajes entre Colegio y Departamento"),
    html.P("Esta visualización permite comparar la evolución de los puntajes globales promedio de un colegio específico con los del departamento seleccionado a lo largo del tiempo."),
    
    html.Label("Seleccione el Departamento para Comparar:"),
    dcc.Dropdown(
        id='comparison-department-dropdown',
        options=[
            {'label': 'Santander', 'value': 'SANTANDER'},
            {'label': 'Norte de Santander', 'value': 'NORTE SANTANDER'}
        ],
        value='SANTANDER'
    ),

    html.Label("Seleccione el Colegio:"),
    dcc.Dropdown(
        id='comparison-college-dropdown',
        options=[{'label': colegio, 'value': colegio} for colegio in data['COLE_NOMBRE_ESTABLECIMIENTO'].unique()],
        value='COL LA QUINTA DEL PUENTE'
    ),
    
    dcc.Graph(id='comparison-line-chart'),
    
    html.H2("Predicción del Rango del Puntaje Global"),
    html.P("Seleccione los valores de las variables para predecir el rango del puntaje global del estudiante."),
    
    html.Div(id='input-div', children=[
        html.Label("Periodo:"),
        dcc.Dropdown(id='PERIODO', options=[{'label': i, 'value': i} for i in ordered_periods], value=ordered_periods[0]),
        
        html.Label("Área de Ubicación:"),
        dcc.Dropdown(id='COLE_AREA_UBICACION', options=[{'label': i, 'value': i} for i in data['COLE_AREA_UBICACION'].unique()], value=data['COLE_AREA_UBICACION'].unique()[0]),
        
        html.Label("Bilingüe:"),
        dcc.Dropdown(id='COLE_BILINGUE', options=[{'label': i, 'value': i} for i in data['COLE_BILINGUE'].unique()], value=data['COLE_BILINGUE'].unique()[0]),
        
        html.Label("Calendario:"),
        dcc.Dropdown(id='COLE_CALENDARIO', options=[{'label': i, 'value': i} for i in data['COLE_CALENDARIO'].unique()], value=data['COLE_CALENDARIO'].unique()[0]),
        
        html.Label("Carácter:"),
        dcc.Dropdown(id='COLE_CARACTER', options=[{'label': i, 'value': i} for i in data['COLE_CARACTER'].unique()], value=data['COLE_CARACTER'].unique()[0]),
        
        html.Label("Departamento de Ubicación:"),
        dcc.Dropdown(id='COLE_DEPTO_UBICACION', options=[{'label': i, 'value': i} for i in data['COLE_DEPTO_UBICACION'].unique()], value=data['COLE_DEPTO_UBICACION'].unique()[0]),
        
        html.Label("Género:"),
        dcc.Dropdown(id='COLE_GENERO', options=[{'label': i, 'value': i} for i in data['COLE_GENERO'].unique()], value=data['COLE_GENERO'].unique()[0]),
        
        html.Label("Jornada:"),
        dcc.Dropdown(id='COLE_JORNADA', options=[{'label': i, 'value': i} for i in data['COLE_JORNADA'].unique()], value=data['COLE_JORNADA'].unique()[0]),
        
        html.Label("Municipio de Ubicación:"),
        dcc.Dropdown(id='COLE_MCPIO_UBICACION', options=[{'label': i, 'value': i} for i in data['COLE_MCPIO_UBICACION'].unique()], value=data['COLE_MCPIO_UBICACION'].unique()[0]),
        
        html.Label("Naturaleza:"),
        dcc.Dropdown(id='COLE_NATURALEZA', options=[{'label': i, 'value': i} for i in data['COLE_NATURALEZA'].unique()], value=data['COLE_NATURALEZA'].unique()[0]),
        
        html.Label("Nombre del Establecimiento:"),
        dcc.Dropdown(id='COLE_NOMBRE_ESTABLECIMIENTO', options=[{'label': i, 'value': i} for i in data['COLE_NOMBRE_ESTABLECIMIENTO'].unique()], value=data['COLE_NOMBRE_ESTABLECIMIENTO'].unique()[0]),
        
        html.Label("Nombre de la Sede:"),
        dcc.Dropdown(id='COLE_NOMBRE_SEDE', options=[{'label': i, 'value': i} for i in data['COLE_NOMBRE_SEDE'].unique()], value=data['COLE_NOMBRE_SEDE'].unique()[0]),
        
        html.Label("Género del Estudiante:"),
        dcc.Dropdown(id='ESTU_GENERO', options=[{'label': i, 'value': i} for i in data['ESTU_GENERO'].unique()], value=data['ESTU_GENERO'].unique()[0]),
        
        html.Label("Privado de Libertad:"),
        dcc.Dropdown(id='ESTU_PRIVADO_LIBERTAD', options=[{'label': i, 'value': i} for i in data['ESTU_PRIVADO_LIBERTAD'].unique()], value=data['ESTU_PRIVADO_LIBERTAD'].unique()[0]),
        
        html.Label("Número de Cuartos en el Hogar:"),
        dcc.Dropdown(id='FAMI_CUARTOSHOGAR', options=[{'label': i, 'value': i} for i in data['FAMI_CUARTOSHOGAR'].unique()], value=data['FAMI_CUARTOSHOGAR'].unique()[0]),
        
        html.Label("Educación de la Madre:"),
        dcc.Dropdown(id='FAMI_EDUCACIONMADRE', options=[{'label': i, 'value': i} for i in data['FAMI_EDUCACIONMADRE'].unique()], value=data['FAMI_EDUCACIONMADRE'].unique()[0]),
        
        html.Label("Educación del Padre:"),
        dcc.Dropdown(id='FAMI_EDUCACIONPADRE', options=[{'label': i, 'value': i} for i in data['FAMI_EDUCACIONPADRE'].unique()], value=data['FAMI_EDUCACIONPADRE'].unique()[0]),
        
        html.Label("Estrato de la Vivienda:"),
        dcc.Dropdown(id='FAMI_ESTRATOVIVIENDA', options=[{'label': i, 'value': i} for i in data['FAMI_ESTRATOVIVIENDA'].unique()], value=data['FAMI_ESTRATOVIVIENDA'].unique()[0]),
        
        html.Label("Número de Personas en el Hogar:"),
        dcc.Dropdown(id='FAMI_PERSONASHOGAR', options=[{'label': i, 'value': i} for i in data['FAMI_PERSONASHOGAR'].unique()], value=data['FAMI_PERSONASHOGAR'].unique()[0]),
        
        html.Label("Tiene Automóvil:"),
        dcc.Dropdown(id='FAMI_TIENEAUTOMOVIL', options=[{'label': i, 'value': i} for i in data['FAMI_TIENEAUTOMOVIL'].unique()], value=data['FAMI_TIENEAUTOMOVIL'].unique()[0]),
        
        html.Label("Tiene Computador:"),
        dcc.Dropdown(id='FAMI_TIENECOMPUTADOR', options=[{'label': i, 'value': i} for i in data['FAMI_TIENECOMPUTADOR'].unique()], value=data['FAMI_TIENECOMPUTADOR'].unique()[0]),
        
        html.Label("Tiene Internet:"),
        dcc.Dropdown(id='FAMI_TIENEINTERNET', options=[{'label': i, 'value': i} for i in data['FAMI_TIENEINTERNET'].unique()], value=data['FAMI_TIENEINTERNET'].unique()[0]),
        
        html.Label("Tiene Lavadora:"),
        dcc.Dropdown(id='FAMI_TIENELAVADORA', options=[{'label': i, 'value': i} for i in data['FAMI_TIENELAVADORA'].unique()], value=data['FAMI_TIENELAVADORA'].unique()[0]),
        
        html.Button('Predecir', id='predict-button', n_clicks=0),
    ]),
    
    html.Div(id='prediction-output')
])

# Callback para actualizar el gráfico de barras por entorno familiar
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('fami-dropdown', 'value'),
     Input('department-dropdown', 'value')]
)
def update_bar_chart(selected_variable, selected_department):
    filtered_df = data[data['COLE_DEPTO_UBICACION'] == selected_department]
    filtered_df = filtered_df.groupby(selected_variable)['PUNT_GLOBAL'].mean().reset_index()
    fig = px.bar(filtered_df, x=selected_variable, y='PUNT_GLOBAL',
                 labels={selected_variable: selected_variable, 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
                 title=f'Promedio del Puntaje Global por {selected_variable} en {selected_department}')
    return fig

# Callback para actualizar el gráfico de barras por características del colegio
@app.callback(
    Output('policy-bar-chart', 'figure'),
    [Input('policy-dropdown', 'value'),
     Input('policy-department-dropdown', 'value')]
)
def update_policy_bar_chart(selected_policy, selected_department):
    filtered_df = data[data['COLE_DEPTO_UBICACION'] == selected_department]
    filtered_df = filtered_df.groupby(selected_policy)['PUNT_GLOBAL'].mean().reset_index()
    fig = px.bar(filtered_df, x=selected_policy, y='PUNT_GLOBAL',
                 labels={selected_policy: selected_policy, 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
                 title=f'Promedio del Puntaje Global por {selected_policy} en {selected_department}')
    return fig

# Callback para actualizar el gráfico de comparación temporal
@app.callback(
    Output('comparison-line-chart', 'figure'),
    [Input('comparison-department-dropdown', 'value'),
     Input('comparison-college-dropdown', 'value')]
)
def update_comparison_line_chart(selected_department, selected_college):
    # Filtrar datos por departamento y colegio
    department_df = data[data['COLE_DEPTO_UBICACION'] == selected_department].groupby('PERIODO')['PUNT_GLOBAL'].mean().reset_index()
    college_df = data[data['COLE_NOMBRE_ESTABLECIMIENTO'] == selected_college].groupby('PERIODO')['PUNT_GLOBAL'].mean().reset_index()
    
    # Rellenar los periodos faltantes con NaN
    department_df = department_df.set_index('PERIODO').reindex(ordered_periods).reset_index()
    college_df = college_df.set_index('PERIODO').reindex(ordered_periods).reset_index()

    # Crear figura
    fig = px.line(department_df, x='PERIODO', y='PUNT_GLOBAL', markers=True, labels={'PERIODO': 'Periodo', 'PUNT_GLOBAL': 'Puntaje Global Promedio'}, title=f'Comparación de Puntajes entre {selected_college} y {selected_department}')
    fig.add_scatter(x=college_df['PERIODO'], y=college_df['PUNT_GLOBAL'], mode='lines+markers', name=selected_college, connectgaps=True)
    
    # Añadir la serie para el departamento
    fig.add_scatter(x=department_df['PERIODO'], y=department_df['PUNT_GLOBAL'], mode='lines+markers', name=selected_department, connectgaps=True)
    
    fig.update_layout(
        xaxis_title='Periodo',
        yaxis_title='Puntaje Global Promedio',
        legend_title='Entidad'
    )

    return fig

# Callback para realizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('PERIODO', 'value'), Input('COLE_AREA_UBICACION', 'value'), Input('COLE_BILINGUE', 'value'),
     Input('COLE_CALENDARIO', 'value'), Input('COLE_CARACTER', 'value'), Input('COLE_DEPTO_UBICACION', 'value'),
     Input('COLE_GENERO', 'value'), Input('COLE_JORNADA', 'value'), Input('COLE_MCPIO_UBICACION', 'value'),
     Input('COLE_NATURALEZA', 'value'), Input('COLE_NOMBRE_ESTABLECIMIENTO', 'value'), Input('COLE_NOMBRE_SEDE', 'value'),
     Input('ESTU_GENERO', 'value'), Input('ESTU_PRIVADO_LIBERTAD', 'value'), Input('FAMI_CUARTOSHOGAR', 'value'),
     Input('FAMI_EDUCACIONMADRE', 'value'), Input('FAMI_EDUCACIONPADRE', 'value'), Input('FAMI_ESTRATOVIVIENDA', 'value'),
     Input('FAMI_PERSONASHOGAR', 'value'), Input('FAMI_TIENEAUTOMOVIL', 'value'), Input('FAMI_TIENECOMPUTADOR', 'value'),
     Input('FAMI_TIENEINTERNET', 'value'), Input('FAMI_TIENELAVADORA', 'value')]
)
def make_prediction(n_clicks, periodo, cole_area_ubicacion, cole_bilingue, cole_calendario, cole_caracter,
                    cole_depto_ubicacion, cole_genero, cole_jornada, cole_mcpio_ubicacion, cole_naturaleza,
                    cole_nombre_establecimiento, cole_nombre_sede, estu_genero, estu_privado_libertad,
                    fami_cuartoshogar, fami_educacionmadre, fami_educacionpadre, fami_estratovivienda,
                    fami_personashogar, fami_tieneautomovil, fami_tienecomputador, fami_tieneinternet,
                    fami_tienelavadora):
    if n_clicks > 0:
        input_data = pd.DataFrame([[
            int(periodo), cole_area_ubicacion, cole_bilingue, cole_calendario, cole_caracter,
            cole_depto_ubicacion, cole_genero, cole_jornada, cole_mcpio_ubicacion, cole_naturaleza,
            cole_nombre_establecimiento, cole_nombre_sede, estu_genero, estu_privado_libertad,
            fami_cuartoshogar, fami_educacionmadre, fami_educacionpadre, fami_estratovivienda,
            fami_personashogar, fami_tieneautomovil, fami_tienecomputador, fami_tieneinternet,
            fami_tienelavadora
        ]], columns=[
            'PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
            'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
            'COLE_MCPIO_UBICACION', 'COLE_NATURALEZA', 'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE',
            'ESTU_GENERO', 'ESTU_PRIVADO_LIBERTAD', 'FAMI_CUARTOSHOGAR', 'FAMI_EDUCACIONMADRE',
            'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_TIENEAUTOMOVIL',
            'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA'
        ])
        
        # Aquí debes añadir cualquier otra transformación de preprocesamiento que hayas hecho durante el entrenamiento

        # Aplicar el mismo preprocesamiento que durante el entrenamiento
        input_data_transformed = pipeline.transform(input_data)
        input_data_transformed = np.append(input_data_transformed, [0])
        input_data_transformed = input_data_transformed.reshape(1, -1)
        # Verificar la forma del input_data
        print("Forma de los datos preprocesados:", input_data_transformed.shape)

        # Realizar la predicción
        prediction = model.predict(input_data_transformed)
        
        prediction_mapping = {0: "0: Bajo", 1: "1: Medio", 2: "2: Alto"}
        prediction_label = prediction_mapping[np.argmax(prediction)]

        return html.Div([
            html.P("El rango de puntaje global predicho es:", style={'font-size': '20px', 'font-weight': 'bold'}),
            html.P(prediction_label, style={'font-size': '20px'})
        ])
    return "Seleccione los valores y haga clic en 'Predecir'"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
