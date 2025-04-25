import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller




# Configuración de la página
st.set_page_config(page_title="Clima Jena (2009–2016)", layout="wide")

# 🎨 Título principal
st.title("🌡️ Comparación y Optimización de Modelos de Series de Tiempo y Aprendizaje Automático")

# 📂 Sidebar de Navegación
seccion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ("Vista Previa Datos", 
     "Distribución Temperatura",  
     "Descomposición",
     "Análisis Estacionalidad",
     "Análisis de Tendencia",
     "Análisis de Estacionariedad")
)

# Función para cargar datos
@st.cache_data
def cargar_datos_reducido():
    df = pd.read_csv("jena_climate_2009_2016.csv")
    df = df[['Date Time', 'T (degC)']]
    df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True)
    return df


df_reducido = cargar_datos_reducido()

# 📊 Contenido según la sección elegida
if seccion == "Distribución Temperatura":
    st.header("📊 Distribución de la Temperatura del Aire 🌡️")

    fig = px.histogram(df_reducido, x='T (degC)', nbins=50, 
                       title='Distribución de la Temperatura',
                       labels={'T (degC)': 'Temperatura (°C)'},
                       color_discrete_sequence=['skyblue'])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    La forma de la distribución es aproximadamente normal, lo que sugiere que la temperatura varía de manera continua alrededor de un rango típico (5°C a 15°C).
    La presencia de valores negativos y extremos indica variaciones estacionales o eventos climáticos relevantes.
    """)

elif seccion == "Descomposición":
    st.header("🧩 Descomposición de la Serie de Tiempo")
    st.write("Visualización temporal de la serie descompuesta:")

    df_temp = df_reducido.copy()
    df_temp.set_index('Date Time', inplace=True)
    resultado = seasonal_decompose(df_temp['T (degC)'].dropna(), model='additive', period=365)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    resultado.observed.plot(ax=axes[0], title='Serie Original', color='black')
    resultado.trend.plot(ax=axes[1], title='Tendencia', color='blue')
    resultado.seasonal.plot(ax=axes[2], title='Estacionalidad', color='green')
    resultado.resid.plot(ax=axes[3], title='Ruido (Componente Aleatorio)', color='red')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
**Serie Original (Negro):** Patrón repetitivo con fluctuaciones estacionales anuales.

**Tendencia (Azul):** Ligera tendencia al alza a partir de 2013, posible indicio de calentamiento.

**Estacionalidad (Verde):** Ciclos anuales bien definidos con picos en verano y valles en invierno.

**Componente Aleatorio (Rojo):** Variaciones no explicadas, posiblemente eventos extremos.
    """)
elif seccion == "Análisis Estacionalidad":
    st.header("🖼️ Análisis visual de Estacionalidad Anual y Mensual")

    # Procesamiento de fechas
    df_reducido = df_reducido.copy()
    df_reducido['Year'] = df_reducido['Date Time'].dt.year
    df_reducido['DayOfYear'] = df_reducido['Date Time'].dt.dayofyear
    df_reducido['Month'] = df_reducido['Date Time'].dt.month

    # 🎛️ Selector de Años
    años_disponibles = sorted(df_reducido['Year'].unique())
    años_seleccionados = st.multiselect("Selecciona el/los año(s) a visualizar:", años_disponibles, default=[2009])

    if años_seleccionados:
        filtro_df = df_reducido[df_reducido['Year'].isin(años_seleccionados)]

        st.subheader("📅 Comparación de Estacionalidad Anual por Año Seleccionado")
        fig_multi_line = px.line(filtro_df, x='DayOfYear', y='T (degC)', color='Year',
                                 labels={'DayOfYear': 'Día del Año', 'T (degC)': 'Temperatura (°C)', 'Year': 'Año'},
                                 title='Comparación de la Estacionalidad Anual')
        st.plotly_chart(fig_multi_line, use_container_width=True)

    else:
        st.warning("Por favor, selecciona al menos un año para visualizar el gráfico.")

    # 📊 Boxplot por Mes (general)
    st.subheader("📦 Distribución de Temperatura por Mes (2009 - 2016)")
    fig_box = px.box(df_reducido, x='Month', y='T (degC)', 
                     labels={'Month': 'Mes', 'T (degC)': 'Temperatura (°C)'},
                     title='Estacionalidad Mensual de la Temperatura')
    fig_box.update_xaxes(tickmode='array', tickvals=list(range(1,13)),
                         ticktext=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
    st.plotly_chart(fig_box, use_container_width=True)

    # 📝 Análisis
    st.markdown("""
El gráfico permite comparar la **estacionalidad anual** entre diferentes años, identificando posibles variaciones o anomalías climáticas.

El boxplot mensual resalta la variabilidad típica de cada mes, siendo evidente el comportamiento estacional con temperaturas más altas en verano y más bajas en invierno.
    """)

elif seccion == "Análisis de Tendencia":
    st.header("📈 Análisis de Tendencia con Regresión Lineal")

    # Copia del dataframe y reset de índice
    df_tendencia = df_reducido.copy().reset_index()

    # 🎛️ Selector de rango de años
    año_min, año_max = int(df_tendencia['Date Time'].dt.year.min()), int(df_tendencia['Date Time'].dt.year.max())
    rango_años = st.slider("Selecciona el rango de años para analizar:", 
                           min_value=año_min, max_value=año_max, value=(año_min, año_max))

    # Filtrar por años seleccionados
    filtro = df_tendencia[(df_tendencia['Date Time'].dt.year >= rango_años[0]) & 
                          (df_tendencia['Date Time'].dt.year <= rango_años[1])].copy()

    if filtro.empty:
        st.warning("No hay datos para el rango seleccionado.")
    else:
        # Calcular días desde la fecha mínima del filtro
        filtro['Fecha_Num'] = (filtro['Date Time'] - filtro['Date Time'].min()).dt.days

        # Eliminar posibles NaN
        filtro = filtro.dropna(subset=['Fecha_Num', 'T (degC)'])

        # 📈 Gráfico de dispersión
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=filtro['Date Time'], 
            y=filtro['T (degC)'], 
            mode='markers',
            marker=dict(color='lightblue', opacity=0.4),
            name='Temperatura Observada'
        ))

        # 📉 Cálculo de regresión lineal
        coef = np.polyfit(filtro['Fecha_Num'], filtro['T (degC)'], 1)
        predicciones = np.poly1d(coef)(filtro['Fecha_Num'])

        fig.add_trace(go.Scatter(
            x=filtro['Date Time'], 
            y=predicciones, 
            mode='lines',
            line=dict(color='red'),
            name='Regresión Lineal'
        ))

        # Configuración del gráfico
        fig.update_layout(title=f"Tendencia de Temperatura ({rango_años[0]} - {rango_años[1]})",
                          xaxis_title="Fecha",
                          yaxis_title="Temperatura (°C)",
                          legend=dict(x=0.01, y=0.99),
                          template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        # 📊 Mostrar resultados de la regresión
        pendiente = coef[0]
        intercepto = coef[1]
        r2 = 1 - (np.sum((filtro['T (degC)'] - predicciones)**2) / np.sum((filtro['T (degC)'] - filtro['T (degC)'].mean())**2))

        st.markdown(f"""
        ### 📄 Resultados de la Regresión Lineal
        **Ecuación de la recta:**  
        `Temperatura = {pendiente:.4f} * Días + {intercepto:.2f}`

        - **Pendiente:** `{pendiente:.4f}` °C por día
        - **Intercepto:** `{intercepto:.2f}` °C
        - **R² (Coeficiente de Determinación):** `{r2:.4f}`

        > Una pendiente positiva sugiere una tendencia creciente de temperatura en el periodo seleccionado.
        """)
elif seccion == "Análisis de Estacionariedad":
    st.header("🔄 Análisis de Estacionariedad con Test de Dickey-Fuller")

    # 🎛️ Checkbox para aplicar diferenciación
    aplicar_diff = st.checkbox("Aplicar diferenciación (lag=1) para analizar estacionariedad", value=False)

    serie = df_reducido.copy()
    serie.set_index('Date Time', inplace=True)
    serie_temp = serie['T (degC)']

    if aplicar_diff:
        serie_temp = serie_temp.diff().dropna()
        st.info("Se aplicó diferenciación de primer orden (lag=1).")

    # 📈 Gráfico de la serie (original o diferenciada)
    st.subheader("Visualización de la Serie Temporal")
    fig = px.line(serie_temp, 
                  labels={'value': 'Temperatura (°C)', 'Date Time': 'Fecha'},
                  title="Serie Temporal " + ("Diferenciada" if aplicar_diff else "Original"))
    st.plotly_chart(fig, use_container_width=True)

    # 🎚️ Selector interactivo para número de lags
    st.subheader("Configuración del Test ADF")
    max_lag = st.slider("Selecciona el número máximo de lags para el test ADF:", min_value=1, max_value=50, value=20)

    # 📊 Aplicar Test ADF con control de lags
    resultado_adf = adfuller(serie_temp, maxlag=max_lag)

    estadistico = resultado_adf[0]
    p_valor = resultado_adf[1]
    valores_criticos = resultado_adf[4]

    st.subheader("📏 Resultado del Test de Dickey-Fuller Aumentado")
    st.write(f"**Estadístico ADF:** {estadistico:.4f}")
    st.write(f"**p-valor:** {p_valor:.4f}")

    st.write("**Valores Críticos:**")
    for clave, valor in valores_criticos.items():
        st.write(f"  - {clave}% : {valor:.4f}")

    # Interpretación automática
    if p_valor < 0.05:
        st.success("✅ La serie **ES estacionaria** (se rechaza la hipótesis nula).")
    else:
        st.error("⚠️ La serie **NO es estacionaria** (no se rechaza la hipótesis nula).")

    st.markdown("""
El test de Dickey-Fuller verifica si la serie tiene raíz unitaria (hipótesis nula).  
Si el **p-valor < 0.05**, podemos considerar la serie como estacionaria.

Puedes ajustar el número de lags si deseas optimizar el análisis según la longitud de la serie.
    """)