import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import re
from textblob import TextBlob
from wordcloud import WordCloud

# Título de la aplicación
st.title("Análisis de sentimientos en X para la comprensión multidimensional del mundo socio digital")
st.subheader("Presenta: Lic. Kevin Roberto Torres Ruiz")
st.subheader("Director: Dr. Luis Alberto Maciel Arellano")
st.subheader("Codirector: Dr. Víctor Hugo Gualajara Estrada")

# Carga de datos
data_file = st.file_uploader("Sube tu archivo JSON (ej. LagosDeMoreno 8623.JSON)", type=["json"])
if not data_file:
    st.warning("Por favor, sube un archivo JSON para continuar.")
    st.stop()

try:
    data = pd.read_json(data_file)
except ValueError:
    st.error("Error al leer el archivo JSON. Asegúrate de que tenga la estructura correcta.")
    st.stop()

# Verificar columnas necesarias
required_columns = ['Likes', 'Retweets', 'Views', 'Replies', 'Timestamp']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Faltan las siguientes columnas en el archivo: {', '.join(missing_columns)}")
    st.stop()

# Variables globales
y = data['Likes']
X = data[['Retweets', 'Views', 'Replies']]

# Menú de navegación en la barra lateral
option = st.sidebar.selectbox("Seleccione análisis", [
    'EDA', 'Regresión Lineal', 'Regresión Logística', 
    'Árbol de Decisión', 'Random Forest', 'Clustering', 
    'ARIMA', 'SARIMAX', 'Leaflet Clustering', 'NLP'
])

# **EDA (Análisis Exploratorio de Datos)**
if option == 'EDA':
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas Descriptivas")
    df = data[['Retweets', 'Views', 'Replies', 'Likes']]
    desc_stats = df.describe()
    extra_stats = pd.DataFrame({
        "Variance": df.var(),
        "Skewness": df.skew(),
        "Kurtosis": df.kurt(),
        "Median": df.median()
    }).T
    complete_stats = pd.concat([desc_stats, extra_stats])
    st.dataframe(complete_stats)

    # Gráficos de dispersión
    st.subheader("Gráficos de Dispersión")
    fig0 = px.scatter(data, x='Retweets', y='Likes', title='Likes vs Retweets')
    st.plotly_chart(fig0)

    fig1 = px.scatter(data, x='Views', y='Likes', title='Likes vs Views')
    st.plotly_chart(fig1)

    fig2 = px.scatter(data, x='Replies', y='Likes', title='Likes vs Replies')
    st.plotly_chart(fig2)

    fig3 = px.scatter_3d(data, x='Retweets', y='Views', z='Likes', color='Likes', title='Likes vs Retweets vs Views')
    st.plotly_chart(fig3)

    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    corr = df.corr()
    st.write(corr)

# **Regresión Lineal**
elif option == 'Regresión Lineal':
    st.header("Regresión Lineal (OLS)")
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    st.text(ols.summary())

    resid = ols.resid
    fitted = ols.fittedvalues
    st.subheader("Verificación de Supuestos")

    # 1. Linealidad
    st.write("**1. Linealidad**")
    st.write("Gráficos de Dispersión para Linealidad")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(data['Retweets'], data['Likes'], alpha=0.5)
    axes[0].set_title('Likes vs Retweets')
    axes[1].scatter(data['Views'], data['Likes'], alpha=0.5)
    axes[1].set_title('Likes vs Views')
    axes[2].scatter(data['Replies'], data['Likes'], alpha=0.5)
    axes[2].set_title('Likes vs Replies')
    st.pyplot(fig)

    # 2. Independencia (Durbin-Watson)
    st.write("**2. Independencia**")
    dw = sm.stats.stattools.durbin_watson(resid)
    st.write(f"Durbin-Watson stat: {dw:.3f}")

    # 3. Homocedasticidad
    st.write("**3. Homocedasticidad**")
    bp_p = het_breuschpagan(resid, Xc)[1]
    st.write(f"Breusch-Pagan p-value: {bp_p:.3f}")

    # 4. Normalidad
    st.write("**4. Normalidad**")
    st.write("Histograma y Q-Q Plot de Residuos")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(resid, kde=True, ax=ax[0])
    ax[0].set_title('Histograma de Residuos')
    sm.qqplot(resid, line='s', ax=ax[1])
    ax[1].set_title('Q-Q Plot')  # Corrección: establecer el título en el eje
    st.pyplot(fig)
    jb_p = jarque_bera(resid)[1]
    st.write(f"Jarque-Bera p-value: {jb_p:.3f}")

    # 5. No Multicolinealidad
    st.write("**5. No Multicolinealidad**")
    vif = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    st.write(pd.DataFrame({'Variable': Xc.columns, 'VIF': vif}))

    # Resumen de supuestos
    st.subheader("Resumen de Supuestos")
    st.write(f"- **Linealidad**: Verifica los gráficos de dispersión.")
    st.write(f"- **Independencia**: Durbin-Watson = {dw:.3f} (debe estar cerca de 2).")
    st.write(f"- **Homocedasticidad**: Breusch-Pagan p-value = {bp_p:.3f} (debe ser > 0.05).")
    st.write(f"- **Normalidad**: Jarque-Bera p-value = {jb_p:.3f} (debe ser > 0.05).")
    st.write(f"- **No Multicolinealidad**: VIF < 10 para todas las variables.")

    
# **Regresión Logística**
elif option == 'Regresión Logística':
    st.header("Regresión Logística")
    mediana_likes = data['Likes'].median()
    ybin = (y > mediana_likes).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, ybin, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    Xtr_s_const = sm.add_constant(Xtr_s)
    logit_model = sm.Logit(ytr, Xtr_s_const).fit()
    st.subheader("Summary del Modelo")
    st.text(logit_model.summary())

    logit = LogisticRegression().fit(Xtr_s, ytr)
    pred = logit.predict(Xte_s)
    proba = logit.predict_proba(Xte_s)[:,1]

    st.subheader("Coeficientes")
    coef_df = pd.DataFrame({'Variable': X.columns, 'Coef': logit.coef_[0], 'Odds Ratio': np.exp(logit.coef_[0])})
    st.write(coef_df)

    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Matriz de Confusión')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    st.pyplot(fig)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(yte, proba)
    auc_val = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f"ROC AUC={auc_val:.3f}", labels={'x':'FPR','y':'TPR'})
    st.plotly_chart(fig)

    st.subheader("Distribución de Probabilidades Predichas")
    fig, ax = plt.subplots()
    ax.hist(proba, bins=50)
    ax.set_title('Distribución de Probabilidades Predichas')
    ax.set_xlabel('Probabilidad de Éxito')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

# **Árbol de Decisión**
elif option == 'Árbol de Decisión':
    st.header("Árbol de Decisión")
    mediana_likes = data['Likes'].median()
    ybin = (y > mediana_likes).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, ybin, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xtr_s, ytr)
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, class_names=['No Exitoso', 'Exitoso'], filled=True, ax=ax)
    st.pyplot(fig)

# **Random Forest**
elif option == 'Random Forest':
    st.header("Random Forest")
    mediana_likes = data['Likes'].median()
    ybin = (y > mediana_likes).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, ybin, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(Xtr_s, ytr)
    pred = rf.predict(Xte_s)
    proba = rf.predict_proba(Xte_s)[:,1]

    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Matriz de Confusión')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    st.pyplot(fig)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(yte, proba)
    auc_val = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f"ROC AUC={auc_val:.3f}", labels={'x':'FPR','y':'TPR'})
    st.plotly_chart(fig)

    st.subheader("Importancia de Características")
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    st.bar_chart(imp)

    st.subheader("Visualización del Primer Árbol")
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(rf.estimators_[0], feature_names=X.columns, class_names=['Bajo', 'Alto'], filled=True, max_depth=3, ax=ax)
    st.pyplot(fig)

# **Clustering (K-Means)**
elif option == 'Clustering':
    st.header("Clustering con K-Means")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    ks = range(2,11)
    inert, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(Xs)
        inert.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(ks, inert, 'bx-')
    ax[0].set_title('Método del Codo')
    ax[1].plot(ks, sil, 'rx-')
    ax[1].set_title('Silhouette Score')
    st.pyplot(fig)

    n_clusters = st.sidebar.slider("Número de Clusters", 2, 10, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xs)
    data['Cluster'] = km.labels_

    st.subheader("Visualización de Clusters")
    fig = px.scatter_3d(data, x='Retweets', y='Views', z='Replies', color='Cluster', title='Clusters en 3D')
    st.plotly_chart(fig)

# **ARIMA**
elif option == 'ARIMA':
    st.header("ARIMA sobre Likes")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    ts = data.set_index('Timestamp')['Likes'].resample('D').sum().fillna(0)
    
    st.subheader("Prueba de Dickey-Fuller")
    result = adfuller(ts)
    st.write(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    
    st.subheader("Autocorrelación")
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    plot_acf(ts, ax=ax[0])
    plot_pacf(ts, ax=ax[1])
    st.pyplot(fig)
    
    m = ARIMA(ts, order=(1,1,1)).fit()
    st.subheader("Summary del Modelo")
    st.text(m.summary())
    
    forecast = m.get_forecast(30)
    f_mean = forecast.predicted_mean
    f_ci = forecast.conf_int()
    
    st.subheader("Pronóstico a 30 días")
    fig = px.line(ts, title='Serie Temporal de Likes')
    fig.add_scatter(x=f_mean.index, y=f_mean, name='Pronóstico', line=dict(color='red'))
    fig.add_scatter(x=f_ci.index, y=f_ci.iloc[:,0], name='CI Inferior', line=dict(dash='dash'))
    fig.add_scatter(x=f_ci.index, y=f_ci.iloc[:,1], name='CI Superior', line=dict(dash='dash'))
    st.plotly_chart(fig)

# **SARIMAX**
elif option == 'SARIMAX':
    st.header("SARIMAX sobre Likes")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    df = data.set_index('Timestamp').resample('D').sum(numeric_only=True).fillna(0)
    endog = df['Likes']
    exog = df[['Retweets', 'Views', 'Replies']]
    
    # Eliminar filas con NaN o infinitos en exog
    exog = exog.replace([np.inf, -np.inf], np.nan).dropna()
    endog = endog.loc[exog.index]  # Alinear endog con exog
    
    st.subheader("Prueba de Dickey-Fuller")
    result = adfuller(endog)
    st.write(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    
    st.subheader("Autocorrelación")
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    plot_acf(endog, ax=ax[0])
    plot_pacf(endog, ax=ax[1])
    st.pyplot(fig)
    
    m = SARIMAX(endog, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    st.subheader("Summary del Modelo")
    st.text(m.summary())
    
    # Generar exog_future usando la media de los últimos 30 días
    exog_future = pd.DataFrame(index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D'))
    for col in exog.columns:
        exog_future[col] = exog[col].tail(30).mean()
    
    forecast = m.get_forecast(steps=30, exog=exog_future)
    f_mean = forecast.predicted_mean
    f_ci = forecast.conf_int()
    
    st.subheader("Pronóstico a 30 días")
    fig = px.line(endog, title='Serie Temporal de Likes')
    fig.add_scatter(x=f_mean.index, y=f_mean, name='Pronóstico', line=dict(color='red'))
    fig.add_scatter(x=f_ci.index, y=f_ci.iloc[:,0], name='CI Inferior', line=dict(dash='dash'))
    fig.add_scatter(x=f_ci.index, y=f_ci.iloc[:,1], name='CI Superior', line=dict(dash='dash'))
    st.plotly_chart(fig)

# **Clustering Espacial con Folium**
elif option == 'Leaflet Clustering':
    st.header("Clustering Espacial con Folium")
    
    # Generar coordenadas simuladas dentro de los límites de la ZMG
    lat_min, lat_max = 20.5, 20.8  # Rango de latitud
    lon_min, lon_max = -103.5, -103.2  # Rango de longitud
    n_points = 1000  # Número de puntos simulados
    lats = np.random.uniform(lat_min, lat_max, n_points)
    lons = np.random.uniform(lon_min, lon_max, n_points)
    coords = pd.DataFrame({'Latitude': lats, 'Longitude': lons})

    # Permitir al usuario seleccionar el número de clusters
    k = st.sidebar.slider("Clusters", 2, 10, 5)
    
    # Aplicar el algoritmo K-Means para clustering
    km = KMeans(n_clusters=k, random_state=42).fit(coords)
    coords['Cluster'] = km.labels_

    # Crear un mapa centrado en la ZMG
    m = folium.Map(location=[20.65, -103.35], zoom_start=11)
    
    # Lista de colores para los clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    # Añadir marcadores al mapa
    for idx, row in coords.iterrows():
        cluster_num = int(row['Cluster'])  # Convertir a entero
        # Ciclar colores si el número de clusters excede la cantidad de colores
        color = colors[cluster_num % len(colors)]
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    # Mostrar el mapa en Streamlit
    folium_static(m)

# **Procesamiento de Lenguaje Natural (NLP)**
elif option == 'NLP':
    st.header("Análisis de Lenguaje Natural (NLP)")
    
    # Verificar si la columna 'Text' existe
    if 'Text' not in data.columns:
        st.error("La columna 'Text' no está presente en los datos.")
    else:
        # Función para limpiar texto
        def clean_text(text):
            text = re.sub(r'http\S+', '', text)  # Eliminar URLs
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
            text = text.lower()  # Convertir a minúsculas
            return text

        # Preprocesar el texto
        data['Cleaned_Text'] = data['Text'].apply(clean_text)
        
        # Análisis de sentimiento
        st.subheader("Análisis de Sentimiento")
        data['Sentiment'] = data['Cleaned_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['Sentiment_Label'] = data['Sentiment'].apply(lambda x: 'Positivo' if x > 0 else ('Negativo' if x < 0 else 'Neutral'))
        
        # Visualización de la distribución de sentimientos
        sentiment_counts = data['Sentiment_Label'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        ax.set_xlabel('Sentimiento')
        ax.set_ylabel('Número de Textos')
        ax.set_title('Distribución de Sentimientos')
        st.pyplot(fig)
        
        # Nube de palabras
        st.subheader("Nube de Palabras")
        all_words = ' '.join(data['Cleaned_Text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)