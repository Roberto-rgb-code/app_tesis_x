# Análisis de sentimientos en X para la comprensión multidimensional del mundo socio digital

## Descripción
Esta aplicación construida con Streamlit permite realizar un análisis integral de datos de una red social (X) para entender el comportamiento de métricas de interacción y el sentimiento de los textos. Ofrece funcionalidades de:

- Análisis Exploratorio de Datos (EDA)
- Modelos de regresión lineal y logística
- Árbol de decisión y Random Forest
- Clustering (K-Means) y clustering espacial con Folium
- Modelos de series temporales (ARIMA, SARIMAX)
- Análisis de Lenguaje Natural (sentimiento, nube de palabras)

Presentación y equipo:
- **Autor:** Lic. Kevin Roberto Torres Ruiz
- **Director:** Dr. Luis Alberto Maciel Arellano
- **Codirector:** Dr. Víctor Hugo Gualajara Estrada

## Requisitos
- Python 3.8+
- Pip para instalar dependencias

## Dependencias
```bash
streamlit
pandas
numpy
plotly
statsmodels
scipy
scikit-learn
matplotlib
seaborn
folium
streamlit-folium
textblob
wordcloud
```

## Instalación
1. Clona este repositorio:
   ```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
   ```
2. Crea y activa un entorno virtual (opcional pero recomendado):
   ```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
   ```
3. Instala las dependencias:
   ```bash
pip install -r requirements.txt
```

## Uso
1. Ejecuta la aplicación:
   ```bash
streamlit run app.py
```
2. En la interfaz, sube tu archivo JSON con columnas obligatorias:
   - `Likes`
   - `Retweets`
   - `Views`
   - `Replies`
   - `Timestamp`
   - (Opcional para NLP) `Text`
3. Selecciona el análisis deseado desde la barra lateral:
   - EDA
   - Regresión Lineal
   - Regresión Logística
   - Árbol de Decisión
   - Random Forest
   - Clustering
   - ARIMA
   - SARIMAX
   - Leaflet Clustering
   - NLP

Cada sección mostrará tablas, gráficos y resúmenes de estadísticas o modelos.

## Estructura del proyecto
```
├── app.py            # Código principal de Streamlit
├── requirements.txt  # Lista de dependencias
├── data/             # Carpeta opcional para JSON de ejemplo
└── README.md         # Documentación del proyecto
```

## Contribuciones
1. Haz un fork de este repositorio.
2. Crea una rama con tu nueva funcionalidad (`git checkout -b feature/nombre`).
3. Realiza tus cambios y haz commit (`git commit -m "Agrega nueva funcionalidad"`).
4. Sube la rama a tu fork (`git push origin feature/nombre`).
5. Abre un Pull Request.

## Licencia
Este proyecto está bajo la licencia MIT. Lee el archivo [LICENSE](LICENSE) para más detalles.

