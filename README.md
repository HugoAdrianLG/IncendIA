# IncendIA – Sistema de Predicción de Incendios

Este proyecto escolar utiliza Machine Learning para predecir zonas de riesgo de incendio en el estado de Durango, México. El resultado se presenta en un mapa interactivo generado en HTML.

## 🔍 Descripción

El script `IncendIA.py` analiza datos climáticos y ambientales desde un archivo Excel (`Data_full.xlsx`) para predecir posibles focos de incendio utilizando modelos de aprendizaje automático.  
El resultado se visualiza en un mapa generado automáticamente (`Mapa_de_predicciones.html`), el cual puede abrirse directamente desde cualquier navegador web.

## ⚙️ Tecnologías y librerías utilizadas

- Python 3.x
- pandas
- numpy
- matplotlib
- folium
- scikit-learn
- openpyxl

## 🧪 Cómo usar

1. Asegúrate de tener Python instalado.
2. Instala las dependencias necesarias (puedes usar un entorno virtual si deseas):
   pip install pandas numpy matplotlib folium scikit-learn openpyxl
3.Ejecuta el script:
   python IncendIA.py
4.Abre el archivo Mapa_de_predicciones.html en tu navegador para visualizar el mapa generado.
