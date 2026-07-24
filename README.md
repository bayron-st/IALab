# IALab — Predicción del tiempo de análisis de muestras de laboratorio

Proyecto del taller *Diseño de Algoritmos de Aprendizaje Profundo* (Inteligencia Artificial Avanzada). Una red neuronal con embeddings de variables categóricas predice el tiempo total (en horas) que tomará analizar una muestra de agua o suelo en un laboratorio ambiental colombiano, a partir de la información disponible al momento de su recepción.

## Contenido

| Archivo | Descripción |
|---|---|
| `generar_dataset.py` | Genera el dataset sintético (`dataset_laboratorio_muestras.csv` / `.xlsx`), 2200 muestras. |
| `entrenamiento_numpy.py` | Implementación de referencia en NumPy puro (sin dependencias pesadas). |
| `modelo_pytorch.py` | Modelo oficial en PyTorch. Entrena y guarda `modelo_turnaround_time.pt` + `preprocesador.json` + `metricas_modelo_pytorch.json`. |
| `app.py` | Interfaz Streamlit: panel de resultados, formulario para estimar el tiempo de una muestra nueva, e historial de predicciones. |
| `reentrenar_con_log.py` | Combina las predicciones ya validadas con tiempo real (`predicciones_log.csv`) con el dataset base, para reentrenar con datos reales. |
| `Instructivo_Ejecucion_Codigo_y_Colab.docx` | Guía paso a paso para ejecutar todo localmente o en Google Colab. |

## Instalación

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Entrenar el modelo

```bash
python generar_dataset.py                 # opcional, ya viene el CSV generado
python modelo_pytorch.py --data dataset_laboratorio_muestras.csv --epochs 600
```

Esto produce (o actualiza) `modelo_turnaround_time.pt`, `preprocesador.json` y `metricas_modelo_pytorch.json`, que la app necesita para funcionar.

## Ejecutar la interfaz

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501` con tres pestañas:

- **Estimar una muestra nueva**: formulario con el tipo de muestra, grupo analítico, prioridad, sede, carga del laboratorio, etc. Devuelve el tiempo estimado, un rango (± MAE del modelo) y si la muestra está en riesgo de exceder el plazo comprometido. Cada consulta queda registrada en `predicciones_log.csv`.
- **Resultados del modelo**: métricas (MAE, RMSE, R², MAPE) y las figuras generadas durante el entrenamiento.
- **Historial de predicciones**: consulta y descarga del log acumulado, con el conteo de predicciones validadas y pendientes.

## Cerrar el ciclo con datos reales

El dataset base es sintético y no cambia solo, y el modelo tampoco aprende de las consultas de la app en tiempo real. Para que sí lo haga:

1. Cuando el informe de una muestra consultada en la app ya se entregó, abre `predicciones_log.csv` y completa manualmente la columna `tiempo_real_horas` (horas reales que tomó) en la fila correspondiente.
2. Ejecuta `python reentrenar_con_log.py`: combina las filas ya validadas con el dataset sintético base y genera `dataset_laboratorio_muestras_actualizado.csv`.
3. Reentrena con `python modelo_pytorch.py --data dataset_laboratorio_muestras_actualizado.csv --epochs 600` (esto sobrescribe `modelo_turnaround_time.pt`, `preprocesador.json` y las métricas).

## Desplegar en Streamlit Community Cloud (opcional)

1. Sube este repositorio a GitHub (incluyendo `modelo_turnaround_time.pt` y `preprocesador.json`).
2. Entra a [share.streamlit.io](https://share.streamlit.io), conecta tu cuenta de GitHub.
3. Selecciona el repositorio, la rama y `app.py` como archivo principal.
4. Streamlit instala `requirements.txt` automáticamente y publica una URL pública.
