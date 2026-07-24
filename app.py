"""
Interfaz Streamlit del proyecto "Predicción del tiempo de análisis de muestras
de agua y suelo". Tres vistas:

  1. "Estimar una muestra nueva": formulario de captura -> estimación de horas
     usando el modelo entrenado en PyTorch (modelo_pytorch.py). Cada consulta
     se registra en predicciones_log.csv.
  2. "Resultados del modelo": métricas y figuras generadas en el entrenamiento.
  3. "Historial de predicciones": consulta y descarga del log acumulado, para
     validarlo con el tiempo real observado y reentrenar (ver reentrenar_con_log.py).

Requiere que ya existan en esta misma carpeta (generados por modelo_pytorch.py):
  - modelo_turnaround_time.pt
  - preprocesador.json
  - metricas_modelo_pytorch.json (o metricas_modelo.json como respaldo)

Ejecutar con:
    streamlit run app.py
"""
import csv
import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch

from modelo_pytorch import TurnaroundTimeNet

LOG_PATH = "predicciones_log.csv"
LOG_COLUMNS = [
    "timestamp_consulta", "tipo_muestra", "subtipo_muestra", "grupo_analitico",
    "num_parametros_solicitados", "prioridad_cliente", "sector_cliente", "sede_laboratorio",
    "carga_laboratorio_dia", "num_muestras_lote", "es_fin_de_semana", "temporada_alta_demanda",
    "mes_recepcion", "fecha_recepcion", "plazo_comprometido_horas", "tiempo_estimado_horas",
    "tiempo_real_horas", "estado_entrega_real",
]


def registrar_prediccion(fila: dict) -> None:
    """Anexa una consulta al log de predicciones (crea el archivo con encabezado
    si todavía no existe). 'tiempo_real_horas' y 'estado_entrega_real' quedan
    vacíos: se completan manualmente después, cuando se conozca el resultado
    real de esa muestra, para poder usarlos en reentrenar_con_log.py."""
    existe = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not existe:
            writer.writeheader()
        writer.writerow({**{"tiempo_real_horas": "", "estado_entrega_real": ""}, **fila})

st.set_page_config(page_title="Laboratorio · Predicción de tiempos", layout="wide")

# ---------------------------------------------------------------------------
# Catálogos de referencia — deben coincidir con generar_dataset.py
# ---------------------------------------------------------------------------
SUBTIPOS = {
    "Agua": ["Potable", "Residual/Vertimiento", "Superficial", "Subterránea"],
    "Suelo": ["Agrícola", "Contaminado/Industrial", "Sedimento", "Relleno Sanitario"],
}
GRUPO_POR_TIPO = {
    "Agua": ["Fisicoquimico_Basico", "Microbiologico", "Metales_Pesados_ICP",
             "DBO5_DQO", "Plaguicidas_Cromatografia"],
    "Suelo": ["Textura_Fertilidad_Suelo", "Metales_Pesados_ICP",
              "Hidrocarburos_HTP", "Microbiologico"],
}
PRIORIDADES = ["Normal", "Urgente", "Express"]
SECTORES = ["Industrial", "Agricola", "Consultoria_Ambiental", "Entidad_Publica", "Minero_Energetico"]
SEDES = ["Bogota", "Medellin", "Cali", "Barranquilla"]
PLAZO_HORAS = {"Normal": 96, "Urgente": 72, "Express": 48}


@st.cache_resource
def cargar_modelo():
    if not (os.path.exists("preprocesador.json") and os.path.exists("modelo_turnaround_time.pt")):
        return None, None
    with open("preprocesador.json", encoding="utf-8") as f:
        prep = json.load(f)
    model = TurnaroundTimeNet(prep["cardinalities"], prep["emb_dims"], n_num=len(prep["NUM_COLS"]))
    state = torch.load("modelo_turnaround_time.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, prep


@st.cache_data
def cargar_metricas():
    for fname in ["metricas_modelo_pytorch.json", "metricas_modelo.json"]:
        if os.path.exists(fname):
            with open(fname, encoding="utf-8") as f:
                return json.load(f), fname
    return None, None


def predecir(model, prep, muestra: dict) -> float:
    """Traduce un diccionario de valores 'crudos' (texto/número) exactamente
    igual que en entrenamiento (mismos mapeos e igual estandarización) y
    ejecuta el forward pass del modelo. Devuelve horas en escala original."""
    horas, _ = predecir_con_trazas(model, prep, muestra)
    return horas


# Nombre legible de cada módulo dentro de model.mlp (ver TurnaroundTimeNet en
# modelo_pytorch.py): Linear(→128), ReLU, Dropout, Linear(→64), ReLU, Dropout,
# Linear(→32), ReLU, Linear(→1). El orden debe coincidir exactamente con el
# nn.Sequential definido allí.
NOMBRES_CAPAS_MLP = [
    "Linear → 128", "ReLU", "Dropout",
    "Linear → 64", "ReLU", "Dropout",
    "Linear → 32", "ReLU",
    "Linear → 1 (salida)",
]


def predecir_con_trazas(model, prep, muestra: dict):
    """Igual que predecir(), pero además captura —con forward hooks reales,
    no simulados— lo que ocurre dentro de la red durante ESTA consulta
    puntual: a qué índice de embedding se mapeó cada categoría, el z-score de
    cada numérica, y la activación de cada capa del MLP. Pensado para el panel
    'qué hace el modelo en background' de la interfaz."""
    categoricas = []
    cat_idx = []
    for c in prep["CAT_COLS"]:
        mapping = prep["cat_maps"][c]
        val = muestra[c]
        if val not in mapping:
            val = list(mapping.keys())[0]  # respaldo si aparece una categoría nunca vista
        idx = mapping[val]
        cat_idx.append(idx)
        vector = model.embeddings[c].weight[idx].detach().numpy().round(3).tolist()
        categoricas.append({"variable": c, "valor": val, "indice": idx, "embedding": vector})

    numericas = []
    num_vals = []
    for c in prep["NUM_COLS"]:
        mean, std = prep["num_mean"][c], prep["num_std"][c]
        z = (muestra[c] - mean) / (std if std else 1)
        num_vals.append(z)
        numericas.append({"variable": c, "valor": muestra[c], "media": mean, "desviacion": std, "z_score": z})

    xc = torch.tensor([cat_idx], dtype=torch.long)
    xn = torch.tensor([num_vals], dtype=torch.float32)

    capas = []

    def registrar_capa(nombre):
        def hook(_modulo, _entrada, salida):
            arr = salida.detach().numpy().reshape(-1)
            capas.append({
                "capa": nombre,
                "dimension": int(arr.shape[0]),
                "activacion_media": float(arr.mean()),
                "norma_l2": float(np.linalg.norm(arr)),
            })
        return hook

    hooks = [capa.register_forward_hook(registrar_capa(nombre))
             for capa, nombre in zip(model.mlp, NOMBRES_CAPAS_MLP)]
    with torch.no_grad():
        pred_log = model(xc, xn).item()
    for h in hooks:
        h.remove()

    horas = float(np.expm1(pred_log))
    input_dim = sum(len(c["embedding"]) for c in categoricas) + len(numericas)

    trazas = {
        "categoricas": categoricas,
        "numericas": numericas,
        "input_dim": input_dim,
        "capas": capas,
        "salida_log1p": pred_log,
        "salida_horas": horas,
    }
    return horas, trazas


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("Predicción del tiempo de análisis de muestras (agua y suelo)")
st.caption("Red neuronal con embeddings categóricos — proyecto de Diseño de Algoritmos de Aprendizaje Profundo")

tab_pred, tab_resultados, tab_historial = st.tabs(
    ["Estimar una muestra nueva", "Resultados del modelo", "Historial de predicciones"]
)

# ---------------------------------------------------------------------------
# Tab 1: formulario de predicción
# ---------------------------------------------------------------------------
with tab_pred:
    model, prep = cargar_modelo()
    if model is None:
        st.warning(
            "No se encontraron 'modelo_turnaround_time.pt' y/o 'preprocesador.json' en esta carpeta. "
            "Ejecuta primero: `python modelo_pytorch.py --data dataset_laboratorio_muestras.csv`"
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            tipo = st.selectbox("Tipo de muestra", ["Agua", "Suelo"])
            subtipo = st.selectbox("Subtipo de muestra", SUBTIPOS[tipo])
            grupo = st.selectbox("Grupo analítico solicitado", GRUPO_POR_TIPO[tipo])
            num_parametros = st.number_input("Número de parámetros solicitados", min_value=1, max_value=20, value=5)
            prioridad = st.selectbox("Prioridad del cliente", PRIORIDADES)
        with col2:
            sector = st.selectbox("Sector del cliente", SECTORES)
            sede = st.selectbox("Sede del laboratorio", SEDES)
            carga = st.number_input("Muestras en cola ese día (carga del laboratorio)", min_value=0, max_value=60, value=8)
            lote = st.number_input("Tamaño del lote recibido junto con la muestra", min_value=1, max_value=20, value=1)
            fecha_recepcion = st.date_input("Fecha de recepción", value=date.today())

        if st.button("Estimar tiempo de análisis", type="primary"):
            es_fin_de_semana = int(fecha_recepcion.weekday() >= 5)
            temporada_alta = int(fecha_recepcion.month in (1, 2, 3, 7, 8, 9))
            mes_sin = float(np.sin(2 * np.pi * fecha_recepcion.month / 12))
            mes_cos = float(np.cos(2 * np.pi * fecha_recepcion.month / 12))

            muestra = {
                "tipo_muestra": tipo, "subtipo_muestra": subtipo, "grupo_analitico": grupo,
                "prioridad_cliente": prioridad, "sector_cliente": sector, "sede_laboratorio": sede,
                "num_parametros_solicitados": float(num_parametros), "carga_laboratorio_dia": float(carga),
                "num_muestras_lote": float(lote), "es_fin_de_semana": float(es_fin_de_semana),
                "temporada_alta_demanda": float(temporada_alta), "mes_sin": mes_sin, "mes_cos": mes_cos,
            }
            horas, trazas = predecir_con_trazas(model, prep, muestra)

            metricas, _ = cargar_metricas()
            mae = metricas["test"]["MAE_horas"] if metricas else None

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Tiempo estimado", f"{horas:.0f} horas")
            if mae:
                c2.metric("Rango esperado (± MAE)", f"{max(horas - mae, 0):.0f} – {horas + mae:.0f} h")

            plazo = PLAZO_HORAS[prioridad]
            if grupo == "DBO5_DQO":
                plazo = max(plazo, 144)
            fuera_de_plazo = horas > plazo
            c3.metric("Plazo comprometido", f"{plazo} h",
                      delta="Riesgo de retraso" if fuera_de_plazo else "Dentro del plazo",
                      delta_color="inverse" if fuera_de_plazo else "normal")

            entrega = pd.Timestamp(fecha_recepcion) + pd.Timedelta(hours=horas)
            st.caption(f"Equivalente aproximado: {horas / 24:.1f} días · Fecha estimada de entrega: {entrega.strftime('%Y-%m-%d %H:%M')}")

            if grupo == "DBO5_DQO":
                st.info(
                    "Este grupo analítico incluye DBO5, con incubación regulatoria mínima de 120 horas "
                    "(5 días): ninguna optimización de proceso puede reducir ese piso normativo."
                )

            with st.expander("Ver el cálculo paso a paso (qué hace el modelo en background)"):
                st.markdown("**1. Variables categóricas → índice de embedding aprendido**")
                st.caption("Cada categoría se traduce al índice que tuvo en entrenamiento, y ese índice busca su vector de embedding dentro de la red (pesos reales del modelo cargado).")
                st.dataframe(
                    pd.DataFrame([{
                        "Variable": c["variable"], "Valor ingresado": c["valor"], "Índice": c["indice"],
                        "Vector de embedding (pesos reales)": ", ".join(f"{v:.3f}" for v in c["embedding"]),
                    } for c in trazas["categoricas"]]),
                    use_container_width=True, hide_index=True,
                )

                st.markdown("**2. Variables numéricas → estandarización (z-score)**")
                st.caption("z = (valor − media de entrenamiento) / desviación estándar de entrenamiento — los mismos parámetros guardados en preprocesador.json.")
                st.dataframe(
                    pd.DataFrame([{
                        "Variable": n["variable"], "Valor ingresado": round(n["valor"], 2),
                        "Media (entrenamiento)": round(n["media"], 2),
                        "Desv. estándar (entrenamiento)": round(n["desviacion"], 2),
                        "Valor estandarizado (z)": round(n["z_score"], 3),
                    } for n in trazas["numericas"]]),
                    use_container_width=True, hide_index=True,
                )

                st.markdown(f"**3. Vector de entrada a la red:** {trazas['input_dim']} dimensiones (embeddings concatenados + numéricas estandarizadas)")

                st.markdown("**4. Activación real de cada capa del MLP para esta consulta**")
                st.caption("Capturado en vivo con forward hooks sobre el modelo cargado — no son valores de ejemplo.")
                df_capas = pd.DataFrame(trazas["capas"])
                st.dataframe(
                    df_capas.rename(columns={
                        "capa": "Capa", "dimension": "Dimensión de salida",
                        "activacion_media": "Activación media", "norma_l2": "Norma L2",
                    }).style.format({"Activación media": "{:.3f}", "Norma L2": "{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.bar_chart(df_capas.set_index("capa")["norma_l2"], height=220)
                st.caption("Norma L2 de la señal a la salida de cada capa: muestra cómo se va transformando conforme se reduce de 128 → 64 → 32 → 1 neurona.")

                st.markdown(
                    f"**5. Salida de la red y transformación inversa:** el modelo produce "
                    f"`{trazas['salida_log1p']:.4f}` en escala log1p(horas). Aplicando "
                    f"`expm1({trazas['salida_log1p']:.4f})` se obtiene la estimación final: "
                    f"**{trazas['salida_horas']:.1f} horas**."
                )

            registrar_prediccion({
                "timestamp_consulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tipo_muestra": tipo, "subtipo_muestra": subtipo, "grupo_analitico": grupo,
                "num_parametros_solicitados": num_parametros, "prioridad_cliente": prioridad,
                "sector_cliente": sector, "sede_laboratorio": sede, "carga_laboratorio_dia": carga,
                "num_muestras_lote": lote, "es_fin_de_semana": es_fin_de_semana,
                "temporada_alta_demanda": temporada_alta, "mes_recepcion": fecha_recepcion.month,
                "fecha_recepcion": fecha_recepcion.strftime("%Y-%m-%d"),
                "plazo_comprometido_horas": plazo, "tiempo_estimado_horas": round(horas, 2),
            })
            st.caption("Esta consulta quedó registrada en predicciones_log.csv (pestaña 'Historial de predicciones').")

# ---------------------------------------------------------------------------
# Tab 2: panel de resultados del modelo
# ---------------------------------------------------------------------------
with tab_resultados:
    metricas, fuente = cargar_metricas()
    if metricas is None:
        st.warning("No se encontró ningún archivo de métricas (metricas_modelo_pytorch.json / metricas_modelo.json) en esta carpeta.")
    else:
        st.caption(f"Métricas cargadas desde: {fuente}")
        claves = ["MAE_horas", "RMSE_horas", "R2", "MAPE_pct"]
        df_metricas = pd.DataFrame({k: v for k, v in metricas.items() if isinstance(v, dict)}).T[claves]
        st.dataframe(df_metricas.style.format("{:.3f}"), use_container_width=True)

        figuras = [
            ("fig_curva_perdida.png", "Curva de pérdida durante el entrenamiento"),
            ("fig_pred_vs_real.png", "Predicho vs. real (conjunto de prueba)"),
            ("fig_residuos.png", "Distribución de residuos"),
            ("fig_importancia_variables.png", "Importancia de variables (permutación)"),
        ]
        cols = st.columns(2)
        for i, (fname, caption) in enumerate(figuras):
            if os.path.exists(fname):
                cols[i % 2].image(fname, caption=caption, use_container_width=True)
            else:
                cols[i % 2].info(f"No se encontró {fname} en esta carpeta.")

# ---------------------------------------------------------------------------
# Tab 3: historial de predicciones (log acumulado)
# ---------------------------------------------------------------------------
with tab_historial:
    if not os.path.exists(LOG_PATH):
        st.info("Todavía no se ha registrado ninguna predicción. Usa la pestaña 'Estimar una muestra nueva' primero.")
    else:
        df_log = pd.read_csv(LOG_PATH, dtype={"tiempo_real_horas": str, "estado_entrega_real": str})
        validadas = df_log["tiempo_real_horas"].notna() & (df_log["tiempo_real_horas"].str.strip() != "")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicciones registradas", len(df_log))
        c2.metric("Validadas con tiempo real", int(validadas.sum()))
        c3.metric("Pendientes de validar", int((~validadas).sum()))

        st.dataframe(df_log.sort_values("timestamp_consulta", ascending=False), use_container_width=True)

        st.download_button(
            "Descargar log completo (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="predicciones_log.csv",
            mime="text/csv",
        )

        st.markdown(
            "**Cómo cerrar el ciclo con datos reales:** cuando el informe de una muestra ya se "
            "entregó, abre `predicciones_log.csv`, busca la fila correspondiente y completa la "
            "columna `tiempo_real_horas` con las horas que realmente tomó (y opcionalmente "
            "`estado_entrega_real` con \"A tiempo\" o \"Retrasado\"). Luego ejecuta:"
        )
        st.code("python reentrenar_con_log.py", language="bash")
        st.caption(
            "Ese script combina las filas ya validadas con el dataset sintético base y genera "
            "dataset_laboratorio_muestras_actualizado.csv, listo para reentrenar el modelo con "
            "python modelo_pytorch.py --data dataset_laboratorio_muestras_actualizado.csv."
        )
