"""
Generación de dataset sintético para el laboratorio de análisis de agua y suelo.
Simula el proceso completo: recepción -> preparación -> análisis -> validación -> informe.
Autor: proyecto académico "Diseño de algoritmos de aprendizaje profundo".
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
N = 2200

# ---------------------------------------------------------------------------
# 1. Catálogos de referencia (basados en prácticas reales de laboratorios
#    ambientales colombianos acreditados IDEAM bajo NTC-ISO/IEC 17025,
#    Resolución 2115 de 2007 y Decreto 1076 de 2015)
# ---------------------------------------------------------------------------
tipos_muestra = ["Agua", "Suelo"]

subtipos = {
    "Agua": ["Potable", "Residual/Vertimiento", "Superficial", "Subterránea"],
    "Suelo": ["Agrícola", "Contaminado/Industrial", "Sedimento", "Relleno Sanitario"],
}

# grupo analítico: (tiempo_base_horas, desviacion_horas, cardinalidad de parametros tipica)
grupos_analiticos = {
    "Fisicoquimico_Basico":   (18, 6,  (3, 8)),
    "Microbiologico":         (36, 10, (1, 4)),
    "Metales_Pesados_ICP":    (60, 14, (3, 12)),
    "DBO5_DQO":               (128, 10, (1, 2)),   # DBO5 exige incubación regulatoria ~5 días
    "Plaguicidas_Cromatografia": (90, 18, (2, 10)),
    "Hidrocarburos_HTP":      (78, 16, (1, 3)),
    "Textura_Fertilidad_Suelo": (54, 12, (3, 9)),
}

grupo_por_tipo = {
    "Agua": ["Fisicoquimico_Basico", "Microbiologico", "Metales_Pesados_ICP",
             "DBO5_DQO", "Plaguicidas_Cromatografia"],
    "Suelo": ["Textura_Fertilidad_Suelo", "Metales_Pesados_ICP",
              "Hidrocarburos_HTP", "Microbiologico"],
}

prioridades = ["Normal", "Urgente", "Express"]
prioridad_factor = {"Normal": 1.00, "Urgente": 0.80, "Express": 0.62}  # reduce tiempo de cola, no la parte regulatoria fija

sectores_cliente = ["Industrial", "Agricola", "Consultoria_Ambiental", "Entidad_Publica", "Minero_Energetico"]
sedes = ["Bogota", "Medellin", "Cali", "Barranquilla"]

analista_experiencia_niveles = ["Junior", "Semi-Senior", "Senior"]

# ---------------------------------------------------------------------------
# 2. Generación registro a registro
# ---------------------------------------------------------------------------
fecha_inicio = datetime(2024, 1, 1)
registros = []

for i in range(N):
    tipo = rng.choice(tipos_muestra, p=[0.58, 0.42])
    subtipo = rng.choice(subtipos[tipo])
    grupo = rng.choice(grupo_por_tipo[tipo])
    base_h, sd_h, (p_min, p_max) = grupos_analiticos[grupo]

    num_parametros = int(rng.integers(p_min, p_max + 1))
    prioridad = rng.choice(prioridades, p=[0.70, 0.22, 0.08])
    sector = rng.choice(sectores_cliente)
    sede = rng.choice(sedes, p=[0.42, 0.27, 0.20, 0.11])

    # fecha/hora de recepción (simulando 2 años de operación, distribuida en días hábiles principalmente)
    dias_offset = int(rng.integers(0, 730))
    hora_recepcion = int(rng.integers(7, 17))
    fecha_recepcion = fecha_inicio + timedelta(days=dias_offset, hours=hora_recepcion)
    dia_semana = fecha_recepcion.weekday()  # 0=lunes
    es_fin_de_semana = 1 if dia_semana >= 5 else 0
    mes = fecha_recepcion.month
    # temporada alta: enero-marzo y julio-septiembre (cierres de informes ambientales / temporada seca-lluvias)
    temporada_alta = 1 if mes in (1, 2, 3, 7, 8, 9) else 0

    # carga del laboratorio el día de recepción (muestras simultáneas en cola)
    carga_base = 12 if temporada_alta else 7
    carga_laboratorio_dia = max(1, int(rng.normal(carga_base, 4)))

    # tamaño del lote recibido junto con la muestra (mismo cliente/mismo día)
    num_muestras_lote = int(rng.choice([1, 1, 1, 2, 3, 4, 5, 8], p=[0.30, 0.15, 0.10, 0.15, 0.12, 0.08, 0.06, 0.04]))

    # nivel de experiencia del analista asignado (conocido solo tras triage; NO se usa como
    # feature de entrada del modelo por no estar disponible al momento de la recepción,
    # se conserva para análisis descriptivo y para simular la variabilidad real del proceso)
    analista_experiencia = rng.choice(analista_experiencia_niveles, p=[0.30, 0.45, 0.25])
    factor_experiencia = {"Junior": 1.15, "Semi-Senior": 1.00, "Senior": 0.88}[analista_experiencia]

    # ---- construcción del tiempo total (horas) ----
    # Nota de diseño: el ruido de "tiempo_base" y el coeficiente de "efecto_parametros"
    # se mantienen deliberadamente moderados (no nulos) para que el dataset conserve
    # variabilidad de proceso realista sin ahogar por completo la señal de las
    # covariables observables (lo que impediría al modelo aprender algo útil).
    tiempo_base = rng.normal(base_h, sd_h * 0.45)
    tiempo_base = max(tiempo_base, base_h * 0.55)

    efecto_parametros = num_parametros * (1.7 + rng.normal(0, 0.15))
    efecto_prioridad = prioridad_factor[prioridad]
    efecto_carga = 1 + max(0, (carga_laboratorio_dia - 8)) * 0.018
    efecto_lote = 1 + (num_muestras_lote - 1) * 0.025
    efecto_fin_semana = 1.18 if es_fin_de_semana else 1.0
    efecto_sede = {"Bogota": 1.00, "Medellin": 1.03, "Cali": 1.06, "Barranquilla": 1.10}[sede]

    tiempo_variable = (tiempo_base + efecto_parametros) * efecto_prioridad * efecto_carga \
        * efecto_lote * efecto_fin_semana * efecto_sede * factor_experiencia

    # reanálisis por falla de control de calidad (evento estocástico, no observable al ingreso;
    # queda como fuente de incertidumbre irreducible/aleatoria a nivel de muestra individual)
    reanalisis_requerido = int(rng.random() < 0.11)
    penalizacion_reanalisis = rng.normal(28, 6) if reanalisis_requerido else 0
    penalizacion_reanalisis = max(penalizacion_reanalisis, 0)

    ruido = rng.normal(0, 3)
    tiempo_total_horas = max(6.0, tiempo_variable + penalizacion_reanalisis + ruido)

    # regla dura regulatoria: DBO5 nunca puede entregarse antes de 120h (5 días de incubación)
    if grupo == "DBO5_DQO":
        tiempo_total_horas = max(tiempo_total_horas, 120 + rng.normal(4, 2))

    fecha_informe = fecha_recepcion + timedelta(hours=tiempo_total_horas)
    plazo_comprometido_horas = {"Normal": 96, "Urgente": 72, "Express": 48}[prioridad]
    if grupo == "DBO5_DQO":
        plazo_comprometido_horas = max(plazo_comprometido_horas, 144)
    estado_entrega = "A tiempo" if tiempo_total_horas <= plazo_comprometido_horas else "Retrasado"

    registros.append({
        "id_muestra": f"LAB-{2024}-{i+1:05d}",
        "fecha_recepcion": fecha_recepcion.strftime("%Y-%m-%d %H:%M"),
        "dia_semana_recepcion": ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"][dia_semana],
        "es_fin_de_semana": es_fin_de_semana,
        "mes_recepcion": mes,
        "temporada_alta_demanda": temporada_alta,
        "tipo_muestra": tipo,
        "subtipo_muestra": subtipo,
        "grupo_analitico": grupo,
        "num_parametros_solicitados": num_parametros,
        "prioridad_cliente": prioridad,
        "sector_cliente": sector,
        "sede_laboratorio": sede,
        "carga_laboratorio_dia": carga_laboratorio_dia,
        "num_muestras_lote": num_muestras_lote,
        "analista_experiencia": analista_experiencia,   # informativo, no usado como input del modelo
        "reanalisis_requerido": reanalisis_requerido,   # informativo, no observable al ingreso (posible fuga de datos)
        "plazo_comprometido_horas": plazo_comprometido_horas,
        "tiempo_total_horas": round(tiempo_total_horas, 2),
        "fecha_informe_entregado": fecha_informe.strftime("%Y-%m-%d %H:%M"),
        "estado_entrega": estado_entrega,
    })

df = pd.DataFrame(registros)
df = df.sort_values("fecha_recepcion").reset_index(drop=True)

out_csv = "dataset_laboratorio_muestras.csv"
out_xlsx = "dataset_laboratorio_muestras.xlsx"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
df.to_excel(out_xlsx, index=False)

print("Registros generados:", len(df))
print(df["estado_entrega"].value_counts(normalize=True))
print(df["tiempo_total_horas"].describe())
print(df.groupby("grupo_analitico")["tiempo_total_horas"].mean().sort_values())
