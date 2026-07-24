"""
Combina el dataset sintético base con las predicciones del log que ya fueron
validadas (columna 'tiempo_real_horas' completada a mano en predicciones_log.csv
tras conocer el resultado real de cada muestra) y genera un dataset actualizado
para reentrenar el modelo con datos reales.

Uso:
    python reentrenar_con_log.py
    python modelo_pytorch.py --data dataset_laboratorio_muestras_actualizado.csv --epochs 600
"""
import sys

import pandas as pd

BASE_DATASET = "dataset_laboratorio_muestras.csv"
LOG_FILE = "predicciones_log.csv"
SALIDA = "dataset_laboratorio_muestras_actualizado.csv"

# Únicas columnas que modelo_pytorch.py / entrenamiento_numpy.py necesitan
# (ver build_features() en modelo_pytorch.py) — cualquier otra columna del
# dataset original se descarta al combinar, para que las filas "reales" y
# "sintéticas" queden compatibles entre sí.
COLUMNAS_MODELO = [
    "tipo_muestra", "subtipo_muestra", "grupo_analitico", "num_parametros_solicitados",
    "prioridad_cliente", "sector_cliente", "sede_laboratorio", "carga_laboratorio_dia",
    "num_muestras_lote", "es_fin_de_semana", "temporada_alta_demanda", "mes_recepcion",
    "tiempo_total_horas",
]


def main():
    try:
        df_base = pd.read_csv(BASE_DATASET)
    except FileNotFoundError:
        print(f"No se encontró {BASE_DATASET} en esta carpeta.")
        sys.exit(1)
    df_base = df_base[COLUMNAS_MODELO].copy()
    df_base["origen"] = "sintetico"

    try:
        df_log = pd.read_csv(LOG_FILE, dtype={"tiempo_real_horas": str})
    except FileNotFoundError:
        print(f"No se encontró {LOG_FILE}. Todavía no se ha registrado ninguna predicción desde la app.")
        sys.exit(1)

    df_log["tiempo_real_horas"] = df_log["tiempo_real_horas"].astype(str).str.strip()
    df_validadas = df_log[(df_log["tiempo_real_horas"] != "") & (df_log["tiempo_real_horas"] != "nan")].copy()

    if df_validadas.empty:
        print(f"No hay predicciones validadas todavía en {LOG_FILE}.")
        print("Abre ese archivo, completa la columna 'tiempo_real_horas' (en horas) para las")
        print("muestras cuyo informe ya fue entregado, y vuelve a ejecutar este script.")
        sys.exit(0)

    df_validadas["tiempo_total_horas"] = df_validadas["tiempo_real_horas"].astype(float)
    df_validadas = df_validadas[COLUMNAS_MODELO[:-1] + ["tiempo_total_horas"]].copy()
    df_validadas["origen"] = "real_validado"

    df_final = pd.concat([df_base, df_validadas], ignore_index=True)
    df_final.to_csv(SALIDA, index=False, encoding="utf-8-sig")

    print(f"Dataset combinado: {len(df_base)} muestras sintéticas + {len(df_validadas)} muestras reales validadas")
    print(f"Guardado en: {SALIDA}\n")
    print("Para reentrenar el modelo con este dataset combinado, ejecuta:")
    print(f"    python modelo_pytorch.py --data {SALIDA} --epochs 600")


if __name__ == "__main__":
    main()
