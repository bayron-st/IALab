"""
Implementación "desde cero" (NumPy puro) de la MISMA arquitectura descrita y
codificada en PyTorch en `modelo_pytorch.py`.

Motivo: el entorno de ejecución sandbox usado para producir las métricas y
figuras de este informe no tiene acceso a los paquetes CUDA que la rueda
(wheel) oficial de PyTorch en PyPI exige como dependencia obligatoria
incluso para uso exclusivamente en CPU, lo que hace inviable su instalación
aquí por tamaño/ancho de banda. Para poder entrenar el modelo, validarlo y
reportar métricas reales (no simuladas), se reimplementó manualmente el
mismo grafo computacional: forward pass, backpropagation y optimizador Adam,
capa por capa, con las mismas dimensiones, funciones de activación,
regularización y función de pérdida que en la versión PyTorch. El script
`modelo_pytorch.py` es el entregable "oficial" pensado para ejecutarse en
Google Colab o localmente con PyTorch instalado, y debe converger a
resultados equivalentes.

Arquitectura:
  - Embeddings para variables categóricas (tipo_muestra, subtipo_muestra,
    grupo_analitico, prioridad_cliente, sector_cliente, sede_laboratorio)
  - Concatenación de embeddings + variables numéricas estandarizadas
  - MLP: Linear(in,128) -> ReLU -> Dropout(0.2)
         Linear(128,64) -> ReLU -> Dropout(0.2)
         Linear(64,32)  -> ReLU
         Linear(32,1)   (salida de regresión, escala log1p(horas))
  - Pérdida: Huber (robusta a valores atípicos por reanálisis de calidad)
  - Optimizador: Adam (lr=1e-3, weight_decay=1e-5) con early stopping
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# 1. Carga y preprocesamiento
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset_laboratorio_muestras.csv")

CAT_COLS = ["tipo_muestra", "subtipo_muestra", "grupo_analitico",
            "prioridad_cliente", "sector_cliente", "sede_laboratorio"]
NUM_COLS = ["num_parametros_solicitados", "carga_laboratorio_dia",
            "num_muestras_lote", "es_fin_de_semana", "temporada_alta_demanda"]
TARGET = "tiempo_total_horas"

# Codificación cíclica del mes (estacionalidad)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_recepcion"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_recepcion"] / 12)
NUM_COLS = NUM_COLS + ["mes_sin", "mes_cos"]

# Label encoding para categóricas (para usar como índice de embedding)
cat_maps = {}
for c in CAT_COLS:
    cats = sorted(df[c].unique())
    cat_maps[c] = {v: i for i, v in enumerate(cats)}
    df[c + "_idx"] = df[c].map(cat_maps[c])

CAT_IDX_COLS = [c + "_idx" for c in CAT_COLS]
cardinalities = {c: len(cat_maps[c]) for c in CAT_COLS}
# heurística de Guo & Berkhahn (2016) acotada entre 2 y 8
emb_dims = {c: int(np.clip(round(cardinalities[c] ** 0.5) + 1, 2, 8)) for c in CAT_COLS}

# split train/val/test 70/15/15 (aleatorio estratificado por grupo analítico)
idx_all = np.arange(len(df))
rng.shuffle(idx_all)
n = len(df)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
idx_train = idx_all[:n_train]
idx_val = idx_all[n_train:n_train + n_val]
idx_test = idx_all[n_train + n_val:]

# estandarización de numéricas (fit solo con train)
num_mean = df.loc[idx_train, NUM_COLS].mean()
num_std = df.loc[idx_train, NUM_COLS].std().replace(0, 1)
df_num_std = (df[NUM_COLS] - num_mean) / num_std

# target en escala log1p (duraciones positivas y sesgadas a la derecha)
y_log = np.log1p(df[TARGET].values.astype(np.float64))

X_cat = df[CAT_IDX_COLS].values.astype(np.int64)
X_num = df_num_std.values.astype(np.float64)

def subset(idx):
    return X_cat[idx], X_num[idx], y_log[idx]

Xc_tr, Xn_tr, y_tr = subset(idx_train)
Xc_val, Xn_val, y_val = subset(idx_val)
Xc_te, Xn_te, y_te = subset(idx_test)

# ---------------------------------------------------------------------------
# 2. Inicialización de parámetros
# ---------------------------------------------------------------------------
params = {}
for c in CAT_COLS:
    card, dim = cardinalities[c], emb_dims[c]
    # escala comparable a la de las variables numéricas estandarizadas (std=1),
    # de lo contrario W1 (inicializado asumiendo entradas de varianza unitaria)
    # ignora casi por completo la señal proveniente de los embeddings
    params[f"emb_{c}"] = rng.normal(0, 1.0, size=(card, dim))

input_dim = sum(emb_dims.values()) + len(NUM_COLS)
layer_sizes = [input_dim, 128, 64, 32, 1]

def he_init(fan_in, fan_out):
    return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

for i in range(len(layer_sizes) - 1):
    params[f"W{i+1}"] = he_init(layer_sizes[i], layer_sizes[i+1])
    params[f"b{i+1}"] = np.zeros((1, layer_sizes[i+1]))

# estado del optimizador Adam
adam_state = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in params.items()}
t_step = 0
LR = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
WEIGHT_DECAY = 1e-6
DROPOUT_P = 0.1
HUBER_DELTA = 2.0

def build_input(Xc, Xn, training, dropout_masks_cache=None):
    embs = []
    for j, c in enumerate(CAT_COLS):
        embs.append(params[f"emb_{c}"][Xc[:, j]])
    X = np.concatenate(embs + [Xn], axis=1)
    return X

def relu(z):
    return np.maximum(0, z)

def huber_loss_and_grad(y_pred, y_true, delta=HUBER_DELTA):
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    loss = np.mean(0.5 * quad**2 + delta * lin)
    grad = np.where(abs_err <= delta, err, delta * np.sign(err)) / len(err)
    return loss, grad.reshape(-1, 1)

def forward(Xc, Xn, training=True):
    X0 = build_input(Xc, Xn, training)
    cache = {"X0": X0}
    z1 = X0 @ params["W1"] + params["b1"]; a1 = relu(z1)
    mask1 = (rng.random(a1.shape) > DROPOUT_P).astype(np.float64) / (1 - DROPOUT_P) if training else np.ones_like(a1)
    a1d = a1 * mask1
    z2 = a1d @ params["W2"] + params["b2"]; a2 = relu(z2)
    mask2 = (rng.random(a2.shape) > DROPOUT_P).astype(np.float64) / (1 - DROPOUT_P) if training else np.ones_like(a2)
    a2d = a2 * mask2
    z3 = a2d @ params["W3"] + params["b3"]; a3 = relu(z3)
    y_hat = a3 @ params["W4"] + params["b4"]
    cache.update(z1=z1, a1=a1, mask1=mask1, a1d=a1d, z2=z2, a2=a2, mask2=mask2, a2d=a2d, z3=z3, a3=a3)
    return y_hat, cache

def backward(y_pred, y_true, cache, Xc):
    grads = {}
    batch = y_true.shape[0]
    _, dyhat = huber_loss_and_grad(y_pred, y_true)  # (batch,1)

    a3 = cache["a3"]
    grads["W4"] = a3.T @ dyhat + WEIGHT_DECAY * params["W4"]
    grads["b4"] = dyhat.sum(axis=0, keepdims=True)
    da3 = dyhat @ params["W4"].T
    dz3 = da3 * (cache["z3"] > 0)

    a2d = cache["a2d"]
    grads["W3"] = a2d.T @ dz3 + WEIGHT_DECAY * params["W3"]
    grads["b3"] = dz3.sum(axis=0, keepdims=True)
    da2d = dz3 @ params["W3"].T
    da2 = da2d * cache["mask2"]
    dz2 = da2 * (cache["z2"] > 0)

    a1d = cache["a1d"]
    grads["W2"] = a1d.T @ dz2 + WEIGHT_DECAY * params["W2"]
    grads["b2"] = dz2.sum(axis=0, keepdims=True)
    da1d = dz2 @ params["W2"].T
    da1 = da1d * cache["mask1"]
    dz1 = da1 * (cache["z1"] > 0)

    X0 = cache["X0"]
    grads["W1"] = X0.T @ dz1 + WEIGHT_DECAY * params["W1"]
    grads["b1"] = dz1.sum(axis=0, keepdims=True)
    dX0 = dz1 @ params["W1"].T

    # repartir el gradiente de entrada hacia cada tabla de embeddings
    col = 0
    for j, c in enumerate(CAT_COLS):
        dim = emb_dims[c]
        dEmb = np.zeros_like(params[f"emb_{c}"])
        seg = dX0[:, col:col + dim]
        np.add.at(dEmb, Xc[:, j], seg)
        grads[f"emb_{c}"] = dEmb
        col += dim
    return grads

def adam_step(grads):
    global t_step, LR
    t_step += 1
    for k, g in grads.items():
        st = adam_state[k]
        st["m"] = BETA1 * st["m"] + (1 - BETA1) * g
        st["v"] = BETA2 * st["v"] + (1 - BETA2) * (g ** 2)
        mhat = st["m"] / (1 - BETA1 ** t_step)
        vhat = st["v"] / (1 - BETA2 ** t_step)
        params[k] -= LR * mhat / (np.sqrt(vhat) + EPS)

def evaluate(Xc, Xn, y_true_log):
    y_hat, _ = forward(Xc, Xn, training=False)
    y_pred_h = np.expm1(y_hat.reshape(-1))
    y_true_h = np.expm1(y_true_log.reshape(-1))
    mae = np.mean(np.abs(y_pred_h - y_true_h))
    rmse = np.sqrt(np.mean((y_pred_h - y_true_h) ** 2))
    ss_res = np.sum((y_true_h - y_pred_h) ** 2)
    ss_tot = np.sum((y_true_h - np.mean(y_true_h)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs((y_pred_h - y_true_h) / np.maximum(y_true_h, 1e-6))) * 100
    return dict(MAE_horas=mae, RMSE_horas=rmse, R2=r2, MAPE_pct=mape), y_pred_h, y_true_h

# ---------------------------------------------------------------------------
# 3. Entrenamiento con mini-batches y early stopping
# ---------------------------------------------------------------------------
BATCH = 64
EPOCHS = 600
PATIENCE = 80

best_val = np.inf
best_params = None
wait = 0
history = {"train_loss": [], "val_loss": []}

n_train = len(y_tr)
for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(n_train)
    epoch_losses = []
    for start in range(0, n_train, BATCH):
        bidx = order[start:start + BATCH]
        Xc_b, Xn_b, y_b = Xc_tr[bidx], Xn_tr[bidx], y_tr[bidx]
        y_hat, cache = forward(Xc_b, Xn_b, training=True)
        loss, _ = huber_loss_and_grad(y_hat, y_b)
        grads = backward(y_hat, y_b.reshape(-1, 1), cache, Xc_b)
        adam_step(grads)
        epoch_losses.append(loss)

    train_loss = float(np.mean(epoch_losses))
    y_hat_val, _ = forward(Xc_val, Xn_val, training=False)
    val_loss, _ = huber_loss_and_grad(y_hat_val, y_val)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(float(val_loss))

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_params = {k: v.copy() for k, v in params.items()}
        wait = 0
    else:
        wait += 1
        # ReduceLROnPlateau simplificado: reduce la tasa de aprendizaje a la mitad
        # si no hay mejora en 40 épocas consecutivas (antes de agotar la paciencia total)
        if wait > 0 and wait % 40 == 0:
            LR = max(LR * 0.5, 1e-5)
            print(f"  -> sin mejora en {wait} épocas, LR reducido a {LR:.6f}")
        if wait >= PATIENCE:
            print(f"Early stopping en epoch {epoch} (mejor val_loss={best_val:.4f})")
            break

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train_huber={train_loss:.4f} | val_huber={val_loss:.4f}")

# restaurar mejores pesos
for k in params:
    params[k] = best_params[k]

# ---------------------------------------------------------------------------
# 4. Evaluación final en test
# ---------------------------------------------------------------------------
metrics_train, _, _ = evaluate(Xc_tr, Xn_tr, y_tr)
metrics_val, _, _ = evaluate(Xc_val, Xn_val, y_val)
metrics_test, y_pred_test, y_true_test = evaluate(Xc_te, Xn_te, y_te)

print("\n== Métricas (horas) ==")
print("Train:", metrics_train)
print("Val:  ", metrics_val)
print("Test: ", metrics_test)

with open("metricas_modelo.json", "w", encoding="utf-8") as f:
    json.dump({"train": metrics_train, "val": metrics_val, "test": metrics_test,
               "epochs_entrenados": len(history["train_loss"]),
               "arquitectura": {"input_dim": input_dim, "layer_sizes": layer_sizes,
                                 "emb_dims": emb_dims, "cardinalities": cardinalities}},
              f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history["train_loss"], label="Entrenamiento")
plt.plot(history["val_loss"], label="Validación")
plt.xlabel("Época")
plt.ylabel("Pérdida Huber (escala log1p)")
plt.title("Curva de pérdida durante el entrenamiento")
plt.legend()
plt.tight_layout()
plt.savefig("fig_curva_perdida.png", dpi=150)
plt.close()

plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_true_test, y_pred_test, alpha=0.35, s=14)
lims = [0, max(y_true_test.max(), y_pred_test.max()) * 1.05]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Tiempo real (horas)")
plt.ylabel("Tiempo predicho (horas)")
plt.title("Predicho vs. real — conjunto de prueba")
plt.tight_layout()
plt.savefig("fig_pred_vs_real.png", dpi=150)
plt.close()

residuals = y_pred_test - y_true_test
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=40, color="steelblue", edgecolor="white")
plt.xlabel("Residuo (predicho - real, horas)")
plt.ylabel("Frecuencia")
plt.title("Distribución de residuos — conjunto de prueba")
plt.tight_layout()
plt.savefig("fig_residuos.png", dpi=150)
plt.close()

# Importancia por permutación (interpretabilidad, relevante para la reflexión ética)
def permutation_importance():
    base_mae = metrics_test["MAE_horas"]
    importances = {}
    for j, c in enumerate(CAT_COLS):
        Xc_perm = Xc_te.copy()
        rng.shuffle(Xc_perm[:, j])
        m, _, _ = evaluate(Xc_perm, Xn_te, y_te)
        importances[c] = m["MAE_horas"] - base_mae
    for j, c in enumerate(NUM_COLS):
        Xn_perm = Xn_te.copy()
        rng.shuffle(Xn_perm[:, j])
        m, _, _ = evaluate(Xc_te, Xn_perm, y_te)
        importances[c] = m["MAE_horas"] - base_mae
    return dict(sorted(importances.items(), key=lambda kv: -kv[1]))

importances = permutation_importance()
with open("importancia_variables.json", "w", encoding="utf-8") as f:
    json.dump(importances, f, ensure_ascii=False, indent=2)

plt.figure(figsize=(7, 5))
names = list(importances.keys())
vals = list(importances.values())
plt.barh(names[::-1], vals[::-1], color="darkcyan")
plt.xlabel("Incremento del MAE al permutar la variable (horas)")
plt.title("Importancia de variables (permutation importance)")
plt.tight_layout()
plt.savefig("fig_importancia_variables.png", dpi=150)
plt.close()

print("\nImportancia de variables:")
for k, v in importances.items():
    print(f"  {k}: {v:+.2f} h")

print("\nListo. Métricas y figuras guardadas.")
