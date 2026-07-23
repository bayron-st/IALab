"""
Predicción del tiempo total de análisis de muestras de laboratorio (agua y suelo)
mediante una red neuronal profunda con embeddings de variables categóricas.

Entregable oficial en PyTorch para el taller "Diseño de algoritmos de aprendizaje
profundo". Pensado para ejecutarse en Google Colab o localmente con PyTorch
instalado (`pip install torch pandas scikit-learn matplotlib`).

Reproduce exactamente la arquitectura, hiperparámetros y procedimiento validados
en `entrenamiento_numpy.py` (implementación de referencia usada para generar las
métricas y figuras del informe en el entorno de prototipado sandbox, que no tenía
acceso a las dependencias CUDA que exige la rueda de PyTorch en PyPI).

Uso:
    python modelo_pytorch.py --data dataset_laboratorio_muestras.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CAT_COLS = ["tipo_muestra", "subtipo_muestra", "grupo_analitico",
            "prioridad_cliente", "sector_cliente", "sede_laboratorio"]
NUM_COLS = ["num_parametros_solicitados", "carga_laboratorio_dia",
            "num_muestras_lote", "es_fin_de_semana", "temporada_alta_demanda",
            "mes_sin", "mes_cos"]
TARGET = "tiempo_total_horas"


class LabDataset(Dataset):
    def __init__(self, Xc, Xn, y):
        self.Xc = torch.tensor(Xc, dtype=torch.long)
        self.Xn = torch.tensor(Xn, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xc[idx], self.Xn[idx], self.y[idx]


class TurnaroundTimeNet(nn.Module):
    """
    Red híbrida de datos tabulares:
      - Una tabla de embeddings por variable categórica (entity embeddings,
        Guo & Berkhahn, 2016) en lugar de one-hot, para capturar similitudes
        latentes entre categorías (p. ej. entre grupos analíticos afines).
      - Concatenación con variables numéricas estandarizadas.
      - MLP de 3 capas ocultas (128, 64, 32) con activación ReLU y Dropout.
      - Capa de salida lineal (regresión) sobre log1p(horas).
    """

    def __init__(self, cardinalities: dict, emb_dims: dict, n_num: int, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(cardinalities[c], emb_dims[c]) for c in cardinalities
        })
        input_dim = sum(emb_dims.values()) + n_num
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat, x_num):
        embs = [self.embeddings[c](x_cat[:, j]) for j, c in enumerate(self.embeddings.keys())]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x).squeeze(-1)


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["mes_sin"] = np.sin(2 * np.pi * df["mes_recepcion"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_recepcion"] / 12)

    cat_maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CAT_COLS}
    for c in CAT_COLS:
        df[c + "_idx"] = df[c].map(cat_maps[c])
    cardinalities = {c: len(cat_maps[c]) for c in CAT_COLS}
    emb_dims = {c: int(np.clip(round(cardinalities[c] ** 0.5) + 1, 2, 8)) for c in CAT_COLS}
    return df, cat_maps, cardinalities, emb_dims


def main(args):
    torch.manual_seed(7)
    np.random.seed(7)

    df = pd.read_csv(args.data)
    df, cat_maps, cardinalities, emb_dims = build_features(df)

    idx = np.random.permutation(len(df))
    n_train = int(len(df) * 0.70)
    n_val = int(len(df) * 0.15)
    idx_train, idx_val, idx_test = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    num_mean = df.loc[idx_train, NUM_COLS].mean()
    num_std = df.loc[idx_train, NUM_COLS].std().replace(0, 1)
    df[NUM_COLS] = (df[NUM_COLS] - num_mean) / num_std

    y_log = np.log1p(df[TARGET].values.astype(np.float32))
    Xc = df[[c + "_idx" for c in CAT_COLS]].values.astype(np.int64)
    Xn = df[NUM_COLS].values.astype(np.float32)

    ds_train = LabDataset(Xc[idx_train], Xn[idx_train], y_log[idx_train])
    ds_val = LabDataset(Xc[idx_val], Xn[idx_val], y_log[idx_val])
    ds_test = LabDataset(Xc[idx_test], Xn[idx_test], y_log[idx_test])

    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=256, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TurnaroundTimeNet(cardinalities, emb_dims, n_num=len(NUM_COLS)).to(device)

    criterion = nn.HuberLoss(delta=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=40)

    best_val, best_state, wait, patience = np.inf, None, 0, 80
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xc, xn, y in dl_train:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(xc, xn)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xc, xn, y in dl_val:
                xc, xn, y = xc.to(device), xn.to(device), y.to(device)
                val_losses.append(criterion(model(xc, xn), y).item())
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping en epoch {epoch} (mejor val_loss={best_val:.4f})")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_huber={train_loss:.4f} | val_huber={val_loss:.4f}")

    model.load_state_dict(best_state)

    def evaluate(dl):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xc, xn, y in dl:
                xc, xn = xc.to(device), xn.to(device)
                preds.append(model(xc, xn).cpu().numpy())
                trues.append(y.numpy())
        y_pred = np.expm1(np.concatenate(preds))
        y_true = np.expm1(np.concatenate(trues))
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        mape = np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1e-6, None))) * 100
        return dict(MAE_horas=float(mae), RMSE_horas=float(rmse), R2=float(r2), MAPE_pct=float(mape))

    metrics = {"train": evaluate(dl_train), "val": evaluate(dl_val), "test": evaluate(dl_test)}
    print("\n== Métricas finales (horas) ==")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open("metricas_modelo_pytorch.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), "modelo_turnaround_time.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset_laboratorio_muestras.csv")
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    main(args)
