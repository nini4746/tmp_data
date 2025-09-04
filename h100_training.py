# -*- coding: utf-8 -*-
"""
training_data2.csv 기반 PyTorch MLP 학습 (H100 GPU 사용)
출력: mlp_position.pt, label_encoder.pkl, scaler.pkl, metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# 0) 설정
# -------------------------------
CSV_PATH = "training_data2.csv"
LABEL_COL = "Position"
MAG_COLS = ["Mag_X", "Mag_Y", "Mag_Z"]
ORI_COLS = ["Ori_X", "Ori_Y", "Ori_Z"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
STANDARDIZE = True
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PTH = "mlp_position.pt"
LE_PKL    = "label_encoder.pkl"
SCALER_PKL= "scaler.pkl"
METRICS_JSON = "metrics.json"

# -------------------------------
# 1) 피처 생성
# -------------------------------
def build_features(df):
    B = df[MAG_COLS].values.astype(float)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    B_xy = np.linalg.norm(B[:, :2], axis=1, keepdims=True)
    eps = 1e-9
    B_unit = B / (B_mag + eps)

    feat = pd.DataFrame(index=df.index)
    feat["B_x"], feat["B_y"], feat["B_z"] = B[:,0], B[:,1], B[:,2]
    feat["B_mag"] = B_mag[:,0]
    feat["B_xy_mag"] = B_xy[:,0]
    feat["Bux"], feat["Buy"], feat["Buz"] = B_unit[:,0], B_unit[:,1], B_unit[:,2]

    for a in ORI_COLS:
        rad = np.deg2rad(df[a].values.astype(float))
        feat[f"{a}_sin"] = np.sin(rad)
        feat[f"{a}_cos"] = np.cos(rad)

    return feat

# -------------------------------
# 2) 데이터 로드 & 분할
# -------------------------------
df = pd.read_csv(CSV_PATH)
need_cols = set([LABEL_COL] + MAG_COLS + ORI_COLS)
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

X_raw = df[MAG_COLS + ORI_COLS]
y_raw = df[LABEL_COL].values

X_tr_df, X_te_df, y_tr_raw, y_te_raw = train_test_split(
    X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_raw
)

feat_tr = build_features(X_tr_df)
feat_te = build_features(X_te_df)

scaler = None
if STANDARDIZE:
    scaler = StandardScaler().fit(feat_tr.values)
    feat_tr.iloc[:, :] = scaler.transform(feat_tr.values)
    feat_te.iloc[:, :] = scaler.transform(feat_te.values)

X_tr, X_te = feat_tr.values, feat_te.values

le = LabelEncoder()
y_tr = le.fit_transform(y_tr_raw)
y_te = le.transform(y_te_raw)

joblib.dump(le, LE_PKL)
if scaler is not None:
    joblib.dump(scaler, SCALER_PKL)

# -------------------------------
# 3) Torch Dataset
# -------------------------------
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

train_ds = TensorDataset(X_tr_t, y_tr_t)
test_ds = TensorDataset(X_te_t, y_te_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 4) 모델 정의
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

input_dim = X_tr.shape[1]
output_dim = len(le.classes_)
model = MLP(input_dim, [512, 256, 128], output_dim).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# 5) 학습 루프
# -------------------------------
print("🚀 Training PyTorch MLP ...")
train_losses, test_losses = [], []

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 평가
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t.to(DEVICE))
        loss_te = criterion(preds, y_te_t.to(DEVICE)).item()
        test_losses.append(loss_te)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Test Loss: {loss_te:.4f}")

print("✅ Training done.")

# -------------------------------
# 6) 평가
# -------------------------------
model.eval()
with torch.no_grad():
    y_tr_pred = model(X_tr_t.to(DEVICE)).argmax(dim=1).cpu().numpy()
    y_te_pred = model(X_te_t.to(DEVICE)).argmax(dim=1).cpu().numpy()

acc_tr = accuracy_score(y_tr, y_tr_pred)
acc_te = accuracy_score(y_te, y_te_pred)
print(f"📊 Accuracy - train: {acc_tr:.4f} | test: {acc_te:.4f}")

class_names = [str(c) for c in le.classes_]
report = classification_report(y_te, y_te_pred, target_names=class_names, digits=4, zero_division=0)
cm = confusion_matrix(y_te, y_te_pred)

print("\n📄 Classification Report (test)")
print(report)
print("🔢 Confusion Matrix (test)")
print(cm)

# -------------------------------
# 7) 저장
# -------------------------------
torch.save(model.state_dict(), MODEL_PTH)

metrics = {
    "train_accuracy": float(acc_tr),
    "test_accuracy": float(acc_te),
    "classes": class_names,
    "confusion_matrix": cm.tolist(),
    "classification_report_text": report,
    "train_loss": train_losses,
    "test_loss": test_losses,
    "features_used": list(feat_tr.columns),
    "standardized": STANDARDIZE,
}
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n💾 Saved: {MODEL_PTH}, {LE_PKL}, {SCALER_PKL}, {METRICS_JSON}")
print("🎯 All done.")