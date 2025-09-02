# -*- coding: utf-8 -*-
"""
training_data2.csv (앞/뒤, 고정자세)로 실내 위치 예측 모델 학습/평가/저장
- 캘리브레이션(하드/소프트 아이언) 단계는 생략
- 특징: B 성분, |B|, |B_xy|, 단위벡터, 각도(sin/cos)
- 출력: mlp_position.pkl, label_encoder.pkl, metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 0) 설정
# -------------------------------
CSV_PATH = "training_data2.csv"     # 같은 폴더면 파일명만
LABEL_COL = "Position"              # 라벨 컬럼명 (필요시 바꾸세요)
MAG_COLS = ["Mag_X", "Mag_Y", "Mag_Z"]
ORI_COLS = ["Ori_X", "Ori_Y", "Ori_Z"]  # deg 가정. 없으면 빈 리스트로 두세요: ORI_COLS=[]
TEST_SIZE = 0.2
RANDOM_STATE = 42
STANDARDIZE = True                  # 표준화 ON/OFF
DEROTATE_MODE = "none"              # "none" | "yaw_only" | "full"  (캘리브레이션 없으니 기본 none 권장)

MODEL_PKL = "mlp_position.pkl"
LE_PKL    = "label_encoder.pkl"
METRICS_JSON = "metrics.json"

# -------------------------------
# 1) 유틸
# -------------------------------
def _rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def _rot_y(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])

def _rot_z(g):
    cg, sg = np.cos(g), np.sin(g)
    return np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])

def euler_zyx_to_R(roll, pitch, yaw):
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

def derotate(B, df, mode="none"):
    if mode == "none" or len(ORI_COLS)==0:
        return B
    roll = np.deg2rad(df[ORI_COLS[0]].values.astype(float))
    pitch= np.deg2rad(df[ORI_COLS[1]].values.astype(float))
    yaw  = np.deg2rad(df[ORI_COLS[2]].values.astype(float))
    out = np.zeros_like(B)
    if mode == "yaw_only":
        for i in range(B.shape[0]):
            out[i] = _rot_z(yaw[i]) @ B[i]
        return out
    if mode == "full":
        for i in range(B.shape[0]):
            out[i] = euler_zyx_to_R(roll[i], pitch[i], yaw[i]) @ B[i]
        return out
    return B

def build_features(df):
    # B 성분
    B = df[MAG_COLS].values.astype(float)

    # (선택) 디로테이션 - 캘리브레이션 없이 쓰는 건 과교정 위험이 있어 기본 끔
    Bc = derotate(B, df, DEROTATE_MODE)

    # 크기/방향 특성
    B_mag = np.linalg.norm(Bc, axis=1, keepdims=True)           # |B|
    B_xy  = np.linalg.norm(Bc[:, :2], axis=1, keepdims=True)    # |B_xy|
    eps = 1e-9
    B_unit = Bc / (B_mag + eps)                                 # 단위벡터

    feat = pd.DataFrame(index=df.index)
    feat["B_x"] = Bc[:,0]; feat["B_y"] = Bc[:,1]; feat["B_z"] = Bc[:,2]
    feat["B_mag"] = B_mag[:,0]
    feat["B_xy_mag"] = B_xy[:,0]
    feat["Bux"] = B_unit[:,0]; feat["Buy"] = B_unit[:,1]; feat["Buz"] = B_unit[:,2]

    # 각도 → sin/cos (있을 때만)
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

# -------------------------------
# 3) 피처 생성 + 스케일러
# -------------------------------
feat_tr = build_features(X_tr_df)
feat_te = build_features(X_te_df)

scaler = None
if STANDARDIZE:
    scaler = StandardScaler().fit(feat_tr.values)
    feat_tr.iloc[:, :] = scaler.transform(feat_tr.values)
    feat_te.iloc[:, :] = scaler.transform(feat_te.values)

X_tr = feat_tr.values
X_te = feat_te.values

# 라벨 인코딩
le = LabelEncoder()
y_tr = le.fit_transform(y_tr_raw)
y_te = le.transform(y_te_raw)

# -------------------------------
# 4) 모델 정의/학습 (MLP)
# -------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=250,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
    verbose=False
)

print("🚀 Training MLP ...")
mlp.fit(X_tr, y_tr)
print("✅ Train done. epochs:", len(getattr(mlp, "loss_curve_", [])))

# -------------------------------
# 5) 평가 (MLP)
# -------------------------------
y_tr_pred = mlp.predict(X_tr)
y_te_pred = mlp.predict(X_te)

acc_tr = accuracy_score(y_tr, y_tr_pred)
acc_te = accuracy_score(y_te, y_te_pred)
print(f"📊 MLP Accuracy - train: {acc_tr:.4f} | test: {acc_te:.4f}")

class_names = [str(c) for c in le.classes_]
label_order = np.arange(len(le.classes_))

report = classification_report(
    y_te, y_te_pred,
    labels=label_order,
    target_names=class_names,
    digits=4,
    zero_division=0
)
cm = confusion_matrix(y_te, y_te_pred, labels=label_order)

print("\n📄 Classification Report (MLP, test)")
print(report)
print("🔢 Confusion Matrix (MLP, test)")
print(cm)

# -------------------------------
# 6) 비교용 베이스라인 (저장은 안 함)
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
knn.fit(X_tr, y_tr)
y_te_knn = knn.predict(X_te)
acc_knn = accuracy_score(y_te, y_te_knn)
print(f"\n🧪 1-NN(cosine) baseline test acc: {acc_knn:.4f}")

# -------------------------------
# 7) 저장
# -------------------------------
joblib.dump(mlp, MODEL_PKL)
joblib.dump(le, LE_PKL)

if 'scaler' in globals() and scaler is not None:
    joblib.dump(scaler, "scaler.pkl")

metrics = {
    "train_accuracy": float(acc_tr),
    "test_accuracy": float(acc_te),
    "classes": class_names,
    "confusion_matrix": cm.tolist(),
    "classification_report_text": report,
    "loss_curve": [float(v) for v in getattr(mlp, "loss_curve_", [])],
    "features_used": list(feat_tr.columns),
    "standardized": STANDARDIZE,
    "derotate_mode": DEROTATE_MODE
}
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n💾 Saved: {MODEL_PKL}, {LE_PKL}, {METRICS_JSON}")
print("🎯 All done.")
