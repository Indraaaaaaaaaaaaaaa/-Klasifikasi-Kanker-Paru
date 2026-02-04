import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Klasifikasi Kanker Paru", page_icon="🫁", layout="centered")

FEATURES = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]

MODEL_PATHS = [
    "models/best_lung_cancer_model.joblib",
    "best_lung_cancer_model.joblib",
    "/mnt/data/models/best_lung_cancer_model.joblib",
    "/mnt/data/best_lung_cancer_model.joblib",
]

METRICS_PATHS = [
    "outputs/metrics.json",
    "/mnt/data/outputs/metrics.json",
]

OUTPUTS_DIRS = [
    "outputs",
    "/mnt/data/outputs",
]

def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

def load_metrics():
    for p in METRICS_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p
    return None, None

def get_outputs_dir():
    for d in OUTPUTS_DIRS:
        if os.path.isdir(d):
            return d
    return None

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def get_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    X = X[FEATURES].copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # ambil index kelas 1 jika ada
        pos_idx = 1
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = len(classes) - 1
        return proba[:, pos_idx]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[1] >= 2:
            scores = scores[:, 1]
        return _sigmoid(np.asarray(scores).reshape(-1))

    pred = model.predict(X)
    return np.asarray(pred).astype(float).reshape(-1)

def predict_df(model, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    # Case-insensitive kolom
    rename_map = {c.lower(): c for c in FEATURES}
    X = X.copy()
    X.columns = [c.strip() for c in X.columns]
    X = X.rename(columns={c: rename_map.get(c.lower(), c) for c in X.columns})

    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Kolom wajib belum lengkap: {missing}")

    # Paksa numerik
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X[FEATURES].isna().any().any():
        bad_cols = X[FEATURES].columns[X[FEATURES].isna().any()].tolist()
        raise ValueError(f"Ada nilai kosong/invalid pada kolom: {bad_cols}")

    for c in FEATURES:
        X[c] = X[c].astype(int)

    prob = get_positive_proba(model, X)
    pred = (prob >= threshold).astype(int)

    out = X[FEATURES].copy()
    out["PROB_LUNG_CANCER"] = prob
    out["PRED_CLASS"] = pred
    return out

def yesno_label(v: int) -> str:
    return "1 - Ya" if v == 1 else "0 - Tidak"


st.title("🫁 Klasifikasi Kanker Paru")
st.caption("Aplikasi demonstrasi (akademik) — bukan pengganti diagnosis medis.")

with st.expander("ℹ️ Petunjuk Input"):
    st.markdown(
        """
        - **GENDER**: `1 = Laki-laki`, `0 = Perempuan`
        - Fitur biner lainnya: `1 = Ya`, `0 = Tidak`
        - **AGE**: umur (angka)
        """
    )

model, model_path = load_model()
metrics_obj, metrics_path = load_metrics()
outputs_dir = get_outputs_dir()

if model is None:
    st.error("Model belum ditemukan. Jalankan training dulu agar `models/best_lung_cancer_model.joblib` terbentuk.")
else:
    st.success(f"✅ Model loaded: `{model_path}`")

threshold = st.slider("Threshold prediksi", 0.05, 0.95, 0.50, 0.01)

# Ringkasan metrik
st.subheader("Ringkasan Evaluasi (dari training)")
if metrics_obj is None:
    st.warning("File metrics.json tidak ditemukan. Jalankan training dulu.")
else:
    st.caption(f"Sumber: `{metrics_path}`")
    best_model = metrics_obj.get("best_model", "-")
    st.write("**Model terbaik (berdasarkan F1 kelas positif):**", best_model)

    # tampilkan tabel ringkas
    rows = []
    for m in metrics_obj.get("metrics", []):
        rows.append({
            "Model": m.get("model"),
            "Akurasi": m.get("accuracy"),
            "Precision (kelas 1)": m.get("precision_pos"),
            "Recall (kelas 1)": m.get("recall_pos"),
            "F1-score (kelas 1)": m.get("f1_pos"),
            "AUC": m.get("auc"),
        })
    dfm = pd.DataFrame(rows).sort_values("F1-score (kelas 1)", ascending=False)
    st.dataframe(dfm, use_container_width=True)

# Visualisasi training
st.subheader("Visualisasi Hasil Training")
if outputs_dir is None:
    st.info("Folder outputs/ belum ada. Jalankan training untuk menghasilkan gambar.")
else:
    # tampilkan beberapa gambar utama jika ada
    candidates = [
        ("Distribusi kelas", "gambar_01_distribusi_kelas.png"),
        ("Heatmap korelasi", "gambar_02_heatmap_korelasi.png"),
        ("CM LR (weight)", "gambar_03_cm_lr_weight.png"),
        ("CM XGB (weight)", "gambar_04_cm_xgb_weight.png"),
        ("CM LR + SMOTE", "gambar_05_cm_lr_smote.png"),
        ("CM XGB + SMOTE", "gambar_06_cm_xgb_smote.png"),
        ("ROC model terbaik", "gambar_07_roc_best.png"),
        ("PR model terbaik", "gambar_08_pr_best.png"),
        ("Feature importance", "gambar_09_feature_importance.png"),
    ]
    for label, fname in candidates:
        fpath = os.path.join(outputs_dir, fname)
        if os.path.exists(fpath):
            st.markdown(f"**{label}**")
            st.image(fpath, use_container_width=True)

# Prediksi
tabs = st.tabs(["🧍 Prediksi 1 Pasien (Form)", "📄 Prediksi Banyak Pasien (Upload CSV)"])

with tabs[0]:
    st.subheader("Input Data (Form)")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("GENDER", options=[1, 0], format_func=lambda x: "1 - Laki-laki" if x == 1 else "0 - Perempuan")
        age = st.slider("AGE (Umur)", 0, 120, 45)
        smoking = st.selectbox("SMOKING", [0, 1], format_func=yesno_label)
        yellow_fingers = st.selectbox("YELLOW_FINGERS", [0, 1], format_func=yesno_label)
        anxiety = st.selectbox("ANXIETY", [0, 1], format_func=yesno_label)
        peer_pressure = st.selectbox("PEER_PRESSURE", [0, 1], format_func=yesno_label)
        chronic_disease = st.selectbox("CHRONIC_DISEASE", [0, 1], format_func=yesno_label)

    with col2:
        fatigue = st.selectbox("FATIGUE", [0, 1], format_func=yesno_label)
        allergy = st.selectbox("ALLERGY", [0, 1], format_func=yesno_label)
        wheezing = st.selectbox("WHEEZING", [0, 1], format_func=yesno_label)
        alcohol = st.selectbox("ALCOHOL_CONSUMING", [0, 1], format_func=yesno_label)
        coughing = st.selectbox("COUGHING", [0, 1], format_func=yesno_label)
        sob = st.selectbox("SHORTNESS_OF_BREATH", [0, 1], format_func=yesno_label)
        swallow = st.selectbox("SWALLOWING_DIFFICULTY", [0, 1], format_func=yesno_label)
        chest_pain = st.selectbox("CHEST_PAIN", [0, 1], format_func=yesno_label)

    if st.button("Prediksi (Form)", type="primary", disabled=(model is None)):
        row = {
            "GENDER": int(gender),
            "AGE": int(age),
            "SMOKING": int(smoking),
            "YELLOW_FINGERS": int(yellow_fingers),
            "ANXIETY": int(anxiety),
            "PEER_PRESSURE": int(peer_pressure),
            "CHRONIC_DISEASE": int(chronic_disease),
            "FATIGUE": int(fatigue),
            "ALLERGY": int(allergy),
            "WHEEZING": int(wheezing),
            "ALCOHOL_CONSUMING": int(alcohol),
            "COUGHING": int(coughing),
            "SHORTNESS_OF_BREATH": int(sob),
            "SWALLOWING_DIFFICULTY": int(swallow),
            "CHEST_PAIN": int(chest_pain),
        }
        X_one = pd.DataFrame([row], columns=FEATURES)

        try:
            out = predict_df(model, X_one, threshold=threshold).iloc[0]
            prob = float(out["PROB_LUNG_CANCER"])
            pred = int(out["PRED_CLASS"])

            st.markdown("### Hasil")
            st.metric("Prediksi kelas", pred, help="0 = tidak kanker paru, 1 = kanker paru")
            st.metric("Probabilitas kanker paru", f"{prob:.4f}")
            st.progress(min(max(prob, 0.0), 1.0))
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

with tabs[1]:
    st.subheader("Upload CSV (Batch)")
    st.markdown(
        """
        **Format CSV** harus punya kolom fitur berikut (boleh uppercase atau lowercase):  
        `GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN`
        """
    )

    template_df = pd.DataFrame([{c: 0 for c in FEATURES}])
    template_df["AGE"] = 45
    st.download_button(
        "Download template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_input_lung_cancer.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("Preview data:", df_in.head())

            if st.button("Prediksi (CSV)", type="primary", disabled=(model is None)):
                out = predict_df(model, df_in, threshold=threshold)
                st.success("Prediksi selesai ✅")
                st.dataframe(out, use_container_width=True)

                st.download_button(
                    label="Download hasil prediksi (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_lung_cancer.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Gagal membaca/prediksi CSV: {e}")

st.divider()
st.caption("© Prototipe Streamlit - Klasifikasi Kanker Paru (Akademik)")
