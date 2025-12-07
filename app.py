
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
st.set_page_config(
    page_title="🏥 Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Diabetes Risk Prediction")
st.markdown("Prediksi risiko diabetes menggunakan **Random Forest**.")

DATA_CSV = "diabetes_dataset.csv"       # <-- pastikan nama file datasetmu ini
MODEL_PATH = "models/diabetes_rf_pipeline.joblib"


# ========== HELPER ==========

def get_target_col(df: pd.DataFrame) -> str:

    possible = ["diagnosed_diabetes", "Outcome", "diabetes", "target", "Diabetes_binary"]
    for col in possible:
        if col in df.columns:
            return col
    # fallback: kolom terakhir dianggap target
    return df.columns[-1]


def detect_positive_class(classes):
    """
    Coba tebak mana label yang berarti 'positif diabetes'.
    """
    lower = [str(c).lower() for c in classes]
    keywords = ["pos", "yes", "ya", "diab", "1", "true"]

    for kw in keywords:
        for c in lower:
            if kw in c:
                return c

    # kalau ada '1' dan '0'
    if "1" in lower and "0" in lower:
        return "1"

    # fallback: kelas terakhir
    return lower[-1]


@st.cache_resource
def load_dataset():
    if not os.path.exists(DATA_CSV):
        st.error(f"File dataset tidak ditemukan: {DATA_CSV}")
        st.stop()
    df = pd.read_csv(DATA_CSV)
    return df


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["pipeline"], obj["target_col"], obj["num_cols"], obj["cat_cols"], obj["classes"]

    df = load_dataset()
    target_col = get_target_col(df)

    # pisahkan fitur & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # deteksi fitur numerik & kategorikal
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # preprocessing
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # model RandomForest (bisa kamu samakan parameternya dengan di notebook)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf),
    ])

    # train di seluruh data (simple, biar sama dengan hasil analisismu kurang lebih)
    pipeline.fit(X, y)

    classes = pipeline.classes_.tolist()

    os.makedirs("models", exist_ok=True)
    obj = {
        "pipeline": pipeline,
        "target_col": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "classes": classes,
    }
    joblib.dump(obj, MODEL_PATH)

    return pipeline, target_col, num_cols, cat_cols, classes


# ========== MAIN APP ==========

def main():
    with st.spinner("Loading model & dataset..."):
        pipeline, target_col, num_cols, cat_cols, classes = load_or_train_model()
        df = load_dataset()


    # --- siapkan default dari dataset ---

    num_defaults = {}
    num_ranges = {}
    cat_choices = {}
    cat_defaults = {}

    df_feat = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

    # numeric
    for col in num_cols:
        if col in df_feat.columns:
            desc = df_feat[col].describe()
            median = float(desc["50%"])
            min_val = float(desc["min"])
            max_val = float(desc["max"])
        else:
            median, min_val, max_val = 0.0, 0.0, 1.0
        num_defaults[col] = median
        num_ranges[col] = (min_val, max_val)

    # categorical
    for col in cat_cols:
        if col in df_feat.columns:
            uniques = sorted(df_feat[col].dropna().unique().tolist())
            if not uniques:
                uniques = ["Unknown"]
        else:
            uniques = ["Unknown"]
        cat_choices[col] = uniques
        cat_defaults[col] = uniques[0]

    # --- layout ---
    col1, col2 = st.columns([1.1, 1.3], gap="medium")

    # ---------- INPUT ----------
    with col1:
        st.subheader("📝 Input Data Pasien")

        input_data = {}

        if num_cols:
            for col in num_cols:
                default = float(num_defaults[col])
                min_val, max_val = num_ranges[col]
                step = (max_val - min_val) / 100.0 if max_val > min_val else 0.1

                input_data[col] = st.number_input(
                    label=col,
                    value=default,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    step=step,
                    format="%.4f",
                    key=f"num_{col}",
                )

        if cat_cols:
            for col in cat_cols:
                choices = cat_choices[col]
                default = cat_defaults[col]
                input_data[col] = st.selectbox(
                    label=col,
                    options=choices,
                    index=choices.index(default) if default in choices else 0,
                    key=f"cat_{col}",
                )

        predict_btn = st.button("🔍 Prediksi Risiko Diabetes", type="primary", use_container_width=True)

    # ---------- OUTPUT ----------
    with col2:
        st.subheader("📊 Hasil Prediksi")

        if predict_btn:
            try:
                # susun dataframe satu baris
                input_df = pd.DataFrame([input_data])

                # prediksi dari pipeline (preprocess + RF)
                pred = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0]

                classes_arr = [str(c) for c in classes]
                lower_classes = [c.lower() for c in classes_arr]

                pos_label_lower = detect_positive_class(classes_arr)
                if pos_label_lower in lower_classes:
                    pos_idx = lower_classes.index(pos_label_lower)
                else:
                    pos_idx = int(np.argmax(proba))

                risk_prob = float(proba[pos_idx])
                confidence = float(np.max(proba))

                st.write(f"**Label prediksi model:** `{pred}`")
                st.metric("Confidence tertinggi model", f"{confidence:.1%}")

                # interpretasi HIGH / LOW risk
                if lower_classes[pos_idx] == str(pred).lower():
                    if risk_prob >= 0.5:
                        st.error("⚠️ HIGH RISK - Model mengindikasikan **risiko diabetes tinggi**.")
                    else:
                        st.warning(
                            "⚠️ Model memilih kelas positif, tapi probabilitasnya tidak terlalu besar.\n"
                            "Perlu kehati-hatian dalam interpretasi."
                        )
                else:
                    st.success("✅ LOW RISK - Model cenderung ke kelas non-diabetes / negatif.")

                st.write(f"**Probabilitas kelas yang dianggap 'positif':** `{risk_prob:.1%}`")

                with st.expander("🔎 Detail probabilitas semua kelas"):
                    prob_df = pd.DataFrame(
                        {"Class": classes_arr, "Probability": [float(p) for p in proba]}
                    )
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Terjadi error saat prediksi: {e}")

        else:
            st.info("👈 Isi form di kiri lalu klik **'Prediksi Risiko Diabetes'**.")



if __name__ == "__main__":
    main()
