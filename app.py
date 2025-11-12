# app.py
"""
Streamlit app to predict loan payback using a CatBoost .cbm model.
Place your files in the same folder:
 - catboost_model.cbm        (CatBoost native saved model)
 - preprocessor.joblib       (fitted ColumnTransformer / preprocessor saved with joblib)
 - label_encoders.joblib     (dict: {col: LabelEncoder})  - recommended
Run:
    pip install streamlit pandas numpy joblib catboost
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import joblib
import pickle

# ---------------- CONFIG ----------------
CATBOOST_CBM = "catboost_model.cbm"       # CatBoost model file
PREPROCESSOR_JOBLIB = "preprocessor.joblib"  # fitted preprocessor (ColumnTransformer)
# files to try for label encoders
LABEL_ENCODERS_FILES = ["label_encoders.joblib", "label_encoder.joblib", "label_encoders.pkl", "label_encoder.pkl"]
# Input features expected by your preprocessor/model (will be auto-detected if preprocessor exposes feature_names_in_)
FEATURES = ["annual_income","debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate", "marital_status","education_level","employment_status","loan_purpose","grade_subgrade"]
NUMERIC_FEATURES = ["annual_income","debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
THRESHOLD = 0.5
# ----------------------------------------

st.set_page_config(page_title="Loan Payback Predictor", layout="centered")
st.title("Loan Payback Predictor")

# ---------- loaders ----------
@st.cache_resource(show_spinner=False)
def load_catboost(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CatBoost model file not found: {path}")
    m = CatBoostClassifier()
    m.load_model(path)
    return m

@st.cache_resource(show_spinner=False)
def load_joblib_opt(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# load catboost
try:
    model = load_catboost(CATBOOST_CBM)
    st.success(f"Loaded CatBoost model from '{CATBOOST_CBM}'")
except Exception as e:
    st.error(f"Failed to load CatBoost model: {e}")
    st.stop()

# load preprocessor
preprocessor = load_joblib_opt(PREPROCESSOR_JOBLIB)
if preprocessor is None:
    st.error(f"Preprocessor file '{PREPROCESSOR_JOBLIB}' not found. This app expects a fitted preprocessor.")
    st.stop()
else:
    st.info(f"Loaded preprocessor: {PREPROCESSOR_JOBLIB}")

# try to load label encoders (dict mapping column->LabelEncoder)
def try_load_label_encoders(candidates):
    for f in candidates:
        if os.path.exists(f):
            try:
                obj = joblib.load(f)
                return obj, f
            except Exception:
                try:
                    with open(f, "rb") as fh:
                        obj = pickle.load(fh)
                        return obj, f
                except Exception:
                    continue
    return None, None

label_encoders_obj, label_file_loaded = try_load_label_encoders(LABEL_ENCODERS_FILES)
label_encoders = None
if label_encoders_obj is None:
    st.info("No label encoders file found. If you used LabelEncoder for 'grade_subgrade' and 'loan_purpose', save them as a dict and place the file here.")
else:
    if isinstance(label_encoders_obj, dict):
        label_encoders = label_encoders_obj
        st.success(f"Loaded label encoders dict from '{label_file_loaded}'. Keys: {list(label_encoders.keys())[:20]}")
    else:
        st.warning(f"Loaded an object from '{label_file_loaded}' but it's not a dict (type={type(label_encoders_obj)}). The app will not auto-apply encoders in this case.")
        label_encoders = None

# ---------- try to infer FEATURES from preprocessor ----------
def infer_features_from_preprocessor(pipe):
    try:
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
        if hasattr(pipe, "named_steps"):
            for name in ("preprocessor", "transformer", "ct", "column_transformer"):
                if name in pipe.named_steps:
                    prep = pipe.named_steps[name]
                    if hasattr(prep, "feature_names_in_"):
                        return list(prep.feature_names_in_)
                    if hasattr(prep, "get_feature_names_out"):
                        return list(prep.get_feature_names_out())
        if hasattr(pipe, "get_feature_names_out"):
            return list(pipe.get_feature_names_out())
    except Exception:
        pass
    return None

detected = infer_features_from_preprocessor(preprocessor)
if detected:
    FEATURES = detected
    st.success("Auto-detected FEATURES from preprocessor.")
else:
    st.info("FEATURES not auto-detected; using manual FEATURES variable (ensure order matches training).")

# require FEATURES
if not FEATURES:
    st.error("FEATURES is empty. Edit the FEATURES list to match your model's input columns.")
    st.stop()

# show features for verification
with st.expander("Model input features (edit FEATURES at top if incorrect)"):
    st.write(FEATURES)

# ---------- Build input form (selectboxes for categoricals) ----------
st.markdown("### Enter borrower information")
form = st.form(key="input_form")
user_inputs = {}

# Option lists (edit if your training categories differ in spelling/case)
marital_opts = ["Single", "Married", "Divorced", "Widowed"]
education_opts = ["Bachelor's", "High School", "Master's", "Other", "PhD"]
employment_opts = ["Employed", "Unemployed", "Self-employed", "Retired", "Student"]
loan_purpose_opts = ["Debt consolidation", "Other", "Car", "Home", "Education", "Business", "Medical", "Vacation"]
grade_subgrade_opts = [
    "C5","C4","C3","C2","C1",
    "D5","D4","D3","D2","D1",
    "B5","B4","B3","B2","B1",
    "E5","E4","E3","E2","E1",
    "F5","F4","F3","F2","F1",
    "A5","A4","A3","A2","A1"
]

NUMERIC_SUFFIXES = ("amt","amount","income","score","rate","ratio","years","age","total","num")

for col in FEATURES:
    label = col.replace("_", " ").title()

    # numeric fields
    if col in NUMERIC_FEATURES or any(col.lower().endswith(s) for s in NUMERIC_SUFFIXES):
        user_inputs[col] = form.number_input(label, value=0.0, format="%.3f")
        continue

    # categorical fields with selectbox
    if col == "marital_status":
        user_inputs[col] = form.selectbox(label, marital_opts, index=0)
        continue

    if col == "education_level":
        user_inputs[col] = form.selectbox(label, education_opts, index=0)
        continue

    if col == "employment_status":
        user_inputs[col] = form.selectbox(label, employment_opts, index=0)
        continue

    if col == "loan_purpose":
        user_inputs[col] = form.selectbox(label, loan_purpose_opts, index=0)
        continue

    if col == "grade_subgrade":
        user_inputs[col] = form.selectbox(label, grade_subgrade_opts, index=0)
        continue

    # fallback text input (rare)
    user_inputs[col] = form.text_input(label, value="")

submit = form.form_submit_button("Predict")

# ---------- helper: safe label-encoder application ----------
def apply_label_encoders_safe(X_df, encoders_dict):
    """
    Map known classes -> integer using encoder.classes_.
    Unknown classes -> -1. Returns transformed DF and list of warnings.
    """
    warnings = []
    X = X_df.copy()
    if not encoders_dict:
        return X, warnings

    for col, le in encoders_dict.items():
        if col not in X.columns:
            continue
        try:
            classes = list(le.classes_)
            mapping = {cls: idx for idx, cls in enumerate(classes)}
        except Exception:
            try:
                mapped_vals = le.transform(X[col].astype(str).fillna("___nan___"))
                X[col] = pd.Series(mapped_vals, index=X.index)
                continue
            except Exception:
                warnings.append(f"Encoder for '{col}' is incompatible; skipping.")
                continue

        mapped = X[col].map(mapping)
        unknown_mask = mapped.isna() & X[col].notna()
        if unknown_mask.any():
            warnings.append(f"Column '{col}' had {unknown_mask.sum()} unknown value(s); mapped to -1.")
        mapped = mapped.fillna(-1).astype(int)
        X[col] = mapped
    return X, warnings

# ---------- On submit: build DF, encode, transform, predict ----------
if submit:
    X = pd.DataFrame([user_inputs], columns=FEATURES)

    # apply label encoders (if available)
    enc_warnings = []
    if label_encoders is not None:
        try:
            X, enc_warnings = apply_label_encoders_safe(X, label_encoders)
        except Exception as e:
            st.error(f"Error applying label encoders: {e}")
            st.stop()
    else:
        # warn if expected label-encoded columns present but no encoders loaded
        if "grade_subgrade" in FEATURES or "loan_purpose" in FEATURES:
            st.warning("Label encoders for 'grade_subgrade' and/or 'loan_purpose' not found. Predictions may fail if preprocessor expects numeric codes.")

    # cast numeric columns
    for c in X.columns:
        if c in NUMERIC_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # preprocess -> model
    try:
        X_proc = preprocessor.transform(X)
    except Exception as e:
        st.error(f"Error during preprocessor.transform(X): {e}")
        st.stop()

    # sparse -> dense and array conversion
    if hasattr(X_proc, "toarray"):
        X_arr = X_proc.toarray()
    elif isinstance(X_proc, pd.DataFrame):
        X_arr = X_proc.values
    else:
        X_arr = np.asarray(X_proc)

    # predict probabilities
    try:
        probs = model.predict_proba(X_arr)
    except Exception as e:
        st.error(f"Error calling model.predict_proba: {e}")
        st.stop()

    # interpret probability
    try:
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
            p = float(probs[0, 1])
        elif isinstance(probs, np.ndarray) and probs.ndim == 1:
            p = float(probs[0])
        else:
            if hasattr(model, "decision_function"):
                score = model.decision_function(X_arr)
                p = float(1.0 / (1.0 + np.exp(-score[0])))
            else:
                pred = model.predict(X_arr)
                p = float(pred[0])
    except Exception as e:
        st.error(f"Could not interpret model output: {e}")
        st.stop()

    # show any encoder warnings
    for w in enc_warnings:
        st.warning(w + " Consider re-fitting encoder or restricting input choices.")

    will_pay = "Yes" if p >= THRESHOLD else "No"
    st.metric("Probability loan will be paid back", f"{p:.4f}")
    st.write(f"Binary decision at threshold {THRESHOLD}: **{will_pay}**")

    # display results & download
    res = X.copy()
    res["loan_paid_back_prob"] = p
    res["loan_paid_back"] = int(p >= THRESHOLD)
    st.write(res.T)

    csv = res.to_csv(index=False)
    st.download_button("Download result CSV", csv, file_name="prediction.csv", mime="text/csv")
