
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
 
# Dossier où se trouve app.py (fonctionne peu importe d'où on lance streamlit)
BASE = Path(__file__).parent
 
# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diagnostic Diabète", page_icon="🩺", layout="centered")
 
st.markdown("""
<style>
h1 { color: #2E5FA3; }
.result-pos { background:#FFE8E8; border:2px solid #CC0000; border-radius:10px; padding:1rem; text-align:center; }
.result-neg { background:#E8FFE8; border:2px solid #009900; border-radius:10px; padding:1rem; text-align:center; }
.stButton>button { width:100%; background:#2E5FA3; color:white; font-size:1.1rem; border-radius:8px; padding:0.6rem; }
</style>
""", unsafe_allow_html=True)
 
# ── Charger les modèles ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Régression Logistique": joblib.load(BASE / "log_reg.pkl"),
        "Arbre de Décision":     joblib.load(BASE / "arbre.pkl"),
        "Forêt Aléatoire":       joblib.load(BASE / "forest.pkl"),
    }
 
@st.cache_resource
def load_preprocessor():
    df = pd.read_csv(BASE / "diabetes.csv")
    df = df.dropna(subset=['glyhb']).copy()
    df = df.drop(columns=['bp.2s', 'bp.2d'])
    for col in ['chol','hdl','ratio','height','weight','bp.1s','bp.1d','waist','hip','time.ppn']:
        df[col] = df[col].fillna(df[col].mean())
    df['frame'] = df['frame'].fillna(df['frame'].mode()[0])
    X = df.drop(columns=['id', 'glyhb'])
    num_cols = ['chol','stab.glu','hdl','ratio','age','height','weight','bp.1s','bp.1d','waist','hip','time.ppn']
    cat_cols = ['location','gender','frame']
    prep = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    prep.fit(X)
    return prep
 
models     = load_models()
preprocessor = load_preprocessor()
 
# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🩺 Diagnostic du Diabète")
st.markdown("Entrez les mesures du patient, choisissez un modèle, et obtenez la prédiction.")
st.divider()
 
# Sélection du modèle
model_choice = st.selectbox("🤖 Modèle à utiliser", list(models.keys()))
 
st.markdown("### 📋 Données du patient")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    age      = st.number_input("Âge",              min_value=1,   max_value=120, value=47)
    chol     = st.number_input("Cholestérol (mg/dL)", min_value=50, max_value=600, value=207)
    hdl      = st.number_input("HDL (mg/dL)",      min_value=10,  max_value=150, value=50)
    ratio    = st.number_input("Ratio Chol/HDL",   min_value=1.0, max_value=20.0,value=4.5, step=0.1)
 
with col2:
    stab_glu = st.number_input("Glycémie stable (mg/dL)", min_value=40, max_value=500, value=107)
    bp1s     = st.number_input("Pression systolique",  min_value=70,  max_value=250, value=137)
    bp1d     = st.number_input("Pression diastolique", min_value=40,  max_value=150, value=83)
    time_ppn = st.number_input("Temps post-prandial (min)", min_value=0, max_value=1600, value=336)
 
with col3:
    height   = st.number_input("Taille (pouces)",      min_value=48, max_value=84, value=66)
    weight   = st.number_input("Poids (livres)",        min_value=80, max_value=400, value=177)
    waist    = st.number_input("Tour de taille (po.)",  min_value=20, max_value=80,  value=38)
    hip      = st.number_input("Tour de hanches (po.)", min_value=25, max_value=80,  value=43)
 
col4, col5, col6 = st.columns(3)
with col4:
    location = st.selectbox("Localisation", ["Buckingham", "Louisa"])
with col5:
    gender   = st.selectbox("Genre", ["female", "male"])
with col6:
    frame    = st.selectbox("Morphologie", ["small", "medium", "large"])
 
st.divider()
 
if st.button("🔍 Prédire"):
    input_df = pd.DataFrame([{
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'height': height, 'weight': weight,
        'bp.1s': bp1s, 'bp.1d': bp1d, 'waist': waist,
        'hip': hip, 'time.ppn': time_ppn,
        'location': location, 'gender': gender, 'frame': frame
    }])
 
    X_input   = preprocessor.transform(input_df)
    model     = models[model_choice]
    pred      = model.predict(X_input)[0]
    proba     = model.predict_proba(X_input)[0][1]
 
    if pred == 1:
        st.markdown(f"""
        <div class="result-pos">
            <h2 style="color:#CC0000">⚠️ DIABÉTIQUE</h2>
            <h3>Probabilité : {proba*100:.1f}%</h3>
            <p>Modèle utilisé : <b>{model_choice}</b> · Consultation médicale recommandée.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-neg">
            <h2 style="color:#009900">✅ NON DIABÉTIQUE</h2>
            <h3>Probabilité diabète : {proba*100:.1f}%</h3>
            <p>Modèle utilisé : <b>{model_choice}</b></p>
        </div>""", unsafe_allow_html=True)
 
    # Barre de probabilité
    st.markdown("#### Probabilité de diabète")
    st.progress(float(proba))
 
    # Résultats des 3 modèles
    st.markdown("#### Comparaison des 3 modèles")
    rows = []
    for name, m in models.items():
        p  = m.predict(X_input)[0]
        pr = m.predict_proba(X_input)[0][1]
        rows.append({
            "Modèle": name,
            "Prédiction": "🔴 Diabétique" if p == 1 else "🟢 Non diabétique",
            "Probabilité (%)": f"{pr*100:.1f}%"
        })
    st.table(pd.DataFrame(rows))