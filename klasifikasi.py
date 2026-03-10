import streamlit as st
import joblib

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

#library model 
from xgboost import XGBClassifier


# --- LOAD ASSETS ---
# Pastikan file-file ini ada di folder yang sama
leEncoders = joblib.load('label_encoders.pkl')
sc_X = joblib.load('scaler.pkl')
model = joblib.load('model.pkl') # Load model XGBoost kamu

st.title("Aplikasi Prediksi Flight Delay")

# --- UI INPUT USER ---
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", options=list(leEncoders['Airline'].classes_))
    route = st.selectbox("Route", options=list(leEncoders['Route'].classes_))
    day_of_week = st.slider("Day of Week (1-7)", 1, 7, 1)

with col2:
    time = st.number_input("Time (Minutes)", min_value=0, max_value=1440)
    length = st.number_input("Length of Flight", min_value=0)

# --- PROSES DATA ---
if st.button("Prediksi Sekarang"):
    # 1. Buat DataFrame dari input
    new_data = pd.DataFrame({
        'Time': [time],
        'Length': [length],
        'Airline': [airline],
        'Route': [route],
        'DayOfWeek': [day_of_week]
    })

    # 2. Label Encoding untuk kolom kategorikal
    for col in ['Airline', 'Route']:
        new_data[col] = leEncoders[col].transform(new_data[col])

    # 3. Scaling 
    new_data_scaled = sc_X.transform(new_data)

    # 4. Predict
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled) # untuk persentase kemungkinannya

    # --- HASIL ---
    if prediction[0] == 1:
        st.error(f"⚠️ Prediksi: **DELAY** (Kemungkinan: {probability[0][1]:.2%})")
        ##st.error(f"⚠️ Prediksi: **DELAY**")
    else:
        st.success(f"✅ Prediksi: **ON TIME** (Kemungkinan: {probability[0][0]:.2%})")

        ##st.success(f"✅ Prediksi: **ON TIME**")
