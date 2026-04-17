import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. CARGA DE RECURSOS
# ==========================================
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_red_neuronal1.joblib')
    scaler = joblib.load('scaler_model.joblib')
    ohe = joblib.load('ohe_model.joblib')
    pca = joblib.load('pca_model.joblib')
    le = joblib.load('le_model.joblib')
    return modelo, scaler, ohe, pca, le

try:
    modelo, scaler, ohe, pca, le = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}")

# ==========================================
# 2. INTERFAZ
# ==========================================
st.title("🚀 Clasificación de Riesgo Crediticio")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    duration = st.number_input("Duration", 1, 72, 12)
    amount = st.number_input("Credit amount", 100, 20000, 1000)
    job = st.selectbox("Job", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
    checking = st.selectbox("Checking account", ["little", "moderate", "rich"])
    purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "others"])

# ==========================================
# 3. PROCESAMIENTO (CORREGIDO)
# ==========================================
if st.button("Evaluar Crédito"):
    try:
        # 1. Crear el DataFrame inicial respetando los nombres exactos del entrenamiento
        input_df = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'Job': job,
            'Housing': housing,
            'Saving accounts': saving,
            'Checking account': checking,
            'Credit amount': amount,
            'Duration': duration,
            'Purpose': purpose
        }])

        # 2. PROCESAR CATEGÓRICAS (OHE)
        # Obtenemos las columnas que el OHE espera (esto evita el error de nombres)
        cols_ohe_esperadas = ohe.feature_names_in_
        df_nominal = input_df[cols_ohe_esperadas]
        
        ohe_encoded = ohe.transform(df_nominal)
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(cols_ohe_esperadas), index=input_df.index)

        # 3. PROCESAR NUMÉRICAS (SCALER)
        # Escalamos solo las columnas que el Scaler conoce
        cols_scaler_esperadas = scaler.feature_names_in_
        df_numericas = input_df[cols_scaler_esperadas].copy()
        df_numericas[cols_scaler_esperadas] = scaler.transform(df_numericas)

        # 4. CONCATENAR Y REORDENAR PARA PCA
        # El PCA espera una estructura específica (Numéricas escaladas + Dummies)
        df_completo = pd.concat([df_numericas, ohe_df], axis=1)
        
        # FORZAMOS el orden de columnas exacto que pide el PCA
        cols_pca_esperadas = pca.feature_names_in_
        df_para_pca = df_completo[cols_pca_esperadas]
        
        # 5. TRANSFORMACIÓN FINAL Y PREDICCIÓN
        datos_pca = pca.transform(df_para_pca)
        prediccion = modelo.predict(datos_pca)
        resultado = le.inverse_transform(prediccion)[0]

        # RESULTADO
        st.divider()
        if resultado == 'good':
            st.success(f"### RESULTADO: {resultado.upper()} ✅")
        else:
            st.error(f"### RESULTADO: {resultado.upper()} ❌")

    except Exception as e:
        st.error(f"Error técnico: {e}")
        st.info("Revisa si los nombres de las columnas en el código coinciden con tu notebook original.")
