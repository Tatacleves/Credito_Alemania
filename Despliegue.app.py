import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE MODELOS
# ==========================================
st.set_page_config(page_title="Credit Score Predictor", layout="wide")

@st.cache_resource
def cargar_recursos():
    # Cargamos el modelo y todos los transformadores guardados en el notebook
    modelo = joblib.load('modelo_red_neuronal1.joblib')
    scaler = joblib.load('scaler_model.joblib')
    ohe = joblib.load('ohe_model.joblib')
    pca = joblib.load('pca_model.joblib')
    # Cargamos el LabelEncoder para traducir la salida (0/1 -> Bad/Good)
    le = joblib.load('le_model.joblib')
    return modelo, scaler, ohe, pca, le

try:
    modelo, scaler, ohe, pca, le = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib. Asegúrate de que estén en la misma carpeta: {e}")

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🚀 Clasificación de Riesgo Crediticio")
st.markdown("Introduce los datos del cliente para evaluar si el riesgo es **Bueno** o **Malo**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=30)
    duration = st.number_input("Duración del Crédito (meses)", min_value=1, max_value=72, value=12)
    amount = st.number_input("Monto del Crédito", min_value=100, max_value=20000, value=1000)
    job = st.selectbox("Nivel de Trabajo (0: Unskilled, 3: Highly Skilled)", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Sexo", ["male", "female"])
    housing = st.selectbox("Vivienda", ["own", "rent", "free"])
    saving = st.selectbox("Cuentas de Ahorro", ["little", "moderate", "quite rich", "rich"])
    checking = st.selectbox("Cuenta Corriente", ["little", "moderate", "rich"])
    purpose = st.selectbox("Propósito del Crédito", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

# ==========================================
# 3. PROCESAMIENTO DE DATOS (PIPELINE)
# ==========================================
if st.button("Evaluar Crédito"):
    # A. Crear DataFrame con los inputs
    input_df = pd.DataFrame([{
        'Age': age, 
        'Credit amount': amount, 
        'Duration': duration, 
        'Job': job,
        'Sex': sex, 
        'Housing': housing, 
        'Saving accounts': saving, 
        'Checking account': checking, 
        'Purpose': purpose
    }])

    # B. Aplicar OneHotEncoding (Nominales)
    nominal_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    ohe_encoded = ohe.transform(input_df[nominal_cols])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(nominal_cols), index=input_df.index)

    # C. Unir con numéricas y eliminar originales categóricas
    df_procesado = pd.concat([input_df.drop(columns=nominal_cols), ohe_df], axis=1)

    # D. Escalado de variables numéricas
    num_cols = ['Age', 'Credit amount', 'Duration', 'Job']
    df_procesado[num_cols] = scaler.transform(df_procesado[num_cols])

    # E. Aplicar PCA (Reducción de dimensionalidad)
    # Importante: El PCA debe recibir las mismas columnas que recibió al entrenar
    datos_pca = pca.transform(df_procesado)

    # F. Predicción Final
    prediccion_num = modelo.predict(datos_pca)
    resultado = le.inverse_transform(prediccion_num)[0]

    # ==========================================
    # 4. MOSTRAR RESULTADO
    # ==========================================
    st.divider()
    if resultado == 'good':
        st.success(f"### Resultado: El cliente es APTO (Riesgo: {resultado.upper()}) ✅")
    else:
        st.error(f"### Resultado: El cliente es de ALTO RIESGO (Riesgo: {resultado.upper()}) ❌")
