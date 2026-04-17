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
    # Cargamos los objetos guardados
    modelo = joblib.load('modelo_red_neuronal1.joblib')
    scaler = joblib.load('scaler_model.joblib')
    ohe = joblib.load('ohe_model.joblib')
    pca = joblib.load('pca_model.joblib')
    le = joblib.load('le_model.joblib')
    return modelo, scaler, ohe, pca, le

try:
    modelo, scaler, ohe, pca, le = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib: {e}")

# ==========================================
# 2. INTERFAZ DE USUARIO
# ==========================================
st.title("🚀 Clasificación de Riesgo Crediticio")
st.markdown("Introduce los datos del cliente para evaluar el riesgo crediticio.")

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
    # A. Crear DataFrame inicial
    # IMPORTANTE: El orden de las columnas debe ser igual al del entrenamiento
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

    # B. Aplicar OneHotEncoding (Nominales)
    # Extraemos solo las que el OHE conoce
    nominal_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    
    # Transformación OHE
    try:
        ohe_encoded = ohe.transform(input_df[nominal_cols])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(nominal_cols), index=input_df.index)
        
        # C. Unir con numéricas
        num_cols = ['Age', 'Credit amount', 'Duration', 'Job']
        df_final_pre_scale = pd.concat([input_df[num_cols], ohe_df], axis=1)

        # D. Escalado
        # El scaler espera las columnas numéricas en su orden
        df_final_pre_scale[num_cols] = scaler.transform(df_final_pre_scale[num_cols])

        # E. PCA
        # El PCA espera todas las columnas (numéricas escaladas + dummies) en el orden original del fit
        # Para asegurar el orden, reordenamos según las columnas que el PCA recuerda
        columnas_entrenamiento = pca.feature_names_in_
        df_para_pca = df_final_pre_scale[columnas_entrenamiento]
        
        datos_pca = pca.transform(df_para_pca)

        # F. Predicción
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
            
    except Exception as e:
        st.error(f"Error en el procesamiento de datos: {e}")
        st.info("Asegúrate de que los datos ingresados coincidan con las categorías del entrenamiento.")
