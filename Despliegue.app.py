import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Credit Risk AI Analyzer",
    page_icon="💰",
    layout="wide"
)

# Estilo CSS personalizado para mejorar la estética
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE RECURSOS (MODELOS)
# ==========================================
@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load('modelo_red_neuronal1.joblib')
        scaler = joblib.load('scaler_model.joblib')
        ohe = joblib.load('ohe_model.joblib')
        pca = joblib.load('pca_model.joblib')
        le = joblib.load('le_model.joblib')
        return modelo, scaler, ohe, pca, le
    except Exception as e:
        st.error(f"Error cargando archivos del modelo: {e}")
        return None, None, None, None, None

modelo, scaler, ohe, pca, le = cargar_recursos()

# ==========================================
# 3. FUNCIÓN DE PROCESAMIENTO (LÓGICA CENTRAL)
# ==========================================
def predecir_riesgo(df_input):
    try:
        # Asegurar nombres de columnas exactos
        nominal_cols = list(ohe.feature_names_in_)
        num_cols = list(scaler.feature_names_in_)
        
        # OHE
        ohe_encoded = ohe.transform(df_input[nominal_cols])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(nominal_cols), index=df_input.index)
        
        # Scaler
        df_num = df_input[num_cols].copy()
        df_num[num_cols] = scaler.transform(df_num)
        
        # Unión y Reordenamiento para PCA
        df_full = pd.concat([df_num, ohe_df], axis=1)
        df_final = df_full[list(pca.feature_names_in_)]
        
        # Predicción
        datos_pca = pca.transform(df_final)
        preds = modelo.predict(datos_pca)
        return le.inverse_transform(preds)
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
        return None

# ==========================================
# 4. INTERFAZ DE USUARIO
# ==========================================
st.title("🏦 Sistema Inteligente de Riesgo Crediticio")
st.markdown("---")

tab1, tab2 = st.tabs(["📝 Formulario Individual", "📂 Carga Masiva (CSV)"])

# --- TAB 1: FORMULARIO INDIVIDUAL ---
with tab1:
    with st.container():
        st.subheader("Datos del Cliente")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Edad", 18, 100, 30)
            sex = st.selectbox("Sexo", ["male", "female"])
            job = st.selectbox("Nivel de Trabajo", [0, 1, 2, 3], help="0: No calificado, 3: Muy calificado")
            
        with c2:
            amount = st.number_input("Monto del Crédito", 100, 20000, 1000)
            duration = st.number_input("Duración (meses)", 1, 72, 12)
            purpose = st.selectbox("Propósito", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])
            
        with c3:
            housing = st.selectbox("Vivienda", ["own", "rent", "free"])
            saving = st.selectbox("Cuentas de Ahorro", ["little", "moderate", "quite rich", "rich"])
            checking = st.selectbox("Cuenta Corriente", ["little", "moderate", "rich"])

    if st.button("🚀 ANALIZAR PERFIL"):
        data = pd.DataFrame([{
            'Age': age, 'Sex': sex, 'Job': job, 'Housing': housing, 
            'Saving accounts': saving, 'Checking account': checking, 
            'Credit amount': amount, 'Duration': duration, 'Purpose': purpose
        }])
        
        resultado = predecir_riesgo(data)
        
        if resultado is not None:
            res = resultado[0].upper()
            if res == 'GOOD':
                st.balloons()
                st.markdown(f'<div class="result-card" style="background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;">'
                            f'<h2>✅ RIESGO BAJO</h2><p>El cliente es <b>APTO</b> para el crédito.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card" style="background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;">'
                            f'<h2>❌ RIESGO ALTO</h2><p>El perfil presenta alta probabilidad de <b>INCUMPLIMIENTO</b>.</p></div>', unsafe_allow_html=True)

# --- TAB 2: CARGA MASIVA ---
with tab2:
    st.subheader("Análisis de Cartera")
    archivo = st.file_uploader("Sube tu archivo CSV con múltiples clientes", type=["csv"])
    
    if archivo:
        df_bulk = pd.read_csv(archivo)
        st.write("Vista previa de los datos subidos:")
        st.dataframe(df_bulk.head())
        
        if st.button("📊 PROCESAR ARCHIVO"):
            resultados_masivos = predecir_riesgo(df_bulk)
            if resultados_masivos is not None:
                df_bulk['Resultado_Prediccion'] = resultados_masivos
                
                # Resumen visual
                c1, c2 = st.columns(2)
                resumen = df_bulk['Resultado_Prediccion'].value_counts()
                c1.metric("Clientes Aptos", resumen.get('good', 0))
                c2.metric("Clientes Riesgosos", resumen.get('bad', 0))
                
                st.success("¡Procesamiento completo!")
                st.dataframe(df_bulk)
                
                # Botón de descarga
                csv_download = df_bulk.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar resultados analizados", data=csv_download, file_name="predicciones_credito.csv", mime="text/csv")
