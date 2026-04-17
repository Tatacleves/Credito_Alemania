import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px  # Librería para gráficos interactivos

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="Credit Risk AI Real-Time", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE MODELOS
# ==========================================
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_red_neuronal1.joblib')
    scaler = joblib.load('scaler_model.joblib')
    ohe = joblib.load('ohe_model.joblib')
    pca = joblib.load('pca_model.joblib')
    le = joblib.load('le_model.joblib')
    return modelo, scaler, ohe, pca, le

modelo, scaler, ohe, pca, le = cargar_recursos()

# Inicializar historial en la sesión si no existe
if 'historial' not in st.session_state:
    st.session_state.historial = pd.DataFrame()

# ==========================================
# 3. LÓGICA DE PREDICCIÓN
# ==========================================
def procesar_y_predecir(df_input):
    try:
        nominal_cols = list(ohe.feature_names_in_)
        num_cols = list(scaler.feature_names_in_)
        
        # OHE y Escalado
        ohe_df = pd.DataFrame(ohe.transform(df_input[nominal_cols]), 
                              columns=ohe.get_feature_names_out(nominal_cols), index=df_input.index)
        df_num = df_input[num_cols].copy()
        df_num[num_cols] = scaler.transform(df_num)
        
        # PCA y Predicción
        df_final = pd.concat([df_num, ohe_df], axis=1)[list(pca.feature_names_in_)]
        preds = modelo.predict(pca.transform(df_final))
        return le.inverse_transform(preds)
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return None

# ==========================================
# 4. INTERFAZ PRINCIPAL
# ==========================================
st.title("🏦 Panel de Control de Riesgo Crediticio")
st.markdown("Análisis de datos en tiempo real")

# Sidebar para entradas rápidas o configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    if st.button("Limpiar Historial"):
        st.session_state.historial = pd.DataFrame()
        st.rerun()

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📝 Nueva Evaluación", "📈 Dashboard Real-Time", "📂 Carga masiva"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("Información Personal y Financiera", expanded=True):
            c1, c2 = st.columns(2)
            age = c1.slider("Edad", 18, 90, 30)
            sex = c2.selectbox("Sexo", ["male", "female"])
            job = c1.selectbox("Nivel Laboral", [0, 1, 2, 3])
            housing = c2.selectbox("Vivienda", ["own", "rent", "free"])
            
            c3, c4 = st.columns(2)
            amount = c3.number_input("Monto solicitado (USD)", 100, 20000, 1500)
            duration = c4.number_input("Meses de plazo", 1, 72, 24)
            purpose = st.selectbox("Propósito", ["radio/TV", "education", "furniture/equipment", "car", "business", "repairs", "vacation/others"])
            saving = c3.selectbox("Ahorros", ["little", "moderate", "quite rich", "rich"])
            checking = c4.selectbox("Cuenta corriente", ["little", "moderate", "rich"])

    if st.button("📊 REALIZAR PREDICCIÓN"):
        nuevo_dato = pd.DataFrame([{
            'Age': age, 'Sex': sex, 'Job': job, 'Housing': housing, 
            'Saving accounts': saving, 'Checking account': checking, 
            'Credit amount': amount, 'Duration': duration, 'Purpose': purpose
        }])
        
        resultado = procesar_y_predecir(nuevo_dato)
        if resultado is not None:
            nuevo_dato['Resultado'] = resultado[0]
            st.session_state.historial = pd.concat([st.session_state.historial, nuevo_dato], ignore_index=True)
            
            with col2:
                if resultado[0] == 'good':
                    st.success(f"### APTO ✅\nEl riesgo es BAJO.")
                else:
                    st.error(f"### NO APTO ❌\nEl riesgo es ALTO.")
                st.info(f"Monto: ${amount} | Plazo: {duration} meses")

with tab2:
    if not st.session_state.historial.empty:
        st.subheader("📊 Análisis de las Evaluaciones del Día")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Evaluados", len(st.session_state.historial))
        m2.metric("Aprobados", len(st.session_state.historial[st.session_state.historial['Resultado'] == 'good']))
        m3.metric("Monto Promedio", f"${st.session_state.historial['Credit amount'].mean():.2f}")
        
        c1, c2 = st.columns(2)
        # Gráfico de tarta de resultados
        fig_pie = px.pie(st.session_state.historial, names='Resultado', title="Distribución de Aprobación",
                         color='Resultado', color_discrete_map={'good':'#28a745', 'bad':'#dc3545'})
        c1.plotly_chart(fig_pie, use_container_width=True)
        
        # Gráfico de dispersión Monto vs Edad
        fig_scatter = px.scatter(st.session_state.historial, x="Age", y="Credit amount", color="Resultado",
                                 title="Relación Edad vs Monto Solicitado", size="Duration")
        c2.plotly_chart(fig_scatter, use_container_width=True)
        
        st.write("### 📄 Registro Detallado (Datos Reales)")
        st.dataframe(st.session_state.historial, use_container_width=True)
    else:
        st.warning("Aún no hay datos para mostrar. Realiza una predicción en la primera pestaña.")

with tab3:
    archivo = st.file_uploader("Sube un CSV para análisis masivo", type=["csv"])
    if archivo:
        df_bulk = pd.read_csv(archivo)
        if st.button("Procesar Todo"):
            res = procesar_y_predecir(df_bulk)
            if res is not None:
                df_bulk['Resultado'] = res
                st.write("### Resultados del Archivo")
                st.dataframe(df_bulk)
                # Opción para añadir al historial general
                if st.button("Sumar estos datos al Dashboard"):
                    st.session_state.historial = pd.concat([st.session_state.historial, df_bulk], ignore_index=True)
                    st.success("Dashboard actualizado.")
