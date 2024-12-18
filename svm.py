import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Inicialización de session_state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = None
if 'hour_ranges' not in st.session_state:
    st.session_state['hour_ranges'] = []
if 'selected_days' not in st.session_state:
    st.session_state['selected_days'] = []
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None

def load_and_process_data(file):
    """Carga y procesa el archivo CSV"""
    df = pd.read_csv(file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Day'] = df['Datetime'].dt.dayofweek
    
    dias = {
        0: 'Lunes',
        1: 'Martes',
        2: 'Miércoles',
        3: 'Jueves',
        4: 'Viernes',
        5: 'Sábado',
        6: 'Domingo'
    }
    df['DayName'] = df['Day'].map(dias)
    return df

def create_features(df):
    """Crea características para el modelo"""
    X = df[['Day', 'Hour', 'Minute', 'Kwh']].values
    return X

def check_anomaly(row, threshold, selected_days, hour_ranges):
    """Verifica si un registro es anómalo según los criterios definidos"""
    reasons = []
    is_anomaly = False
    
    # Verificar consumo
    if row['Kwh'] > threshold:
        reasons.append(f"Consumo > {threshold:.2f} kWh")
        is_anomaly = True
    
    # Verificar día
    if row['Day'] not in selected_days:
        reasons.append("Día no permitido")
        is_anomaly = True
    
    # Verificar hora
    if not any(start <= row['Hour'] <= end for start, end in hour_ranges):
        reasons.append("Fuera de horario")
        is_anomaly = True
    
    return is_anomaly, ', '.join(reasons) if reasons else 'Normal'

def train_model(X, threshold, hour_ranges, selected_days):
    """Entrena el modelo SVM considerando umbrales de hora y día"""
    y = np.ones(len(X))
    
    # Crear DataFrame temporal para facilitar el chequeo
    temp_df = pd.DataFrame({
        'Day': X[:, 0],
        'Hour': X[:, 1],
        'Minute': X[:, 2],
        'Kwh': X[:, 3]
    })
    
    # Aplicar criterios de anomalía
    for idx, row in temp_df.iterrows():
        is_anomaly, _ = check_anomaly(row, threshold, selected_days, hour_ranges)
        if is_anomaly:
            y[idx] = 0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_scaled, y)
    
    return svm, scaler

def update_model():
    """Actualiza el modelo con los parámetros actuales"""
    if st.session_state['training_data'] is not None:
        X_train = create_features(st.session_state['training_data'])
        model, scaler = train_model(
            X_train,
            st.session_state['threshold'],
            st.session_state['hour_ranges'],
            st.session_state['selected_days']
        )
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        return True
    return False

def format_dataframe(df):
    """Aplica formato al DataFrame para su visualización"""
    if 'Kwh' in df.columns:
        df['Kwh'] = df['Kwh'].round(2)
    if 'Porcentaje sobre umbral' in df.columns:
        df['Porcentaje sobre umbral'] = df['Porcentaje sobre umbral'].round(2)
    return df

def main():
    st.title("Detector de Anomalías en Consumo Energético")
    
    st.sidebar.header("Configuración")
    
    # Carga de archivo de entrenamiento
    st.sidebar.subheader("Archivo de Entrenamiento")
    training_file = st.sidebar.file_uploader("Cargar archivo CSV de entrenamiento", type=['csv'])
    
    if training_file is not None:
        df_train = load_and_process_data(training_file)
        st.session_state['training_data'] = df_train
        
        st.subheader("Datos de Entrenamiento")
        st.dataframe(format_dataframe(df_train), use_container_width=True)
        
        # Configuración de filtros
        st.sidebar.subheader("Configuración de Filtros")
        
        # 1. Filtro de consumo
        max_kwh = float(df_train['Kwh'].max())
        threshold = st.sidebar.slider(
            "Umbral de consumo normal (kWh)",
            min_value=0.0,
            max_value=max_kwh,
            value=max_kwh/2,
            step=0.1,
            key='threshold_slider'
        )
        st.session_state['threshold'] = threshold
        
        # 2. Filtro de días
        dias_semana = {
            'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
            'Viernes': 4, 'Sábado': 5, 'Domingo': 6
        }
        selected_days = st.sidebar.multiselect(
            "Días considerados normales",
            options=list(dias_semana.keys()),
            default=list(dias_semana.keys())[:5],
            key='days_multiselect'
        )
        st.session_state['selected_days'] = [dias_semana[day] for day in selected_days]
        
        # 3. Filtro de horas
        st.sidebar.subheader("Rangos de Horas Normales")
        num_ranges = st.sidebar.number_input("Número de rangos horarios", min_value=1, value=1, key='num_ranges')
        hour_ranges = []
        
        for i in range(num_ranges):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start = st.number_input(f"Inicio rango {i+1}", min_value=0, max_value=24, value=8, key=f'start_{i}')
            with col2:
                end = st.number_input(f"Fin rango {i+1}", min_value=0, max_value=24, value=18, key=f'end_{i}')
            if start < end:
                hour_ranges.append((start, end))
            else:
                st.sidebar.warning(f"El rango {i+1} no es válido (inicio debe ser menor que fin)")
        
        st.session_state['hour_ranges'] = hour_ranges
        
        # Botón para entrenar/actualizar modelo
        if st.sidebar.button("Entrenar/Actualizar Modelo"):
            with st.spinner("Entrenando modelo..."):
                if update_model():
                    st.success("Modelo entrenado exitosamente!")
                else:
                    st.error("Error al entrenar el modelo")
        
        # Mostrar configuración actual
        st.subheader("Configuración de Detección de Anomalías")
        st.write(f"• Consumo máximo normal: {threshold:.2f} kWh")
        st.write(f"• Días normales: {', '.join(selected_days)}")
        st.write("• Rangos horarios normales:")
        for start, end in hour_ranges:
            st.write(f"  - {start:02d}:00 a {end:02d}:00")
        
        # Carga de archivo para predicción
        st.sidebar.subheader("Archivo para Predicción")
        prediction_file = st.sidebar.file_uploader("Cargar archivo CSV para predicción", type=['csv'])
        
        if prediction_file is not None and st.session_state['model'] is not None:
            df_pred = load_and_process_data(prediction_file)
            
            # Aplicar criterios de anomalía directamente
            anomaly_results = [
                check_anomaly(row, threshold, st.session_state['selected_days'], hour_ranges)
                for _, row in df_pred.iterrows()
            ]
            
            df_pred['Clasificación'] = ['Anómalo' if is_anomaly else 'Normal' 
                                      for is_anomaly, _ in anomaly_results]
            df_pred['Razón'] = [reason for _, reason in anomaly_results]
            
            # Mostrar resultados
            st.subheader("Resultados de la Predicción")
            display_columns = ['Datetime', 'DayName', 'Hour', 'Minute', 'Kwh', 'Clasificación', 'Razón']
            st.dataframe(format_dataframe(df_pred[display_columns]), use_container_width=True)
            
            # Estadísticas
            st.subheader("Estadísticas de Predicción")
            pred_normal = sum(df_pred['Clasificación'] == 'Normal')
            pred_anomaly = sum(df_pred['Clasificación'] == 'Anómalo')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Consumos Normales", pred_normal)
            with col2:
                st.metric("Consumos Anómalos", pred_anomaly)
            
            # Detalle de anomalías
            st.subheader("Detalle de Consumos Anómalos")
            df_anomalies = df_pred[df_pred['Clasificación'] == 'Anómalo'].copy()
            
            if not df_anomalies.empty:
                df_anomalies = df_anomalies.sort_values(by='Kwh', ascending=False)
                df_anomalies['Porcentaje sobre umbral'] = ((df_anomalies['Kwh'] - threshold) / threshold * 100)
                
                display_columns_anomalies = ['Datetime', 'DayName', 'Hour', 'Minute', 'Kwh', 'Porcentaje sobre umbral', 'Razón']
                st.dataframe(format_dataframe(df_anomalies[display_columns_anomalies]), use_container_width=True)
                
                # Mostrar estadísticas en columnas
                st.write("\nEstadísticas de consumos anómalos:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Consumo máximo", f"{df_anomalies['Kwh'].max():.2f} kWh")
                with col2:
                    st.metric("Consumo promedio", f"{df_anomalies['Kwh'].mean():.2f} kWh")
                with col3:
                    st.metric("Desviación estándar", f"{df_anomalies['Kwh'].std():.2f} kWh")
                
                # Distribución de anomalías
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Distribución por día de la semana:")
                    st.dataframe(df_anomalies['DayName'].value_counts(), use_container_width=True)
                
                with col2:
                    st.write("Distribución por hora:")
                    st.dataframe(df_anomalies['Hour'].value_counts().sort_index(), use_container_width=True)
            else:
                st.info("No se detectaron consumos anómalos en los datos.")
        elif prediction_file is not None:
            st.warning("Por favor, entrena primero el modelo usando el botón 'Entrenar/Actualizar Modelo'")

if __name__ == "__main__":
    main()
