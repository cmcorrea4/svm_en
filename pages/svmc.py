import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Inicialización de session_state
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = None
if 'hour_ranges' not in st.session_state:
    st.session_state['hour_ranges'] = []
if 'selected_days' not in st.session_state:
    st.session_state['selected_days'] = []
if 'config_ready' not in st.session_state:
    st.session_state['config_ready'] = False

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

def check_anomaly(row, threshold, selected_days, hour_ranges):
    """Verifica si un registro es anómalo según los criterios definidos"""
    reasons = []
    criteria_count = 0
    
    # Verificar consumo
    if row['Kwh'] > threshold:
        reasons.append(f"Consumo > {threshold:.2f} kWh")
        criteria_count += 1
    
    # Verificar día
    if row['Day'] not in selected_days:
        reasons.append("Día no permitido")
        criteria_count += 1
    
    # Verificar hora
    if not any(start <= row['Hour'] <= end for start, end in hour_ranges):
        reasons.append("Fuera de horario")
        criteria_count += 1
    
    # Es anómalo solo si cumple los tres criterios
    is_anomaly = criteria_count == 3
    reason_text = ', '.join(reasons) if reasons else 'Normal'
    
    return is_anomaly, reason_text, criteria_count

def update_config():
    """Actualiza la configuración y marca como lista para analizar"""
    st.session_state['config_ready'] = True

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
        selected_days_numeric = [dias_semana[day] for day in selected_days]
        st.session_state['selected_days'] = selected_days_numeric
        
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
        
        # Botón para actualizar configuración
        if st.sidebar.button("Actualizar Configuración"):
            with st.spinner("Actualizando configuración..."):
                update_config()
                st.success("Configuración actualizada exitosamente!")
        
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
        
        if prediction_file is not None and st.session_state['config_ready']:
            df_pred = load_and_process_data(prediction_file)
            
            # Aplicar criterios de anomalía directamente
            anomaly_results = [
                check_anomaly(row, st.session_state['threshold'], 
                            st.session_state['selected_days'], 
                            st.session_state['hour_ranges'])
                for _, row in df_pred.iterrows()
            ]
            
            df_pred['Clasificación'] = ['Anómalo' if is_anomaly else 'Normal' 
                                      for is_anomaly, _, _ in anomaly_results]
            df_pred['Razón'] = [reason for _, reason, _ in anomaly_results]
            df_pred['Criterios Cumplidos'] = [count for _, _, count in anomaly_results]
            
            # Mostrar resultados
            st.subheader("Resultados de la Predicción")
            display_columns = ['Datetime', 'DayName', 'Hour', 'Minute', 'Kwh', 'Clasificación', 'Razón', 'Criterios Cumplidos']
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
                
                display_columns_anomalies = ['Datetime', 'DayName', 'Hour', 'Minute', 'Kwh', 'Porcentaje sobre umbral', 'Razón', 'Criterios Cumplidos']
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
            st.warning("Por favor, actualiza la configuración usando el botón 'Actualizar Configuración'")

if __name__ == "__main__":
    main()
