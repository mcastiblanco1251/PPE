import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import datetime
import os
from PIL import Image
import io
import base64
import time
import pandas as pd
import gdown

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Detección EPP",
    page_icon="🛡️",
    layout="wide"
)

# Configuración de EPP requerido
EPP_REQUERIDO = {
    'gafas': 'Gafas de seguridad',
    'casco': 'Casco de seguridad', 
    'mangas': 'Mangas de protección',
    'tapa_oidos': 'Protección auditiva',
    'botas': 'Botas de seguridad',
    'chaleco': 'Chaleco reflectivo',
    'guantes': 'Guantes de seguridad'
}

# Mapeo de clases YOLO (ajusta según tu modelo)
YOLO_CLASSES = {
    0: 'persona',
    1: 'gafas',
    2: 'casco',
    3: 'mangas', 
    4: 'tapa_oidos',
    5: 'botas',
    6: 'chaleco',
    7: 'guantes'
}

@st.cache_resource
def load_model():
    model_path = 'mi_modelo_pesado.pt'
    if not os.path.isfile(model_path):
        # URL de descarga directa de Google Drive
        url = 'https://drive.google.com/uc?export=download&id=1MwuDl0ccFeL6R9ZV70icywD49mU0dr_e'
        gdown.download(url, model_path, quiet=False)
    model = YOLO(model_path)
    return model

# Cargar el modelo
model_path = load_model()

class EPPDetector:
    def __init__(self, model_path):
        """Inicializar el detector de EPP"""
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = 0.5
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            self.model = None
    
    def detect_epp(self, image):
        """Detectar EPP en la imagen"""
        if self.model is None:
            return None, None
        
        results = self.model(image, conf=self.confidence_threshold)
        return results, image
    
    def analyze_compliance(self, results):
        """Analizar cumplimiento de EPP"""
        detected_epp = set()
        persons_detected = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in YOLO_CLASSES:
                        class_name = YOLO_CLASSES[class_id]
                        if class_name == 'persona':
                            persons_detected += 1
                        else:
                            detected_epp.add(class_name)
        
        # Verificar cumplimiento
        required_epp = set(EPP_REQUERIDO.keys())
        missing_epp = required_epp - detected_epp
        
        is_compliant = len(missing_epp) == 0 and persons_detected > 0
        
        return {
            'compliant': is_compliant,
            'persons_detected': persons_detected,
            'detected_epp': list(detected_epp),
            'missing_epp': list(missing_epp)
        }
    
    def draw_detections(self, image, results):
        """Dibujar las detecciones en la imagen"""
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Obtener coordenadas
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in YOLO_CLASSES:
                        class_name = YOLO_CLASSES[class_id]
                        
                        # Color según la clase
                        if class_name == 'persona':
                            color = (0, 255, 0)  # Verde para persona
                        else:
                            color = (255, 0, 0)  # Rojo para EPP
                        
                        # Dibujar rectángulo
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Dibujar etiqueta
                        label = f"{class_name}: {confidence:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated_image

def send_email_alert(image, analysis, sender_email,recipient_email, sender_password):
    """Enviar alerta por email"""
    try:
        # Configurar servidor SMTP (ajustar según tu proveedor)
        smtp_server = "smtp.gmail.com"  # Cambiar por tu servidor
        smtp_port = 587
        sender_email = sender_email  # Cambiar por tu email
        sender_password = sender_password  # Usar variable de entorno en producción
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"🚨 ALERTA EPP - Incumplimiento Detectado - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Cuerpo del mensaje
        missing_items = [EPP_REQUERIDO[item] for item in analysis['missing_epp']]
        detected_items = [EPP_REQUERIDO.get(item, item) for item in analysis['detected_epp']]
        
        body = f"""
        🚨 ALERTA DE SEGURIDAD - INCUMPLIMIENTO EPP DETECTADO 🚨
        
        Fecha y Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        DETALLES DEL INCUMPLIMIENTO:
        📊 Personas detectadas: {analysis['persons_detected']}
        ✅ EPP Detectado: {', '.join(detected_items) if detected_items else 'Ninguno'}
        ❌ EPP Faltante: {', '.join(missing_items)}
        
        ACCIÓN REQUERIDA:
        Por favor, verificar inmediatamente el cumplimiento de normas de seguridad en el área.
        
        Este mensaje fue generado automáticamente por el Sistema de Detección EPP.
        
        ---
        Departamento HSE
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Adjuntar imagen
        if image is not None:
            img_buffer = io.BytesIO()
            cv2.imwrite('temp_violation.jpg', image)
            with open('temp_violation.jpg', 'rb') as f:
                img_data = f.read()
            
            img_attachment = MIMEImage(img_data)
            img_attachment.add_header('Content-Disposition', 'attachment', filename='incumplimiento_epp.jpg')
            msg.attach(img_attachment)
            
            # Limpiar archivo temporal
            os.remove('temp_violation.jpg')
        
        # Enviar email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Error al enviar email: {e}")
        return False

def save_violation_record(image, analysis):
    """Guardar registro de violación"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"violation_{timestamp}.jpg"
    
    # Crear directorio si no existe
    os.makedirs("violations", exist_ok=True)
    
    # Guardar imagen
    filepath = os.path.join("violations", filename)
    cv2.imwrite(filepath, image)
    
    # Guardar registro en CSV
    record = {
        'timestamp': datetime.datetime.now(),
        'filename': filename,
        'persons_detected': analysis['persons_detected'],
        'missing_epp': ', '.join(analysis['missing_epp']),
        'detected_epp': ', '.join(analysis['detected_epp'])
    }
    
    csv_file = "violations/violation_log.csv"
    df = pd.DataFrame([record])
    
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    
    return filepath

def main():
    st.title("🛡️ Sistema de Detección de Elementos de Protección Personal (EPP)")
    st.markdown("---")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Cargar modelo
    model_path = '../Eppv8.pt' #st.sidebar.text_input("Ruta del modelo YOLO:", value='../Project3-PPE-Detection/Eppv8.pt')
    confidence = st.sidebar.slider("Umbral de confianza:", 0.1, 1.0, 0.5, 0.1)
    
    # Configuración de email
    st.sidebar.subheader("📧 Configuración Email")
    email_enabled = st.sidebar.checkbox("Activar alertas por email", value=False)
    recipient_email = st.sidebar.text_input("Email destinatario:", value="e-mail")
    sender_email = st.sidebar.text_input("Email salida:", value=" tu e-mail")
    sender_password=st.sidebar.text_input("Contraseña del mail:", value="contraseña tu e-mail", type="password")
    # Inicializar detector
    if 'detector' not in st.session_state:
        st.session_state.detector = EPPDetector(model_path)
        st.session_state.detector.confidence_threshold = confidence
    
    # Actualizar umbral de confianza
    st.session_state.detector.confidence_threshold = confidence
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📁 Análisis de Imagen","📷 Detección en Tiempo Real",  "📊 Historial"])
    
    with tab2:
        st.header("Detección en Tiempo Real")
        
        # Controles de cámara
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_camera = st.button("🎥 Iniciar Cámara", type="primary")
        
        with col2:
            stop_camera = st.button("⏹️ Detener Cámara")
        
        with col3:
            capture_frame = st.button("📸 Capturar Frame")
        
        # Placeholder para video
        video_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # Inicializar estado de cámara
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        if start_camera:
            st.session_state.camera_active = True
        
        if stop_camera:
            st.session_state.camera_active = False
        
        # Procesamiento de video en tiempo real
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)  # Usar cámara por defecto
            
            if not cap.isOpened():
                st.error("No se pudo acceder a la cámara")
            else:
                stframe = st.empty()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error al capturar frame")
                        break
                    
                    # Procesar frame
                    results, _ = st.session_state.detector.detect_epp(frame)
                    
                    if results:
                        # Analizar cumplimiento
                        analysis = st.session_state.detector.analyze_compliance(results)
                        
                        # Dibujar detecciones
                        annotated_frame = st.session_state.detector.draw_detections(frame, results)
                        
                        # Mostrar estado de cumplimiento
                        if not analysis['compliant'] and analysis['persons_detected'] > 0:
                            # Mostrar alerta
                            cv2.putText(annotated_frame, "¡INCUMPLIMIENTO DETECTADO!", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            missing_text = f"Falta: {', '.join([EPP_REQUERIDO[item] for item in analysis['missing_epp']])}"
                            cv2.putText(annotated_frame, missing_text, (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Mostrar frame
                        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
                        
                        # Capturar violación si se presiona el botón
                        if capture_frame and not analysis['compliant'] and analysis['persons_detected'] > 0:
                            # Guardar registro
                            filepath = save_violation_record(annotated_frame, analysis)
                            
                            # Enviar email si está habilitado
                            if email_enabled:
                                if send_email_alert(annotated_frame, analysis, recipient_email):
                                    st.success("✅ Alerta enviada por email exitosamente")
                                else:
                                    st.error("❌ Error al enviar email")
                            
                            st.success(f"📸 Violación registrada: {filepath}")
                            
                            # Mostrar alerta en la interfaz
                            with alert_placeholder.container():
                                st.error(f"""
                                🚨 **INCUMPLIMIENTO DETECTADO**
                                
                                👥 Personas: {analysis['persons_detected']}
                                
                                ❌ **EPP Faltante:**
                                {chr(10).join([f"• {EPP_REQUERIDO[item]}" for item in analysis['missing_epp']])}
                                
                                ✅ **EPP Detectado:**
                                {chr(10).join([f"• {EPP_REQUERIDO.get(item, item)}" for item in analysis['detected_epp']]) if analysis['detected_epp'] else "• Ninguno"}
                                """)
                    
                    time.sleep(0.1)  # Pequeña pausa para no sobrecargar
                
                cap.release()
    
    with tab1:
        st.header("Análisis de Imagen")
        st.markdown("Sube una imagen de una persona y detecta sus EPPs")

        uploaded_file = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Leer imagen
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Mostrar imagen original
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original")
                st.image(image,use_container_width=True)
            
            # Procesar imagen
            results, _ = st.session_state.detector.detect_epp(image_cv)
            
            if results:
                # Analizar cumplimiento
                analysis = st.session_state.detector.analyze_compliance(results)
                
                # Dibujar detecciones
                annotated_image = st.session_state.detector.draw_detections(image_cv, results)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Resultado del Análisis")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                # Mostrar resultados
                st.markdown("---")
                st.subheader("📊 Resultados del Análisis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Personas Detectadas", analysis['persons_detected'])
                
                with col2:
                    compliance_status = "✅ CUMPLE" if analysis['compliant'] else "❌ NO CUMPLE"
                    st.metric("Estado", compliance_status)
                
                with col3:
                    st.metric("EPP Detectados", len(analysis['detected_epp']))
                
                # Detalles del análisis
                if not analysis['compliant'] and analysis['persons_detected'] > 0:
                    st.error(f"""
                    🚨 **INCUMPLIMIENTO DETECTADO**
                    
                    ❌ **EPP Faltante:**
                    {chr(10).join([f"• {EPP_REQUERIDO[item]}" for item in analysis['missing_epp']])}
                    """)
                    
                    # Botón para registrar violación
                    if st.button("📝 Registrar Violación"):
                        filepath = save_violation_record(annotated_image, analysis)
                        
                        if email_enabled:
                            if send_email_alert(annotated_image, analysis, recipient_email):
                                st.success("✅ Violación registrada y alerta enviada por email")
                            else:
                                st.warning("⚠️ Violación registrada pero error al enviar email")
                        else:
                            st.success("✅ Violación registrada")
                
                if analysis['detected_epp']:
                    st.success(f"""
                    ✅ **EPP Detectado:**
                    {chr(10).join([f"• {EPP_REQUERIDO.get(item, item)}" for item in analysis['detected_epp']])}
                    """)
    
    with tab3:
        st.header("📊 Historial de Violaciones")
        
        # Verificar si existe el archivo de registro
        csv_file = "violations/violation_log.csv"
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader(f"Total de violaciones: {len(df)}")
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Fecha inicio:", value=df['timestamp'].min().date())
            
            with col2:
                end_date = st.date_input("Fecha fin:", value=df['timestamp'].max().date())
            
            # Filtrar datos
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df.loc[mask]
            
            # Mostrar tabla
            st.dataframe(filtered_df, use_container_width=True)
            
            # Estadísticas
            if len(filtered_df) > 0:
                st.subheader("📈 Estadísticas")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Violaciones en período", len(filtered_df))
                
                with col2:
                    avg_persons = filtered_df['persons_detected'].mean()
                    st.metric("Promedio personas/violación", f"{avg_persons:.1f}")
                
                with col3:
                    total_persons = filtered_df['persons_detected'].sum()
                    st.metric("Total personas involucradas", total_persons)
        
        else:
            st.info("No hay registros de violaciones aún.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🛡️ Sistema de Detección EPP | Desarrollado por SAINECO/p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()