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
import pandas as pd
#import tempfile
import gdown

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Detección EPP",
    page_icon="🛡️",
    layout="wide"
)

# Configuración de EPP requerido
EPP_REQUERIDO = {
    'Mask': 'Tapabocas',
    'Hardhat': 'Casco de seguridad', 
   
    
    
    'Safety Vest': 'Chaleco reflectivo',
    
}

# Mapeo de clases YOLO (ajusta según tu modelo)
YOLO_CLASSES = {0:'Hardhat', 1:'Mask', 2:'NO-Hardhat', 3:'NO-Mask', 4:'NO-Safety Vest', 5:'Person', 6:'Safety Cone', 7:'Safety Vest', 8:'machinery', 9:'vehicle'}
#{
#     0: 'botas',
#     1: 'gafas',
#     2: 'casco',
#     3: 'mangas', 
#     4: 'tapa_oidos',
#     5: 'persona',
#     6: 'persona',
#     7: 'guantes'
# }

@st.cache_resource
def load_model():
    model_path = 'Eppv8.pt'
    if not os.path.isfile(model_path):
        # URL de descarga directa de Google Drive
        url = 'https://drive.google.com/uc?export=download&id=1MwuDl0ccFeL6R9ZV70icywD49mU0dr_e'
        gdown.download(url, model_path, quiet=False)
    model = YOLO(model_path)
    return model

# Inicializar variables de sesión
if 'violation_log' not in st.session_state:
    st.session_state.violation_log = []

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

class EPPDetector:
    def __init__(self, model_path):
        """Inicializar el detector de EPP"""
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = 0.5
            st.success("✅ Modelo YOLO cargado exitosamente")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            st.info("Por favor, sube tu archivo de modelo YOLO (.pt)")
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
                        if class_name == 'Person':
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
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in YOLO_CLASSES:
                        class_name = YOLO_CLASSES[class_id]
                        
                        # Color según la clase
                        if class_name == 'persona':
                            color = (0, 255, 0)
                        else:
                            color = (255, 165, 0)
                        
                        # Dibujar rectángulo
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                        
                        # Dibujar etiqueta
                        label = f"{class_name}: {confidence:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated_image, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                        cv2.putText(annotated_image, label, (x1 + 5, y1 - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_image

def send_email_alert(image_bytes, analysis, recipient_email, sender_config):
    """Enviar alerta por email"""
    try:
        smtp_server = sender_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = sender_config.get('smtp_port', 587)
        sender_email = sender_config.get('sender_email')
        sender_password = sender_config.get('sender_password')
        
        if not all([sender_email, sender_password]):
            st.warning("⚠️ Configuración de email incompleta. No se enviará el correo.")
            return False
        
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Personas detectadas: {analysis['persons_detected']}
✅ EPP Detectado: {', '.join(detected_items) if detected_items else 'Ninguno'}
❌ EPP Faltante: {', '.join(missing_items)}

ACCIÓN REQUERIDA:
Por favor, verificar inmediatamente el cumplimiento de normas de seguridad en el área.

Este mensaje fue generado automáticamente por el Sistema de Detección EPP.

---
 Departamento HSE
Sistema de Monitoreo Automático
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Adjuntar imagen
        if image_bytes is not None:
            img_attachment = MIMEImage(image_bytes)
            img_attachment.add_header('Content-Disposition', 'attachment', 
                                     filename=f'incumplimiento_epp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
            msg.attach(img_attachment)
        
        # Enviar email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error al enviar email: {e}")
        return False

def save_violation_record(image_np, analysis):
    """Guardar registro de violación en memoria"""
    timestamp = datetime.datetime.now()
    
    # Convertir imagen a bytes para almacenar
    is_success, buffer = cv2.imencode(".jpg", image_np)
    image_bytes = buffer.tobytes()
    
    record = {
        'timestamp': timestamp,
        'datetime_str': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'persons_detected': analysis['persons_detected'],
        'missing_epp': ', '.join([EPP_REQUERIDO[item] for item in analysis['missing_epp']]),
        'detected_epp': ', '.join([EPP_REQUERIDO.get(item, item) for item in analysis['detected_epp']]),
        'image': image_bytes
    }
    
    st.session_state.violation_log.append(record)
    
    return record

def main():
    st.title("🛡️ Sistema de Detección de Elementos de Protección Personal (EPP)")
    #st.markdown("### AGP Glass - Sistema en la Nube")
    st.markdown("---")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        
        # Cargar modelo YOLO
        st.subheader("📦 Modelo YOLO")
        # uploaded_model = st.file_uploader("Cargar modelo YOLO (.pt)", type=['pt'])
        
        
        uploaded_model='Epp8v.pt'
        
        if uploaded_model is not None:
            model_path = load_model()
            # Guardar modelo temporalmente
            # with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            #     tmp_file.write(uploaded_model.read())
            #     model_path = tmp_file.name
            
            # Inicializar detector


            if 'detector' not in st.session_state or st.session_state.get('current_model') != uploaded_model:
                st.session_state.detector = EPPDetector(model_path)
                st.session_state.current_model = uploaded_model
                model_loaded = True
            else:
                model_loaded = True
        else:
            # Intentar usar modelo por defecto
            if os.path.exists('best.pt'):
                if 'detector' not in st.session_state:
                    st.session_state.detector = EPPDetector('best.pt')
                model_loaded = True
            else:
                st.warning("⚠️ Por favor, carga tu modelo YOLO entrenado")
        
        if model_loaded and hasattr(st.session_state, 'detector'):
            # Configuración de detección
            st.subheader("🎯 Parámetros de Detección")
            confidence = st.slider("Umbral de confianza:", 0.1, 1.0, 0.5, 0.05)
            st.session_state.detector.confidence_threshold = confidence
            

            # EPP Requerido
            st.subheader("✅ EPP Requerido")
            for key, value in EPP_REQUERIDO.items():
                st.checkbox(value, value=True, key=f"epp_{key}", disabled=True)
            
            st.markdown("---")
            
            # Configuración de email
            st.subheader("📧 Configuración Email")
            email_enabled = st.checkbox("Activar alertas por email", value=False)
            
            if email_enabled:
                recipient_email = st.text_input("Email destinatario:", 
                                               value="Persona HSE e-mail")
                
                with st.expander("⚙️ Configuración SMTP"):
                    smtp_server = st.text_input("Servidor SMTP:", value="smtp.gmail.com")
                    smtp_port = st.number_input("Puerto SMTP:", value=587)
                    sender_email = st.text_input("Email remitente:")
                    sender_password = st.text_input("Contraseña/Token:", type="password")
                    
                    st.info("""
                    💡 **Tip para Gmail:**
                    1. Activa la verificación en 2 pasos
                    2. Genera una "Contraseña de aplicación"
                    3. Usa esa contraseña aquí
                    """)
                    
                    sender_config = {
                        'smtp_server': smtp_server,
                        'smtp_port': smtp_port,
                        'sender_email': sender_email,
                        'sender_password': sender_password
                    }
            else:
                recipient_email = "infraccion_hse@agpglass.com"
                sender_config = {}
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 12px;'>
            <p>🛡️ Sistema EPP v2.0<br>Optimizado para Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Verificar si el modelo está cargado
    if not model_loaded:
        st.warning("⚠️ Por favor, carga tu modelo YOLO desde la barra lateral para comenzar")
        return
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Carga de Imágenes", "📸 Captura desde Navegador", "📊 Historial", "ℹ️ Información"])
    
    with tab1:
        st.header("📤 Análisis de Imágenes Cargadas")
        st.info("💡 Puedes cargar múltiples imágenes para análisis por lotes")
        
        uploaded_files = st.file_uploader(
            "Cargar imágenes para análisis", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.subheader(f"📁 {len(uploaded_files)} imagen(es) cargada(s)")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Imagen {idx + 1}: {uploaded_file.name}")
                
                # Leer imagen
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ Original")
                    st.image(image, use_container_width=True)
                
                # Procesar imagen
                with st.spinner('🔍 Analizando imagen...'):
                    results, _ = st.session_state.detector.detect_epp(image_cv)
                
                if results:
                    # Analizar cumplimiento
                    analysis = st.session_state.detector.analyze_compliance(results)
                    
                    # Dibujar detecciones
                    annotated_image = st.session_state.detector.draw_detections(image_cv, results)
                    
                    # Agregar banner de estado en la imagen
                    if not analysis['compliant'] and analysis['persons_detected'] > 0:
                        h, w = annotated_image.shape[:2]
                        cv2.rectangle(annotated_image, (0, 0), (w, 60), (0, 0, 255), -1)
                        cv2.putText(annotated_image, "INCUMPLIMIENTO DETECTADO!", (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    elif analysis['compliant'] and analysis['persons_detected'] > 0:
                        h, w = annotated_image.shape[:2]
                        cv2.rectangle(annotated_image, (0, 0), (w, 60), (0, 255, 0), -1)
                        cv2.putText(annotated_image, "CUMPLIMIENTO VERIFICADO", (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("🔍 Análisis")
                        st.image(annotated_image_rgb, use_container_width=True)
                    
                    # Mostrar resultados
                    st.markdown("---")
                    
                    # Métricas
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("👥 Personas", analysis['persons_detected'])
                    
                    with metric_col2:
                        compliance_status = "✅ CUMPLE" if analysis['compliant'] else "❌ NO CUMPLE"
                        compliance_color = "normal" if analysis['compliant'] else "inverse"
                        st.metric("Estado", compliance_status)
                    
                    with metric_col3:
                        st.metric("🛡️ EPP Detectados", len(analysis['detected_epp']))
                    
                    # Detalles
                    if not analysis['compliant'] and analysis['persons_detected'] > 0:
                        st.error(f"""
                        ### 🚨 INCUMPLIMIENTO DETECTADO
                        
                        **❌ EPP Faltante:**
                        {chr(10).join([f"• {EPP_REQUERIDO[item]}" for item in analysis['missing_epp']])}
                        """)
                        
                        # Acciones
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if st.button(f"📝 Registrar Violación", key=f"reg_{idx}"):
                                record = save_violation_record(annotated_image, analysis)
                                st.success("✅ Violación registrada en el historial")
                        
                        with col_b:
                            if email_enabled and st.button(f"📧 Enviar Alerta", key=f"email_{idx}"):
                                with st.spinner("Enviando email..."):
                                    is_success, buffer = cv2.imencode(".jpg", annotated_image)
                                    image_bytes = buffer.tobytes()
                                    
                                    if send_email_alert(image_bytes, analysis, recipient_email, sender_config):
                                        st.success("✅ Alerta enviada por email exitosamente")
                                        # También registrar
                                        save_violation_record(annotated_image, analysis)
                                    else:
                                        st.error("❌ Error al enviar email")
                    
                    elif analysis['persons_detected'] > 0:
                        st.success(f"""
                        ### ✅ CUMPLIMIENTO VERIFICADO
                        
                        **EPP Detectado:**
                        {chr(10).join([f"• {EPP_REQUERIDO.get(item, item)}" for item in analysis['detected_epp']])}
                        """)
                    else:
                        st.warning("⚠️ No se detectaron personas en la imagen")
                    
                    # Descargar imagen analizada
                    buf = io.BytesIO()
                    Image.fromarray(annotated_image_rgb).save(buf, format='JPEG')
                    st.download_button(
                        label="⬇️ Descargar imagen analizada",
                        data=buf.getvalue(),
                        file_name=f"analisis_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=f"download_{idx}"
                    )
                
                st.markdown("---")
    
    with tab2:
        st.header("📸 Captura desde Navegador")
        st.info("""
        💡 **Instrucciones:**
        1. Haz clic en "Tomar foto"
        2. Permite el acceso a la cámara en tu navegador
        3. Captura la imagen
        4. El sistema analizará automáticamente el EPP
        """)
        
        # Widget de cámara de Streamlit
        camera_photo = st.camera_input("📷 Tomar foto")
        
        if camera_photo is not None:
            # Procesar la foto capturada
            image = Image.open(camera_photo)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Foto Capturada")
                st.image(image, use_container_width=True)
            
            # Procesar imagen
            with st.spinner('🔍 Analizando en tiempo real...'):
                results, _ = st.session_state.detector.detect_epp(image_cv)
            
            if results:
                analysis = st.session_state.detector.analyze_compliance(results)
                annotated_image = st.session_state.detector.draw_detections(image_cv, results)
                
                # Banner de estado
                if not analysis['compliant'] and analysis['persons_detected'] > 0:
                    h, w = annotated_image.shape[:2]
                    cv2.rectangle(annotated_image, (0, 0), (w, 60), (0, 0, 255), -1)
                    cv2.putText(annotated_image, "INCUMPLIMIENTO DETECTADO!", (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                elif analysis['compliant'] and analysis['persons_detected'] > 0:
                    h, w = annotated_image.shape[:2]
                    cv2.rectangle(annotated_image, (0, 0), (w, 60), (0, 255, 0), -1)
                    cv2.putText(annotated_image, "CUMPLIMIENTO VERIFICADO", (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("🔍 Resultado")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                st.markdown("---")
                
                # Métricas
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("👥 Personas", analysis['persons_detected'])
                with m2:
                    status = "✅ CUMPLE" if analysis['compliant'] else "❌ NO CUMPLE"
                    st.metric("Estado", status)
                with m3:
                    st.metric("🛡️ EPP Detectados", len(analysis['detected_epp']))
                
                # Resultados detallados
                if not analysis['compliant'] and analysis['persons_detected'] > 0:
                    st.error(f"""
                    ### 🚨 INCUMPLIMIENTO DETECTADO
                    
                    **❌ EPP Faltante:**
                    {chr(10).join([f"• {EPP_REQUERIDO[item]}" for item in analysis['missing_epp']])}
                    """)
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        if st.button("📝 Registrar Violación", key="reg_cam"):
                            record = save_violation_record(annotated_image, analysis)
                            st.success("✅ Violación registrada")
                    
                    with col_y:
                        if email_enabled and st.button("📧 Enviar Alerta HSE", key="email_cam"):
                            with st.spinner("Enviando..."):
                                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                                image_bytes = buffer.tobytes()
                                
                                if send_email_alert(image_bytes, analysis, recipient_email, sender_config):
                                    st.success("✅ Alerta enviada")
                                    save_violation_record(annotated_image, analysis)
                
                elif analysis['persons_detected'] > 0:
                    st.success(f"""
                    ### ✅ CUMPLIMIENTO VERIFICADO
                    
                    **EPP Detectado:**
                    {chr(10).join([f"• {EPP_REQUERIDO.get(item, item)}" for item in analysis['detected_epp']])}
                    """)
    
    with tab3:
        st.header("📊 Historial de Violaciones")
        
        if st.session_state.violation_log:
            st.subheader(f"📋 Total de violaciones registradas: {len(st.session_state.violation_log)}")
            
            # Botones de acción
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🗑️ Limpiar historial"):
                    st.session_state.violation_log = []
                    st.rerun()
            
            with col_btn2:
                # Exportar a CSV
                if st.session_state.violation_log:
                    df_export = pd.DataFrame([{
                        'Fecha y Hora': r['datetime_str'],
                        'Personas': r['persons_detected'],
                        'EPP Faltante': r['missing_epp'],
                        'EPP Detectado': r['detected_epp']
                    } for r in st.session_state.violation_log])
                    
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv,
                        file_name=f"reporte_violaciones_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            
            # Mostrar cada registro
            for idx, record in enumerate(reversed(st.session_state.violation_log)):
                with st.expander(f"🚨 Violación #{len(st.session_state.violation_log) - idx} - {record['datetime_str']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Mostrar imagen
                        img_array = np.frombuffer(record['image'], dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                        **📅 Fecha:** {record['datetime_str']}  
                        **👥 Personas detectadas:** {record['persons_detected']}  
                        **❌ EPP Faltante:** {record['missing_epp']}  
                        **✅ EPP Detectado:** {record['detected_epp']}
                        """)
                        
                        # Descargar imagen individual
                        buf = io.BytesIO(record['image'])
                        st.download_button(
                            label="⬇️ Descargar imagen",
                            data=buf.getvalue(),
                            file_name=f"violacion_{record['datetime_str'].replace(':', '-').replace(' ', '_')}.jpg",
                            mime="image/jpeg",
                            key=f"dl_{idx}"
                        )
            
            # Estadísticas
            st.markdown("---")
            st.subheader("📈 Estadísticas Generales")
            
            total_persons = sum(r['persons_detected'] for r in st.session_state.violation_log)
            avg_persons = total_persons / len(st.session_state.violation_log) if st.session_state.violation_log else 0
            
            stat1, stat2, stat3 = st.columns(3)
            
            with stat1:
                st.metric("Total Violaciones", len(st.session_state.violation_log))
            
            with stat2:
                st.metric("Total Personas Involucradas", total_persons)
            
            with stat3:
                st.metric("Promedio Personas/Violación", f"{avg_persons:.1f}")
        
        else:
            st.info("📭 No hay violaciones registradas aún")
            st.markdown("""
            Las violaciones se registrarán automáticamente cuando:
            - Se detecte un incumplimiento de EPP
            - Se presione el botón "Registrar Violación"
            - Se envíe una alerta por email
            """)
    
    with tab4:
        st.header("ℹ️ Información del Sistema")
        
        st.markdown("""
        ### 🛡️ Sistema de Detección EPP
        
        **Versión:** 2.0 Cloud Edition  
        **Optimizado para:** Streamlit Cloud
        
        ---
        
        ### 📋 EPP Monitoreado
        """)
        
        for key, value in EPP_REQUERIDO.items():
            st.markdown(f"- ✅ **{value}**")
        
        st.markdown("""
        ---
        
        ### 🚀 Características
        
        1. **Análisis por Lotes**
           - Carga múltiples imágenes simultáneamente
           - Procesamiento automático de cada imagen
        
        2. **Captura desde Navegador**
           - Usa la cámara de tu dispositivo
           - Análisis en tiempo real
        
        3. **Alertas Automáticas**
           - Envío de emails al detectar incumplimientos
           - Registro fotográfico automático
        
        4. **Historial Completo**
           - Almacenamiento de todas las violaciones
           - Exportación a CSV
           - Descarga de imágenes
        
        5. **Compatible con Móviles**
           - Funciona en smartphones y tablets
           - Interfaz responsive
        
        ---
        
        ### 🔧 Configuración Recomendada
        
        #### Para Gmail:
        1. Ve a tu cuenta de Google
        2. Seguridad → Verificación en 2 pasos (actívala)
        3. Contraseñas de aplicaciones → Genera una nueva
        4. Usa esa contraseña en la configuración SMTP
        
        #### Para Outlook/Office 365:
        ```
        Servidor: smtp.office365.com
        Puerto: 587
        ```
        
        #### Para otros proveedores:
        Consulta la documentación de tu proveedor de email
        
        ---
        
        ### 📞 Soporte
        
        Para reportar problemas o sugerencias:
        - 📧 Email: infraccion_hse@agpglass.com
        - 🏢 Departamento: HSE - AGP Glass
        
        ---
        
        ### 🔒 Privacidad y Seguridad
        
        - Las imágenes se procesan en memoria
        - No se almacenan en servidor permanentemente
        - Los datos se pierden al cerrar la sesión (por seguridad)
        - Para almacenamiento permanente, descarga los reportes CSV
        
        ---
        
        ### 💡 Consejos de Uso
        
        ✅ **Recomendaciones:**
        - Usa buena iluminación para mejores resultados
        - Asegúrate que las personas estén completamente visibles
        - Mantén una distancia apropiada (2-5 metros)
        - Evita imágenes borrosas o con mucho movimiento
        
        ❌ **Evita:**
        - Ángulos muy cerrados o distantes
        - Contraluz fuerte
        - Obstrucciones parciales
        - Múltiples personas superpuestas
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 14px;'>
            🛡️ <b>Sistema de Detección EPP v2.0</b><br>
            AGP Glass - Departamento HSE<br>
            Desarrollado con ❤️ usando Streamlit + YOLO v8<br>
            <i>Optimizado para Streamlit Cloud</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()