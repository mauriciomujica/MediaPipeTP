"""
Aplicación Streamlit para detección de landmarks faciales.
"""
import streamlit as st
from PIL import Image
from src.detector import FaceLandmarkDetector
from src.utils import pil_to_cv2, cv2_to_pil, resize_image
from src.config import TOTAL_LANDMARKS, VISUALIZATION_MODES


# Configuración de la página
st.set_page_config(
    page_title="Detector de Landmarks Faciales",
    layout="wide"
)

# Título y descripción
st.title("Detector de Landmarks Faciales")
st.markdown("""
Esta aplicación detecta **478 puntos clave** en rostros humanos usando MediaPipe.
Subí una imagen con un rostro y mirá la magia de la visión por computadora.
""")

# Sidebar con información
with st.sidebar:
    st.header("Información")
    st.markdown("""
    ### ¿Qué son los Landmarks?
    Son puntos de referencia que mapean:
    - Ojos (iris, párpados)
    - Nariz (puente, fosas)
    - Boca (labios, comisuras)
    - Contorno facial
    
    ### Aplicaciones
    - Filtros AR (Instagram)
    - Análisis de expresiones
    - Animación facial
    - Autenticación biométrica
    """)
    
    st.divider()
    st.caption("Desarrollado en el Laboratorio 2 - IFTS24")

# Selector de modo de visualización
visualization_mode = st.pills(
    "Modo de visualización",
    VISUALIZATION_MODES,
    help="Elige cómo visualizar los landmarks faciales",
    default=VISUALIZATION_MODES[0]
)

# Uploader de imagen
uploaded_file = st.file_uploader(
    "Subí una imagen con un rostro",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Cargar imagen
    imagen_original = Image.open(uploaded_file)
    
    # Convertir a formato OpenCV
    imagen_cv2 = pil_to_cv2(imagen_original)
    
    # Redimensionar si es muy grande
    imagen_cv2 = resize_image(imagen_cv2, max_width=800)
    
    # Columnas para mostrar antes/después
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(cv2_to_pil(imagen_cv2), use_container_width=True)
    
    # Detectar landmarks
    with st.spinner("Detectando landmarks faciales..."):
        detector = FaceLandmarkDetector()
        imagen_procesada, landmarks, info = detector.detect(imagen_cv2, visualization_mode)
        detector.close()
    
    with col2:
        st.subheader("Landmarks Detectados")
        st.image(cv2_to_pil(imagen_procesada), use_container_width=True)
    
    # Mostrar información de detección
    st.divider()
    
    if info["deteccion_exitosa"]:
        st.success("Detección exitosa")
        
        # Métricas
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Rostros detectados", info["rostros_detectados"])
        
        with metric_col2:
            st.metric("Landmarks detectados", f"{info['total_landmarks']}/{TOTAL_LANDMARKS}")
        
        with metric_col3:
            porcentaje = (info['total_landmarks'] / TOTAL_LANDMARKS) * 100
            st.metric("Precisión", f"{porcentaje:.1f}%")

        # Mostrar métricas adicionales
        st.subheader("Métricas Faciales")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Apertura Boca", f"{info['apertura_boca']:.2f} px")

        with metric_col2:
            st.metric("Apertura Ojo Izq", f"{info['apertura_ojos']['izquierdo']:.2f} px")

        with metric_col3:
            st.metric("Apertura Ojo Der", f"{info['apertura_ojos']['derecho']:.2f} px")

        # Inclinación de cabeza
        st.metric("Inclinación Cabeza", f"{info['inclinacion_cabeza']:.2f}°")
    else:
        st.error("No se detectó ningún rostro en la imagen")
        st.info("""
        **Consejos**:
        - Asegurate de que el rostro esté bien iluminado
        - El rostro debe estar mirando hacia la cámara
        - Probá con una imagen de mayor calidad
        """)

else:
    # Mensaje de bienvenida
    st.info("Subí una imagen para comenzar la detección")
    
    # Ejemplo visual
    st.markdown("### Ejemplo de Resultado")
    st.image(
        "https://ai.google.dev/static/mediapipe/images/solutions/face_landmarker_keypoints.png?hl=es-419",
        caption="MediaPipe detecta 478 landmarks faciales",
        width=400
    )