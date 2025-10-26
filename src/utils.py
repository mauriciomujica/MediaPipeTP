"""
Funciones auxiliares para procesamiento de imágenes y cálculos de métricas.
"""
import cv2
import numpy as np
from PIL import Image
import math


def pil_to_cv2(pil_image):
    """
    Convierte una imagen PIL a formato OpenCV (numpy array BGR).
    
    Args:
        pil_image (PIL.Image): Imagen en formato PIL
    
    Returns:
        numpy.ndarray: Imagen en formato OpenCV (BGR)
    """
    # Convertir PIL a RGB numpy array
    rgb_array = np.array(pil_image.convert('RGB'))
    # Convertir RGB a BGR (formato OpenCV)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(cv2_image):
    """
    Convierte una imagen OpenCV a formato PIL.
    
    Args:
        cv2_image (numpy.ndarray): Imagen en formato OpenCV (BGR)
    
    Returns:
        PIL.Image: Imagen en formato PIL (RGB)
    """
    # Convertir BGR a RGB
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convertir a PIL
    pil_image = Image.fromarray(rgb_array)
    return pil_image


def resize_image(image, max_width=800):
    """
    Redimensiona la imagen manteniendo el aspect ratio.
    
    Args:
        image (numpy.ndarray): Imagen OpenCV
        max_width (int): Ancho máximo deseado
    
    Returns:
        numpy.ndarray: Imagen redimensionada
    """
    alto, ancho = image.shape[:2]
    
    if ancho > max_width:
        ratio = max_width / ancho
        nuevo_ancho = max_width
        nuevo_alto = int(alto * ratio)
        image = cv2.resize(image, (nuevo_ancho, nuevo_alto))

    return image


def calcular_apertura_boca(landmarks, alto, ancho):
    """
    Calcula la apertura de la boca basada en la distancia entre labios.

    Args:
        landmarks: Objeto de landmarks de MediaPipe
        alto (int): Alto de la imagen
        ancho (int): Ancho de la imagen

    Returns:
        float: Distancia en píxeles
    """
    # Landmark 13: labio superior, 14: labio inferior
    punto_superior = landmarks.landmark[13]
    punto_inferior = landmarks.landmark[14]

    y1 = punto_superior.y * alto
    y2 = punto_inferior.y * alto

    distancia = abs(y2 - y1)
    return distancia


def calcular_apertura_ojos(landmarks, alto, ancho):
    """
    Calcula la apertura de los ojos (izquierdo y derecho).

    Args:
        landmarks: Objeto de landmarks de MediaPipe
        alto (int): Alto de la imagen
        ancho (int): Ancho de la imagen

    Returns:
        dict: Diccionario con aperturas de ojo izquierdo y derecho
    """
    # Ojo izquierdo: 159 (superior), 145 (inferior)
    # Ojo derecho: 386 (superior), 374 (inferior)
    ojo_izq_sup = landmarks.landmark[159]
    ojo_izq_inf = landmarks.landmark[145]
    ojo_der_sup = landmarks.landmark[386]
    ojo_der_inf = landmarks.landmark[374]

    y_izq_sup = ojo_izq_sup.y * alto
    y_izq_inf = ojo_izq_inf.y * alto
    y_der_sup = ojo_der_sup.y * alto
    y_der_inf = ojo_der_inf.y * alto

    apertura_izq = abs(y_izq_inf - y_izq_sup)
    apertura_der = abs(y_der_inf - y_der_sup)

    return {"izquierdo": apertura_izq, "derecho": apertura_der}


def calcular_inclinacion_cabeza(landmarks, alto, ancho):
    """
    Calcula la inclinación de la cabeza basada en la posición de los ojos.

    Args:
        landmarks: Objeto de landmarks de MediaPipe
        alto (int): Alto de la imagen
        ancho (int): Ancho de la imagen

    Returns:
        float: Ángulo de inclinación en grados
    """
    # Usar ojos izquierdo y derecho
    ojo_izq = landmarks.landmark[33]  # Centro del ojo izquierdo
    ojo_der = landmarks.landmark[263]  # Centro del ojo derecho

    x1 = ojo_izq.x * ancho
    y1 = ojo_izq.y * alto
    x2 = ojo_der.x * ancho
    y2 = ojo_der.y * alto

    # Calcular el ángulo con respecto a la horizontal
    delta_x = x2 - x1
    delta_y = y2 - y1
    angulo_radianes = math.atan2(delta_y, delta_x)
    angulo_grados = math.degrees(angulo_radianes)

    return angulo_grados