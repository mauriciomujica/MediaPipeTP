"""
Detector de landmarks faciales usando MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from .config import (
    FACE_MESH_CONFIG,
    LANDMARK_COLOR,
    LANDMARK_RADIUS,
    LANDMARK_THICKNESS,
)
from .utils import (
    calcular_apertura_boca,
    calcular_apertura_ojos,
    calcular_inclinacion_cabeza,
)


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar landmarks faciales.
    """

    def __init__(self):
        """Inicializa el detector de MediaPipe."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(**FACE_MESH_CONFIG)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect(self, image, visualization_mode="Puntos simples"):
        """
        Detecta landmarks faciales en la imagen con opciones de visualización.

        Args:
            image (numpy.ndarray): Imagen en formato BGR (OpenCV)
            visualization_mode (str): Modo de visualización ("Puntos simples", "Puntos + malla conectada", "Solo contornos principales", "Heatmap de densidad de puntos")

        Returns:
            tuple: (imagen_procesada, landmarks, info)
                - imagen_procesada: imagen con landmarks dibujados
                - landmarks: objeto de landmarks de MediaPipe
                - info: diccionario con información de detección
        """
        # Convertir BGR a RGB para MediaPipe
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        resultados = self.face_mesh.process(imagen_rgb)

        # Crear copia para dibujar
        imagen_con_puntos = image.copy()

        info = {
            "rostros_detectados": 0,
            "total_landmarks": 0,
            "deteccion_exitosa": False,
            "apertura_boca": 0,
            "apertura_ojos": {"izquierdo": 0, "derecho": 0},
            "inclinacion_cabeza": 0,
        }

        # Si se detectaron rostros
        if resultados.multi_face_landmarks:
            info["rostros_detectados"] = len(resultados.multi_face_landmarks)

            # Tomar el primer rostro
            rostro = resultados.multi_face_landmarks[0]
            info["total_landmarks"] = len(rostro.landmark)
            info["deteccion_exitosa"] = True

            # Dibujar landmarks según el modo
            alto, ancho = image.shape[:2]

            if visualization_mode == "Puntos simples":
                for punto in rostro.landmark:
                    coord_x_pixel = int(punto.x * ancho)
                    coord_y_pixel = int(punto.y * alto)

                    cv2.circle(
                        imagen_con_puntos,
                        (coord_x_pixel, coord_y_pixel),
                        LANDMARK_RADIUS,
                        LANDMARK_COLOR,
                        LANDMARK_THICKNESS,
                    )
            elif visualization_mode == "Puntos + malla conectada":
                self.mp_drawing.draw_landmarks(
                    imagen_con_puntos,
                    rostro,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            elif visualization_mode == "Solo contornos principales":
                self.mp_drawing.draw_landmarks(
                    imagen_con_puntos,
                    rostro,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=LANDMARK_COLOR,
                        thickness=LANDMARK_THICKNESS,
                        circle_radius=LANDMARK_RADIUS,
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=LANDMARK_COLOR, thickness=1
                    ),
                )
            elif visualization_mode == "Heatmap de densidad de puntos":
                imagen_con_puntos = self._create_heatmap(
                    imagen_con_puntos, rostro, ancho, alto
                )

            # Calcular métricas
            info["apertura_boca"] = calcular_apertura_boca(rostro, alto, ancho)
            info["apertura_ojos"] = calcular_apertura_ojos(rostro, alto, ancho)
            info["inclinacion_cabeza"] = calcular_inclinacion_cabeza(
                rostro, alto, ancho
            )

            return imagen_con_puntos, rostro, info

        # No se detectó rostro
        return imagen_con_puntos, None, info

    def _create_heatmap(self, image, landmarks, width, height):
        """
        Crea un heatmap de densidad basado en los landmarks.

        Args:
            image (numpy.ndarray): Imagen base
            landmarks: Objeto de landmarks de MediaPipe
            width (int): Ancho de la imagen
            height (int): Alto de la imagen

        Returns:
            numpy.ndarray: Imagen con heatmap superpuesto
        """
        # Crear una imagen en blanco para el heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Agregar Gaussian blobs en cada landmark
        for punto in landmarks.landmark:
            x = int(punto.x * width)
            y = int(punto.y * height)
            # Crear un kernel Gaussiano
            sigma = 10  # Ajustar para el tamaño del blob
            size = int(3 * sigma)
            if size % 2 == 0:
                size += 1
            kernel = cv2.getGaussianKernel(size, sigma)
            kernel_2d = kernel * kernel.T

            # Asegurar que las coordenadas estén dentro de los límites
            y_start = max(0, y - size // 2)
            y_end = min(height, y + size // 2 + 1)
            x_start = max(0, x - size // 2)
            x_end = min(width, x + size // 2 + 1)

            # Agregar al heatmap
            heatmap[y_start:y_end, x_start:x_end] += kernel_2d[
                size // 2 - (y - y_start) : size // 2 + (y_end - y),
                size // 2 - (x - x_start) : size // 2 + (x_end - x),
            ]

        # Normalizar y aplicar colormap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superponer el heatmap en la imagen original con transparencia
        alpha = 0.5
        result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return result

    def close(self):
        """Libera recursos del detector."""
        self.face_mesh.close()
