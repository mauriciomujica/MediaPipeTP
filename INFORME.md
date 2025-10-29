# Informe: Detector de Landmarks Faciales con MediaPipe

## Introducción

### ¿Qué son los Landmarks Faciales?

Los landmarks faciales son puntos de referencia clave que mapean características anatómicas específicas del rostro humano. En el contexto de la visión por computadora, estos puntos representan coordenadas precisas que definen estructuras como:

- **Ojos**: Iris, párpados superior e inferior, cejas
- **Nariz**: Puente nasal, fosas nasales, punta
- **Boca**: Labios superior e inferior, comisuras
- **Contorno facial**: Línea de la mandíbula, frente, mejillas
- **Ceño**: Área entre las cejas

### Importancia de los Landmarks

Los landmarks faciales son fundamentales en aplicaciones modernas de IA:

1. **Filtros AR**: Permiten superponer elementos virtuales (como gafas o sombreros) de manera realista
2. **Análisis de expresiones**: Detectan emociones mediante cambios en la geometría facial
3. **Animación**: Crean avatares realistas que imitan movimientos faciales
4. **Autenticación biométrica**: Verifican identidad mediante comparación de rasgos faciales
5. **Medicina**: Ayudan en diagnósticos y tratamientos de condiciones faciales

MediaPipe, desarrollado por Google, detecta **478 puntos** con alta precisión, permitiendo aplicaciones avanzadas de visión por computadora.

## Arquitectura

### Estructura del Proyecto

El proyecto sigue una arquitectura modular organizada en los siguientes componentes:

```
MediaPipeTP/
├── app.py                 # Interfaz principal de Streamlit
├── src/
│   ├── detector.py        # Clase FaceLandmarkDetector
│   ├── config.py          # Configuraciones del modelo
│   └── utils.py           # Funciones auxiliares
├── requirements.txt       # Dependencias Python
└── packages.txt          # Paquetes del sistema
```

### Diagrama de Arquitectura

```mermaid
    A[Usuario sube imagen] --> B[Streamlit App]
    B --> C[PIL Image]
    C --> D[Conversión PIL→OpenCV]
    D --> E[Redimensionamiento]
    E --> F[FaceLandmarkDetector]
    F --> G[MediaPipe FaceMesh]
    G --> H[Procesamiento de landmarks]
    H --> I[Visualización según modo]
    I --> J[Cálculo de métricas]
    J --> K[Resultado final]
    K --> L[Display en Streamlit]

    subgrafico "Modos de Visualización"
        M1[Puntos simples]
        M2[Puntos + malla conectada]
        M3[Solo contornos principales]
        M4[Heatmap de densidad]
    fin

    I --> M1
    I --> M2
    I --> M3
    I --> M4
```

### Flujo de Datos

1. **Entrada**: Imagen subida por el usuario (PIL)
2. **Preprocesamiento**: Conversión a OpenCV, redimensionamiento
3. **Detección**: MediaPipe procesa la imagen y extrae landmarks
4. **Visualización**: Renderizado según el modo seleccionado
5. **Métricas**: Cálculo de aperturas y ángulos
6. **Salida**: Imagen procesada y métricas mostradas en la UI

## Decisiones de Diseño

### Separación Modular

Elegí una estructura modular por las siguientes razones:

- **Mantenibilidad**: Cada módulo tiene responsabilidades claras
- **Reutilización**: Componentes pueden usarse en otros proyectos
- **Testabilidad**: Funciones aisladas facilitan las pruebas unitarias
- **Escalabilidad**: Fácil agregar nuevas funcionalidades

### Configuración Centralizada

El archivo `config.py` centraliza todos los parámetros:

```python
FACE_MESH_CONFIG = {
    "static_image_mode": True,
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5
}
```

Esto permite ajustes sin modificar el código principal.

### Manejo de Imágenes

- **PIL ↔ OpenCV**: Conversión necesaria porque MediaPipe requiere formato específico
- **Redimensionamiento**: Optimización de rendimiento para imágenes grandes
- **Manejo de memoria**: Liberación explícita de recursos del detector

### Modos de Visualización Múltiples

Implementé 4 modos para diferentes casos de uso:

1. **Puntos simples**: Visualización básica y rápida
2. **Puntos + malla**: Vista completa con conexiones
3. **Contornos principales**: Enfoque en rasgos principales
4. **Heatmap**: Análisis de densidad de puntos

## Desafíos

### Conversión de Coordenadas

**Problema**: MediaPipe devuelve coordenadas normalizadas (0-1), pero OpenCV requiere píxeles absolutos.

**Solución**: Conversión explícita usando dimensiones de imagen:

```python
coord_x_pixel = int(punto.x * ancho)
coord_y_pixel = int(punto.y * alto)
```

### Cálculo de Métricas Faciales

**Problema**: Calcular distancias y ángulos precisos entre landmarks específicos.

**Solución**: Investigación de índices de landmarks de MediaPipe:

- Boca: landmarks 13 (superior) y 14 (inferior)
- Ojos: 159/145 (izquierdo), 386/374 (derecho)
- Inclinación: Ángulo entre centros de ojos

### Implementación del Heatmap

**Problema**: Crear una visualización de densidad que muestre concentración de landmarks.

**Solución**: Algoritmo de convolución Gaussiana:

```python
# Crear kernel Gaussiano
kernel = cv2.getGaussianKernel(size, sigma)
kernel_2d = kernel * kernel.T

# Superponer en imagen
result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
```

### Integración con Streamlit

**Problema**: Manejar estado y actualizaciones en tiempo real.

**Solución**: Arquitectura stateless con procesamiento por demanda.

## Uso de Kilo

Durante el desarrollo, Kilo Code me ayudó a implementar las siguientes funciones:

### Adición de Modos de Visualización

Kilo ayudó a implementar los diferentes modos de visualización en `detector.py`. El agente sugirió la estructura condicional y proporcionó el código para cada modo:

```python
if visualization_mode == "Puntos simples":
    # Dibujar círculos simples
elif visualization_mode == "Puntos + malla conectada":
    # Usar MediaPipe drawing utils
# ... etc
```

### Implementación de Métricas Faciales

Kilo colaboró en la creación de funciones en `utils.py` para calcular:

- Apertura de boca
- Apertura de ojos (izquierdo y derecho)
- Inclinación de cabeza

El agente proporcionó los índices correctos de landmarks y las fórmulas matemáticas.

### Capturas del Proceso

![Alt text](img\simples1.png?raw=true "Optional Title")
![Alt text](img\simples2.png?raw=true "Optional Title")
![Alt text](img\puntosmalla1.png?raw=true "Optional Title")
![Alt text](img\puntosmalla2.png?raw=true "Optional Title")
![Alt text](img\contornos1.png?raw=true "Optional Title")
![Alt text](img\contornos2.png?raw=true "Optional Title")
![Alt text](img\heatmap1.png?raw=true "Optional Title")
![Alt text](img\heatmap2.png?raw=true "Optional Title")

## Conclusiones

### Aprendizajes Principales

1. **Profundidad de MediaPipe**: La biblioteca ofrece precisión excepcional con 478 landmarks, pero requiere comprensión detallada de índices y coordenadas.

2. **Importancia de la Modularidad**: La separación en `detector.py`, `config.py` y `utils.py` facilitó el desarrollo incremental y las pruebas.

3. **Conversión de Formatos**: El manejo de PIL ↔ OpenCV es crucial en aplicaciones de visión por computadora con interfaces web.

4. **Visualización Creativa**: Los diferentes modos de visualización demuestran cómo una misma data puede presentarse de múltiples formas útiles.

5. **Integración AI en Desarrollo**: Kilo Code demostró ser una herramienta valiosa para acelerar el desarrollo, especialmente en tareas técnicas específicas como cálculos matemáticos y optimizaciones.

### Impacto del Proyecto

Este detector de landmarks faciales proporciona una base sólida para aplicaciones avanzadas de visión por computadora, desde filtros AR hasta análisis biométrico. La interfaz intuitiva de Streamlit lo hace accesible para usuarios no técnicos, mientras que la arquitectura modular permite extensiones futuras.

### Próximos Pasos

- Implementar detección en tiempo real con webcam
- Agregar análisis de expresiones emocionales
- Integrar con modelos de reconocimiento facial
- Optimizar rendimiento para dispositivos móviles

---

**Desarrollado en el Laboratorio 2 - IFTS24**