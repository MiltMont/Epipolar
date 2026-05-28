# Descripción general del proyecto

*Lenguaje*: Python
*Dataset*: TUM RGB-D
*Métodos*: Geometría epipolar e ICP
*Producto final*: Código, gráficas e informe comparativo

En este proyecto se desarrollará un sistema de estimación de trayectorias de cámara a partir
de datos RGB-D. Se utilizarán dos enfoques geométricos con dos subconjuntos del conjunto
de datos TUM RGB-D.

- Estimación de movimiento mediante geometría epipolar con el subconjunto xyz.
- Estimación de movimiento mediante y registro geométrico mediante el algoritmo ICP
(Iterative Closest Point) con el subconjunto pioneer_slam.

## 1.1  Objetivo

El objetivo general es calcular, visualizar y evaluar trayectorias de cámara para los con-
juntos xyz y pioneer_slam del dataset TUM RGB-D, usando geometría epipolar e ICP, y
comparando el desempeño de cada método variando 5 parámetros.

## 1.2  Objetivos específicos

- Descargar y leer imágenes RGB, mapas de profundidad y trayectorias reales.
- Detectar y emparejar puntos de características usando ORB o SIFT.
- Implementar el cálculo de la matriz fundamental mediante los algoritmos de 8 puntos,
7 puntos y RANSAC con subconjuntos de 100 puntos.
- Recuperar movimiento relativo entre cámaras mediante la matriz esencial.
- Encadenar movimientos relativos para obtener una trayectoria global estimada.
- Implementar ICP para alinear nubes de puntos obtenidas de mapas de profundidad.
- Probar al menos cinco configuraciones de parámetros para geometría epipolar y cinco
configuraciones para ICP.
- Calcular errores absolutos y relativos de trayectoria.
- Visualizar trayectorias estimadas y trayectorias reales.
- Entregar un informe comparativo con resultados, análisis y conclusiones.
