# Instrucciones

## 2.1  Preparación del dataset

Descargar los conjuntos xyz y pioneer_slam del dataset TUM RGB-D que pueden encontrar
en la siguiente que pueden encontrar en la siguiente [liga](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Cada conjunto debe contener, según corresponda:

- imágenes RGB;
- imágenes de profundidad;
- archivo de trayectoria real o ground truth;
- archivos de asociación temporal, si están disponibles.

## 2.2  Geometría epipolar

### Detección de puntos característicos

Para cada par de imágenes consecutivas, detectar puntos característicos usando una de las
siguientes opciones:

1. ORB
2. SIFT

### Emparejamiento de descriptores

Implementar un procedimiento de emparejamiento de descriptores. Se recomienda incluir

- Distancia entre descriptores.
- Selección de los mejores N emparejamientos.
- Prueba de razón de Lowe, cuando aplique.

El resultado debe ser un conjunto de n correspondencias:

$$
\{ (x_i, x_i') \}_{i=1}^n
$$

donde $x_i$ es un punto de la primera imagen y $x_i'$ su correspondiente en la segunda imagen.

### Obtener la matriz fundamental del sistema mediante tres métodos distintos

- Algoritmo de 8 puntos.
- Algoritmo de 7 puntos.
- Algoritmo RANSAC con 100 puntos.

### Calcular la matriz esencial

Obtener la matriz de parámetros intrínsecos $K$ a partir de los datos de calibración proporcionados en los formatos de archivo ([liga](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)), calcular:

$$
E = K^T F K
$$

### Descomponer la matriz esencial

Descomponer E mediante SVD:

$$
E = U \Sigma V^T
$$

A partir de esta descomposición, obtener las posibles rotaciones y traslaciones relativas entre
cámaras.

### Usar triangulación para elegir la solución físicamente válida

La descomposición de $E$ produce varias soluciones posibles para $(R, t)$. Se debe seleccionar
la solución que coloque la mayor cantidad de puntos reconstruidos frente a ambas cámaras.

### Construcción de trayectoria

Componer transformaciones relativas entre imágenes consecutivas.

$$
\begin{bmatrix}
R_{k,k+1} & t_{k,k+1} \\
0 & 1
\end{bmatrix}  
$$

## 2.3  ICP para nubes de puntos

### Generar pares de nubes de puntos

A partir de mapas de profundidad consecutivos, generar dos nubes de puntos:

$$
P = \{p_i\}_{i=1}^m, Q=\{q_j\}_{j=1}^n
$$

La nube $P$ será transformada para alinearse con la nube $Q$.

### Encontrar vecinos más cercanos

Para cada punto transformado $p_i$, buscar su vecino más cercano en $Q$. Se puede implementar
una búsqueda directa o utilizar KDTree de SciPy.

### Estimar la transformación rígida óptima

Dadas las correspondencias (p_i, q_i), calcular los centroides:

$$
\bar{p} = \frac{1}{m} \sum_{i=1}^m p_i, \ \ \ \bar{q} = \frac{1}{m}\sum_{i=1}^m q_i
$$
Centrar los puntos:
$$
\bar{p_i}' = p_i -\bar{p}, \ \ \ \bar{q_i}' = q_i - \bar{q}.
$$
construir la matriz de covarianza:

$$
H = \sum_i \bar{p_i}' \bar{q_i}' ^T
$$

calcular la SVD:

$$
H = U \Sigma V^T
$$
entonces:
$$
R = VU^T, \ \ \ t = \bar{q} - R\bar{p}
$$
si $det(R) < 0$ , se toma la matriz $-R$.

### Iterar hasta convergencia

En cada iteración de ICP:

1. Transformar la nube origen.
2. Buscar vecinos más cercanos.
3. Rechazar correspondencias con distancia demasiado grande.
4. Estimar una nueva transformación rígida.
5. Actualizar la transformación acumulada.
6. Calcular el error medio.
7. Detener si el cambio de error es menor que una tolerancia.
