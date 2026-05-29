# Estimación de Trayectorias con Geometría Epipolar e ICP

Estimación y evaluación de trayectorias de cámara sobre el dataset **TUM RGB-D** mediante dos pipelines implementados desde cero:

1. **Geometría epipolar** (`freiburg1_xyz`) — estimación de F por algoritmos de 8 puntos, 7 puntos y RANSAC; descomposición de la matriz esencial E; encadenamiento de transformaciones SE(3).
2. **ICP** (`pioneer_slam`) — nubes de puntos desde mapas de profundidad, emparejamiento por vecino más próximo (KDTree), alineación rígida por SVD, iteración hasta convergencia.

Ambos pipelines se evalúan contra ground truth con **ATE** (mean/median/std/max/RMSE) y **RTE**.

---

## Instalación del entorno

Requisitos previos: **Python 3.12–3.13** y [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clonar el repositorio
git clone <url-del-repo> && cd Tray

# Instalar dependencias (crea el entorno virtual automáticamente)
uv sync

# Descargar las secuencias TUM necesarias
bash scripts/download_tum.sh
```

Las secuencias se descargan en `data/` (ignorado por git). Si se prefiere una descarga manual, las URLs están en el script.

---

## Uso rápido

```bash
# Ver opciones del CLI
uv run python -m tray --help

# Ejecutar una configuración concreta
uv run python -m tray run --config configs/epipolar_orb_8pt.yaml

# Ejecutar todos los experimentos
for cfg in configs/*.yaml; do uv run python -m tray run --config "$cfg"; done

# Tests y linter
uv run pytest -q
uv run ruff check src tests
```

Los resultados (métricas JSON, trayectoria `.npy`, gráficas PNG) se guardan en `results/<nombre_config>/`.

---

## Experimentos disponibles

| Config | Pipeline | Método |
|---|---|---|
| `epipolar_orb_8pt` | Epipolar | ORB + 8 puntos |
| `epipolar_orb_7pt` | Epipolar | ORB + 7 puntos |
| `epipolar_orb_ransac` | Epipolar | ORB + RANSAC |
| `epipolar_sift_lowe60` | Epipolar | SIFT + Lowe 0.60 |
| `epipolar_sift_lowe75` | Epipolar | SIFT + Lowe 0.75 |
| `icp_baseline` | ICP | Parámetros base |
| `icp_tight` | ICP | Umbral de rechazo estricto |
| `icp_loose_reject` | ICP | Umbral de rechazo amplio |
| `icp_voxel` | ICP | Submuestreo por vóxel |
| `icp_epipolar_init` | ICP | Inicialización epipolar |

---

## Arquitectura

```
src/tray/
├── data/           # Carga del dataset TUM RGB-D (asociaciones, GT, intrínsecos)
├── features/       # Detectores ORB/SIFT, emparejamiento BFMatcher + Lowe
├── epipolar/       # Estimación de F (8-pt, 7-pt, RANSAC), descomposición de E,
│                   # triangulación DLT, control de quiralidad, pipeline de trayectoria
├── icp/            # Nube de puntos desde profundidad, SVD, iteración ICP, pipeline
├── eval/           # Alineación Umeyama, ATE, RTE
├── viz/            # Gráficas 3D de trayectorias, visualización de matches
├── experiments/    # Runner YAML → pipeline → results/
└── cli.py          # Punto de entrada `python -m tray run`
configs/            # 10 configuraciones YAML reproducibles
tests/              # Tests unitarios con datos sintéticos; cross-check vs. OpenCV
```

El pipeline epipolar usa alineación Umeyama **con escala** (ambigüedad monocular); el pipeline ICP usa alineación **sin escala** (métrica). OpenCV se emplea exclusivamente para I/O, extracción de características y como referencia en tests — los algoritmos principales son implementaciones propias.
