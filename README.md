# Efecto de la ventana y el solapamiento en características EEG

Análisis exploratorio (no clínico) de cómo la longitud de ventana (1, 2 y 4 s) y el solapamiento (0, 50 y 75 %) afectan a la extracción de características EEG en tres enfermedades: trastorno de estrés postraumático (TEPT), enfermedad de Alzheimer y depresión mayor. Parte del artículo OnTEPT (Salazar-Castro et al., 2026), que integra características EEG sin documentar con qué parámetros se calcularon ni qué tan fiables son.

**Autores:** Aldrin Josue Chavez Piguave, Jhony Alfonso Peñaherrera Loor, Marck Anthony Chiza Segovia y Fernando Chalanero Chantera Matango.

## Estado

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Selección de tres conjuntos de datos públicos | Hecho |
| 2 | Mismo preprocesamiento, 9 configuraciones de ventana, estabilidad (ICC) y tiempo | Hecho |
| 3 | Clasificador por conjunto de datos e interpretación con SHAP | Pendiente |

## Documentos

- Reporte de avance: [`Report/reporte.pdf`](Report/reporte.pdf)
- Diapositivas: [`Slides/diapositivas.pdf`](Slides/diapositivas.pdf)

## Datos (no incluidos en el repositorio)

| Conjunto | Pacientes / controles | Fuente | Licencia |
|---|---|---|---|
| TEPT (Rahmani et al., 2018) | 6 / 6 | figshare, doi:10.6084/m9.figshare.5393338 | CC BY 4.0 |
| Alzheimer (Miltiadous et al., 2023) | 36 / 29 | OpenNeuro ds004504 | CC0 |
| Depresión mayor (Mumtaz et al., 2017) | 28 / 27 | figshare, doi:10.6084/m9.figshare.4244171 | CC BY 4.0 |

Los conjuntos de datos nunca se mezclan en un mismo modelo: cada uno se registró con un equipo distinto.

## Reproducir

Requiere Python 3.11 y, para los documentos, una distribución de LaTeX con `latexmk`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -I descargar_datos.py      # ~3 GB en datos/
.venv/bin/python -I paso2_features.py       # resultados/paso2/: características, CSV y figuras
.venv/bin/python -I material_reporte.py     # figura de ejemplo y tablas del reporte
cd Report && latexmk -pdf reporte.tex
cd ../Slides && latexmk -pdf diapositivas.tex
```

## Estructura

| Ruta | Contenido |
|---|---|
| `descargar_datos.py` | Descarga y verifica los datos (tamaño y MD5); se puede reanudar |
| `paso2_features.py` | Armonización de la señal, ventanas, 228 características, estabilidad y tiempos |
| `material_reporte.py` | Figura de ejemplo y filas de las tablas en LaTeX, leídas de los CSV |
| `resultados/paso2/` | CSV de resultados, figuras y tablas (las características por ventana se regeneran) |
| `Report/` | Reporte en LaTeX y su PDF |
| `Slides/` | Diapositivas en LaTeX (Beamer) y su PDF |
| `Fundamental_papers/` | Artículo OnTEPT (acceso abierto, licencia CC BY-NC-ND 4.0) |
