# Detector de Matrículas — ALPR

Proyecto de detección y lectura automática de matrículas usando Python, OpenCV y un modelo OCR basado en CNN.

El sistema procesa una imagen o una carpeta de imágenes y ejecuta un pipeline completo:

1. Detecta posibles zonas de matrícula.
2. Segmenta los caracteres de la matrícula.
3. Clasifica cada carácter con un modelo OCR.
4. Devuelve el resultado por consola y, opcionalmente, en un archivo CSV.

---

## Estructura del proyecto

La estructura esperada del proyecto es:

```text
detector-matricules/
├── main.py
├── alpr/
│   ├── __init__.py
│   ├── config.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── io.py
│   ├── detector/
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── segmenter/
│   │   ├── __init__.py
│   │   └── segmenter.py
│   └── ocr/
│       ├── __init__.py
│       └── infer.py
├── data/
│   └── raw/
├── models/
│   └── char_cnn_best.pth
├── output/
└── resources/
    └── fonts/
```

Es importante ejecutar el programa desde la raíz del proyecto:

```bash
cd /home/vallu/UNI/detector-matricules
```

No se recomienda ejecutar el programa desde dentro de la carpeta `alpr/`, ya que puede provocar errores de imports.

---

## Archivo de configuración

El archivo principal de configuración está en:

```text
alpr/config.py
```

Este archivo contiene las rutas y parámetros principales del proyecto, por ejemplo:

```python
DATA_RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")
MODEL_CNN_PATH = MODELS_DIR / "char_cnn_best.pth"
OUTPUT_DIR = Path("output")
```

También contiene parámetros del detector, segmentador y OCR.

Para importar la configuración correctamente desde otros módulos, se debe usar:

```python
from alpr import config
```

No se debe usar:

```python
import config
```

a menos que `config.py` esté en la misma carpeta desde la que se ejecuta el programa.

---

## Instalación

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo:

```bash
source .venv/bin/activate
```

Instalar dependencias principales:

```bash
pip install opencv-python numpy torch torchvision
```

Si el proyecto tiene un archivo `requirements.txt`, se puede instalar todo con:

```bash
pip install -r requirements.txt
```

---

## Preparar las imágenes

Coloca las imágenes que quieras procesar dentro de:

```text
data/raw/
```

Formatos soportados:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
```

Ejemplo:

```text
data/raw/coche1.jpg
data/raw/coche2.png
```

---

## Ejecutar el programa

### Procesar una sola imagen

```bash
python main.py --input data/raw/coche1.jpg
```

### Procesar una carpeta completa

```bash
python main.py --input data/raw/
```

### Guardar resultados en CSV

```bash
python main.py --input data/raw/ --output output/results.csv
```

### Ejecutar en modo debug

```bash
python main.py --input data/raw/coche1.jpg --debug
```

### Mostrar la imagen con la matrícula detectada

```bash
python main.py --input data/raw/coche1.jpg --show
```

### Usar un modelo OCR concreto

```bash
python main.py --input data/raw/ --model models/char_cnn_best.pth
```

---

## Argumentos disponibles

| Argumento   | Obligatorio | Descripción                                           |
| ----------- | ----------: | ----------------------------------------------------- |
| `--input`   |          Sí | Ruta a una imagen o carpeta de imágenes.              |
| `--output`  |          No | Ruta donde guardar un CSV con los resultados.         |
| `--model`   |          No | Ruta al modelo OCR entrenado.                         |
| `--debug`   |          No | Activa el modo debug y muestra información adicional. |
| `--show`    |          No | Muestra visualmente los resultados con OpenCV.        |
| `--verbose` |          No | Activa logs más detallados.                           |

Ejemplo completo:

```bash
python main.py --input data/raw/ --output output/results.csv --debug
```

---

## Funcionamiento interno

El programa sigue estas fases:

### 1. Carga del modelo OCR

Primero se carga el modelo entrenado:

```python
model, _class_to_idx, idx_to_class, device = load_model(model_path)
```

Si no existe el modelo, el programa mostrará un error y pedirá entrenarlo primero.

Por defecto, el modelo se busca en:

```text
models/char_cnn_best.pth
```

---

### 2. Carga de imágenes

Si `--input` es una imagen, se procesa solo esa imagen.

Si `--input` es una carpeta, se procesan todas las imágenes válidas dentro de esa carpeta.

---

### 3. Detección de matrículas

El detector busca regiones candidatas a matrícula en la imagen.

En modo normal usa:

```python
detect_boxes(img_bgr)
```

En modo debug usa:

```python
detect_debug(img_bgr)
```

---

### 4. Segmentación de caracteres

Para cada zona candidata, el sistema intenta separar los caracteres de la matrícula.

En modo normal usa:

```python
segment(roi)
```

En modo debug usa:

```python
segmenta_caixa(roi)
```

Si no se encuentran caracteres válidos, esa región se descarta.

---

### 5. OCR por carácter

Cada carácter segmentado se pasa al modelo OCR:

```python
predict(c, model, idx_to_class, device)
```

El modelo devuelve:

```text
carácter predicho + confianza
```

Los caracteres con baja confianza pueden marcarse como `?`.

---

### 6. Resultado final

Para cada matrícula detectada, el programa guarda:

```text
archivo
matrícula detectada
número de caracteres
confianza media
posición de la caja detectada
```

Si se usa `--output`, estos resultados se guardan en CSV.

---

## Salida por consola

Ejemplo de salida:

```text
12:30:01  INFO     Carregant model OCR…
12:30:02  INFO     Processant 2 imatge(s)…
12:30:03  INFO       coche1.jpg                     1234ABC     conf=91%  (120 ms)
12:30:03  INFO       coche2.jpg                     [sense detecció]  (95 ms)
12:30:03  INFO     Resultat: 1/2 imatges amb matrícula detectada
```

---

## Salida CSV

Si se ejecuta:

```bash
python main.py --input data/raw/ --output output/results.csv
```

se genera un CSV con columnas como:

```csv
file,plate,n_chars,mean_conf,box
coche1.jpg,1234ABC,7,0.9123,"(120, 80, 210, 45)"
```

---

## Errores comunes

### Error: `ModuleNotFoundError: No module named 'config'`

Este error ocurre cuando algún archivo tiene:

```python
import config
```

La solución es cambiarlo por:

```python
from alpr import config
```

o, si el archivo está dentro de una subcarpeta de `alpr`, usar import relativo:

```python
from .. import config
```

---

### Error: `the following arguments are required: --input`

Este error aparece si se ejecuta:

```bash
python main.py
```

El argumento `--input` es obligatorio. Hay que indicar una imagen o carpeta:

```bash
python main.py --input data/raw/
```

---

### Error: no encuentra el modelo OCR

Si aparece un error relacionado con:

```text
models/char_cnn_best.pth
```

significa que el modelo no existe todavía o está en otra ruta.

Soluciones:

1. Entrenar primero el modelo OCR.
2. Copiar el modelo dentro de `models/`.
3. Pasar la ruta manualmente:

```bash
python main.py --input data/raw/ --model ruta/al/modelo.pth
```

---

## Notas importantes

* Ejecutar siempre desde la raíz del proyecto.
* Mantener `config.py` dentro de `alpr/`.
* Usar imports tipo:

```python
from alpr import config
```

* No ejecutar desde dentro de la carpeta `alpr/`.
* Comprobar que las carpetas tienen archivo `__init__.py`.

---

## Estado actual

El pipeline ya está preparado para:

* Cargar imágenes.
* Detectar posibles matrículas.
* Segmentar caracteres.
* Ejecutar OCR carácter por carácter.
* Mostrar resultados.
* Exportar resultados en CSV.

La parte de lectura final contextual de matrícula aparece preparada pero comentada:

```python
# from alpr.reader.reader import read_plate
# plate = read_plate(chars_v)
```

Actualmente el campo `plate` puede aparecer como valor temporal si esa parte todavía no está conectada.
