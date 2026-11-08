# Deteccion y clasificacion de embarcaciones en imagenes satelitales

[Read in English](README.md)

Un detector que encuentra cada embarcacion en imagenes satelitales opticas de muy alta resolucion, la encierra en una caja horizontal y la clasifica en uno de cinco tipos: lancha, buque de carga, barco pesquero, buque militar o embarcacion de servicio. Combina tres detectores YOLO26, aumento en inferencia con ocho orientaciones, fusion de cajas con Weighted Boxes Fusion y calibracion de confianza por clase. Todos los resultados de este repositorio se midieron con el evaluador oficial del reto.

## Logro

> **Tercer puesto** en el SpaceLAB Hackaton FAC + PUJ (GeoVision Colombia), Pontificia Universidad Javeriana, Bogota, 7 de noviembre de 2026.
>
> - Equipo: **DualCore** (Sergio Lopez y Santiago Rodriguez).
> - Mejor puntaje en el leaderboard publico: **0.5658** (Puntaje = 0.7 x mAP@0.5:0.95 + 0.3 x F1macro@0.5).
> - Posicion final en la particion privada: **tercer puesto**.

## El reto

El hackaton lo organizaron la Facultad de Ingenieria de la Pontificia Universidad Javeriana (PUJ) y la Fuerza Aeroespacial Colombiana (FAC). El trabajo esta ligado a un satelite de observacion de la Tierra que la FAC planea lanzar en 2027 para defensa y vigilancia maritima.

Colombia tiene costas sobre dos oceanos y un extenso territorio maritimo que vigilar. Saber que embarcaciones hay, donde estan y de que tipo son es clave para la seguridad y la defensa, la lucha contra la pesca ilegal y el contrabando, la busqueda y el rescate, y la gestion de puertos. Las imagenes satelitales de muy alta resolucion cubren grandes extensiones de mar y costa, pero revisarlas a mano es lento y costoso. El reto pedia un modelo que hiciera ese trabajo de forma automatica.

**Tarea.** Para cada imagen, producir una lista de cajas horizontales, cada una con una clase y una confianza entre 0 y 1.

| ID | Clase | Incluye |
| --- | --- | --- |
| 0 | small_boat | Lanchas, botes, yates y embarcaciones de recreo |
| 1 | cargo_ship | Portacontenedores, graneleros, tanqueros y barcazas de carga |
| 2 | fishing_vessel | Barcos pesqueros |
| 3 | warship | Fragatas, destructores, portaaviones y buques de apoyo naval |
| 4 | work_vessel | Remolcadores, dragas, gruas flotantes y buques de trabajo |

**Datos.** Los organizadores unificaron cuatro colecciones publicas de teledeteccion (FAIR1M, ShipRSImageNet, VHRShips y xView) en las cinco clases, con etiquetas en formato YOLO. Los datos no se incluyen en este repositorio.

| Conjunto | Imagenes | Objetos | Uso |
| --- | --- | --- | --- |
| train | 5,926 | 22,155 | Entrenamiento, con etiquetas |
| test | 801 | 3,964 | Validacion local, con etiquetas |
| oculto | 1,743 | 6,811 | Evaluacion oficial, sin etiquetas |

Caracteristicas clave, segun el documento del reto y nuestra auditoria de cada imagen y etiqueta:

- Los objetos son diminutos: un barco ocupa en promedio cerca del 0.3% de la imagen, y muchos menos del 0.05%.
- Hay escenas densas, con mas de 200 embarcaciones en una sola imagen (puertos, astilleros, bases navales).
- Las imagenes son JPEG RGB de 600 a 1,500 pixeles por lado, de distintos sensores, resoluciones y estilos de anotacion.
- La vista es cenital, asi que los barcos aparecen en cualquier orientacion.
- work_vessel es la clase minoritaria en entrenamiento (1,712 cajas, el 7.7% del total).

**Evaluacion.** Puntaje = 0.7 x mAP@0.5:0.95 + 0.3 x F1macro@0.5.

- El mAP es la precision promedio estilo COCO (interpolacion de 101 puntos) sobre umbrales de IoU de 0.50 a 0.95, promediada sobre las cinco clases. Usa todas las cajas enviadas, hasta 300 por imagen.
- El F1 solo cuenta las cajas con confianza igual o superior al umbral del equipo (`umbral.txt` en el envio), con IoU de 0.5, promediado sobre las cinco clases.
- El leaderboard mostraba el puntaje sobre un 30% aleatorio del conjunto oculto (523 imagenes). El ranking final tomo el mejor envio publico de cada equipo y lo califico sobre el 70% restante (1,220 imagenes).
- Hasta 15 envios, con al menos 20 minutos entre uno y otro. No se permitian datos etiquetados externos, en particular las cuatro colecciones de origen, porque contienen las imagenes del conjunto oculto.

## Por que importa

Un analista no puede revisar a ojo cada kilometro cuadrado de costa. Un detector que marca y clasifica cientos de embarcaciones en segundos le permite concentrarse en las detecciones relevantes, y los umbrales por clase que calibramos deciden que detecciones se convierten en alertas.

La solucion se construyo pensando en su despliegue en el satelite:

- Todos los detectores son modelos YOLO26 de una sola etapa, del extremo compacto de la familia. El modelo YOLO26s tiene cerca de 9.5 millones de parametros y 20.8 GFLOPs a 640 pixeles.
- Un solo modelo con una pasada por imagen es, por si mismo, una configuracion completa. La configuracion de maxima precision agrega modelos y orientaciones, y ambas cosas se pueden ajustar al computo disponible a bordo.
- La calibracion por clase no agrega computo: solo reescala las confianzas.
- La salida es simple y estandar: cajas horizontales, con una clase y una confianza por deteccion.
- Los modelos se entrenaron solo con los datos del reto, partiendo de pesos publicos de COCO.

## Nuestra solucion

El envio final se produce en cinco pasos:

1. **Tres detectores.** YOLO26m a 1280 pixeles (entrenado y luego reentrenado desde sus mejores pesos), un YOLO26s a 1024 pixeles entrenado en AWS y un YOLO26s a 1024 pixeles que entreno Santiago Rodriguez en Kaggle.
2. **Aumento en inferencia con ocho orientaciones.** Cada imagen se predice en las ocho orientaciones de un cuadrado: identidad, volteo horizontal, volteo vertical, rotacion de 180 grados, rotaciones de 90 y 270 grados, transpuesta y antitranspuesta. Las cajas se devuelven a la imagen original.
3. **Weighted Boxes Fusion (WBF).** Los 24 conjuntos de predicciones (3 modelos x 8 orientaciones) se fusionan por clase con un umbral de IoU de 0.70, confianzas promediadas y pesos por modelo de 1.5 para YOLO26m, 0.5 para el YOLO26s de AWS y 1.5 para el YOLO26s de Kaggle. Se conservan las 300 mejores cajas por imagen.
4. **Calibracion por clase.** En el conjunto de prueba se busca el umbral que maximiza el F1 de cada clase, y las confianzas de cada clase se reescalan con una funcion monotona lineal por tramos para que ese umbral coincida con un unico umbral global de 0.50. El orden dentro de cada clase no cambia, asi que el mAP se mantiene y el F1 sube.
5. **Formato y verificacion oficiales.** Los archivos se escriben con `formato_envio.py` de los organizadores y se revisan con su comando `verificar`.

Por que este enfoque funciona con estos datos:

- Los modelos se complementan. YOLO26m a 1280 encontro muchos mas barcos pesqueros, y los YOLO26s ajustaron mejor las cajas de lanchas y buques militares. Al fusionarlos, cada clase quedo por encima de cada modelo individual.
- En una imagen satelital no hay arriba ni abajo, pero el entrenamiento nunca mostro a los modelos rotaciones de 90 grados. Agregarlas en inferencia fue la mayor mejora individual despues del ensamble.
- WBF promedia las coordenadas de varias predicciones, lo que ajusta las cajas. Eso importa porque 7 de los 10 umbrales de IoU de la metrica son de 0.65 o mas estrictos.

## Como lo entrenamos

**Uso de los datos.** Los tres modelos se entrenaron con el mismo conjunto train. El conjunto test, que tiene etiquetas, sirvio como validacion durante el entrenamiento y como referencia local para cada decision. Como tambien se uso para elegir checkpoints y umbrales, los puntajes locales son algo optimistas. El conjunto oculto solo se uso para generar envios.

**Auditoria y preprocesamiento.** Antes de entrenar revisamos las 8,470 imagenes y las 26,119 cajas. Todas las imagenes eran JPEG RGB validas, sin archivos corruptos, sin imagenes vacias y sin etiquetas invalidas, asi que no se descarto ninguna. Las imagenes con bruma o muy claras se conservaron. La dimension de las cajas definio la resolucion: al llevar el lado largo a 640 pixeles, la mitad de las lanchas de test mide menos de 16 pixeles; a 1024 es el 10.2% y a 1280 el 2.9%. No se aplico preprocesamiento previo; Ultralytics redimensiona y rellena cada imagen a la resolucion de entrenamiento.

**Aumento de datos.** Volteos horizontales y verticales (probabilidad de 0.5 cada uno), mosaico (probabilidad de 1.0, desactivado en las ultimas epocas), variacion de color HSV (0.015 / 0.7 / 0.4), traslacion de 0.1 y escala de 0.5. Sin rotacion, cizallamiento ni mixup.

**Optimizacion.** Los cuatro checkpoints registran el optimizador AdamW con tasa de aprendizaje de 0.001111, factor final de tasa de aprendizaje de 0.01, momentum de 0.937, weight decay de 0.0005 y calendario coseno de tasa de aprendizaje; en las corridas de AWS fue la eleccion automatica de optimizador de Ultralytics. Todas las corridas usaron precision mixta y semilla 0, y partieron de pesos publicos de COCO o de nuestros propios checkpoints.

| Modelo | Punto de partida | Resolucion | Lote | Epocas | Tiempo de entrenamiento | Otros parametros | Script |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s (AWS) | yolo26s.pt (COCO) | 1024 | 16 | 34 | 57 min | Limite de 0.95 h, calentamiento de 3 epocas, patience 15, mosaico apagado en las ultimas 5 epocas | `entrenar_baseline.py` de los organizadores |
| YOLO26m | yolo26m.pt (COCO) | 1280 | 6 | 11 | 51 min | Limite de 0.85 h, calentamiento de 1 epoca, mosaico apagado en las ultimas 2 epocas | `scripts/train.py` |
| YOLO26m, continuacion | Mejores pesos de la corrida anterior | 1280 | 6 | 9 | 45 min | Limite de 0.75 h, mismos parametros | `scripts/train.py` |
| YOLO26s (Kaggle, Santiago Rodriguez) | yolo26s.pt (COCO) | 1024 | 8 | Sin registro (30 configuradas, detenido por el limite de tiempo) | 2.25 h | Limite de 2.25 h, calentamiento de 3 epocas, patience 15, mosaico apagado en las ultimas 5 epocas, Ultralytics 8.4.205 | Notebook de Kaggle, no incluido en este repositorio |

**Hardware.** AWS EC2 g5.xlarge (una NVIDIA A10G de 24 GB, 4 vCPU), Ultralytics 8.4.203, PyTorch 2.14.1 (version para CUDA 12.6) y Python 3.11. Durante el entrenamiento la GPU se mantuvo cerca de su uso maximo (entre 84% y 100% en nuestras mediciones). El modelo de Kaggle se entreno con el hardware de Kaggle.

**Evaluacion y seleccion.** Cada numero proviene de `evaluar.py` de los organizadores sobre el conjunto test, incluido su barrido de umbral (`--barrer-umbral`). Antes de cada envio tambien simulamos el leaderboard publico: 300 subconjuntos aleatorios del 30% del conjunto test, calificados con las funciones del evaluador oficial. Solo enviamos un candidato si superaba a nuestro mejor envio en al menos el 95% de ellos. El script de simulacion no esta incluido en este repositorio.

**Reproducir el entrenamiento.** Ejecutar desde la raiz del repositorio con los datos en su lugar (ver Instalacion). Con un limite de tiempo, el numero de epocas depende del hardware.

```bash
.venv/bin/python Entrega_Context/kit_participantes/entrenar_baseline.py --datos Entrega_Context/spacelab_dataset --modelo weights/yolo26s.pt --imgsz 1024 --horas 0.95 --batch 16 --workers 4 --proyecto runs/exp01 --nombre yolo26s_1024
.venv/bin/python scripts/train.py --model weights/yolo26m.pt --imgsz 1280 --hours 0.85 --batch 6 --workers 4 --project runs/exp03 --name yolo26m_1280
.venv/bin/python scripts/train.py --model runs/exp03/yolo26m_1280/weights/best.pt --imgsz 1280 --hours 0.75 --batch 6 --workers 4 --project runs/exp05 --name yolo26m_1280_ft
```

La configuracion del modelo de Kaggle coincide con el script de baseline de los organizadores usando `--modelo yolo26s.pt --imgsz 1024 --horas 2.25 --batch 8`.

## Recorrido de decisiones

Los pasos estan en el orden en que los tomamos. Los puntajes son locales, calculados con `evaluar.py` sobre test, salvo los marcados como publicos.

1. **Primero, un flujo completo.** Corrimos el baseline de los organizadores con el 5% de los datos de entrenamiento y una sola epoca en un portatil. El puntaje (0.0227) no tenia valor, y no se buscaba que lo tuviera: el objetivo era comprobar que entrenamiento, evaluacion, empaquetado y verificador oficial funcionaban de principio a fin. Tambien mostro que el portatil era demasiado lento para entrenar de verdad (nuestra estimacion: cerca de 12 minutos por epoca a 640 pixeles), asi que el entrenamiento real paso a una GPU en la nube.
2. **Auditoria antes de entrenar.** El analisis de la dimension de las cajas descarto los 640 pixeles y fijo el entrenamiento en 1024 y 1280. Tambien mostro que no hacia falta dividir las imagenes en mosaicos (todas caben en una entrada de 1280) y que no habia que retirar ninguna imagen.
3. **YOLO26s a 1024 en lugar del baseline sin cambios.** Como solo habia tiempo para una corrida de entrenamiento real, la usamos con YOLO26s y no con el baseline oficial YOLO26n. Resultado: 0.4654. Los buques militares salieron muy bien (AP 0.742) y los pesqueros fueron el cuello de botella (AP 0.127, recall 0.27).
4. **Calibracion por clase.** Cada clase recibe su propio umbral optimo de F1, integrado en el unico umbral oficial. Resultado: 0.4691, con el mAP intacto. Al revisar el codigo encontramos un error en nuestra propia formula de F1 antes de que afectara un resultado real, y fijamos la formula correcta con una prueba unitaria. Primer envio: **0.4912 publico**.
5. **Aumento en inferencia con volteos, y un callejon sin salida.** Cuatro volteos fusionados con WBF a IoU 0.55 dieron 0.4784, pero el AP de buques militares bajo de 0.742 a 0.693: la fusion unia buques atracados uno al lado del otro. Subir el IoU de fusion a 0.70 lo corrigio (buques militares 0.765, puntaje 0.4919); 0.80 no aporto mas (mAP 0.4427 frente a 0.4432). Con calibracion: 0.4956, y **0.5127 publico**.
6. **Un modelo mas grande, y una leccion sobre el ruido.** YOLO26m a 1280, con un calentamiento mas corto y menos epocas finales sin mosaico para adaptarse a una corrida corta, llego a 0.4737 tras 11 epocas (0.4789 calibrado). Encontro muchos mas barcos pesqueros (AP 0.237 frente a 0.127), pero fue mas debil en las imagenes de shiprsimagenet y vhrships. Su envio obtuvo **0.4868 publico**, por debajo del YOLO26s calibrado (0.4912), aunque localmente estaba cerca de 0.01 por encima. En ese momento construimos la simulacion del leaderboard; esa simulacion le daba a ese envio solo un 76% de probabilidad de superar a YOLO26s, asi que una mejora local de 0.01 no era un indicio confiable.
7. **Ensamble de dos modelos, retenido.** YOLO26m mas el YOLO26s de AWS, cada uno con volteos, obtuvo 0.5321. El envio separado de Santiago con su modelo de Kaggle ya habia llegado a **0.5263 publico**, y frente a una reproduccion local de su flujo el ensamble solo ganaba el 73% de los sorteos simulados, asi que no lo enviamos.
8. **Continuar el entrenamiento de YOLO26m.** Entrenamos de nuevo YOLO26m desde sus mejores pesos. El reinicio primero bajo el mAP50-95 de validacion a 0.324 (un nuevo calendario de tasa de aprendizaje y el mosaico activo otra vez), luego se recupero hasta 0.459, nuestro mejor modelo individual.
9. **Ensamble de tres modelos.** Sumamos el YOLO26s de Kaggle, entrenado durante 2.25 horas. En test, nuestra configuracion de fusion (cuatro volteos y confianza promediada) obtuvo 0.5395, mientras que la del flujo de Santiago (orientaciones con rotacion y confianza maxima) obtuvo 0.5315, porque tomar la confianza maxima entre tres modelos dejaba pasar mas falsas alarmas. Calibrado: 0.5410. Gano el 97% de los sorteos simulados y obtuvo **0.5476 publico**.
10. **Ocho orientaciones y pesos por modelo.** Guardamos una sola vez las predicciones crudas de los tres modelos en las ocho orientaciones, de modo que calificar cada configuracion de fusion tomaba cerca de 2 segundos en lugar de una corrida completa de prediccion. Las ocho orientaciones obtuvieron 0.5581 frente a 0.5395 de los cuatro volteos. Una busqueda sobre 35 configuraciones mostro que dar mas peso a los dos modelos mas fuertes (1.5 / 0.5 / 1.5) sumaba cerca de 0.003; elegimos una configuracion en el centro de la zona estable y no el mejor resultado aislado. Calibrado: 0.5635. Gano el 100% de los sorteos simulados y obtuvo **0.5658 publico**, nuestro mejor resultado.

Tambien evaluamos un detector publico de embarcaciones de una sola clase, entrenado con imagenes Sentinel-2 de 10 metros. No lo usamos porque detecta una sola clase y se entreno con una resolucion muy inferior a la de estos datos.

**Decisiones de mayor impacto**, medidas con el puntaje local de test:

| Decision | Antes | Despues | Mejora |
| --- | --- | --- | --- |
| Ensamble de YOLO26m con YOLO26s (ambos con volteos) | 0.4919 | 0.5276 | +0.036 |
| Volteos en inferencia con WBF a IoU 0.70 (YOLO26s) | 0.4654 | 0.4919 | +0.027 |
| Ocho orientaciones en lugar de cuatro volteos (tres modelos) | 0.5395 | 0.5581 | +0.019 |
| IoU de fusion de 0.70 en lugar de 0.55 (YOLO26s con volteos) | 0.4784 | 0.4919 | +0.014 |
| Tercer modelo y continuacion del entrenamiento de YOLO26m | 0.5276 | 0.5395 | +0.012 |
| Calibracion por clase (flujo final) | 0.5615 | 0.5635 | +0.002 |

**Experimentos y puntajes:**

| Experimento | Puntaje local | mAP50-95 | F1 | Puntaje publico |
| --- | --- | --- | --- | --- |
| YOLO26s (AWS), 1024 | 0.4654 | 0.4112 | 0.5921 | - |
| YOLO26s (AWS), calibrado | 0.4691 | 0.4112 | 0.6044 | 0.4912 |
| YOLO26s (AWS), volteos, WBF 0.55 | 0.4784 | 0.4248 | 0.6037 | - |
| YOLO26s (AWS), volteos, WBF 0.70 | 0.4919 | 0.4432 | 0.6057 | - |
| YOLO26s (AWS), volteos, calibrado | 0.4956 | 0.4432 | 0.6180 | 0.5127 |
| YOLO26m, 1280 | 0.4737 | 0.4136 | 0.6137 | - |
| YOLO26m, calibrado | 0.4789 | 0.4136 | 0.6313 | 0.4868 |
| YOLO26m + YOLO26s (AWS), volteos, calibrado | 0.5321 | 0.4810 | 0.6509 | no enviado |
| YOLO26s (Kaggle), flujo de Santiago | cerca de 0.528 (reproduccion local) | - | - | 0.5263 |
| Tres modelos, cuatro volteos, calibrado | 0.5410 | 0.4928 | 0.6536 | 0.5476 |
| Tres modelos, ocho orientaciones, con pesos, calibrado | **0.5635** | **0.5125** | **0.6825** | **0.5658** |

Los detalles completos, con resultados por clase y el motivo de cada corrida, estan en `experiments.md` y `submissions.md`.

## Resultados

Mejor envio: tres modelos, ocho orientaciones, WBF con pesos y calibracion por clase.

- Leaderboard publico: **0.5658**. Posicion final en la particion privada: **tercer puesto**.
- Test local: puntaje 0.5635, mAP@0.5:0.95 0.5125, mAP@0.5 0.6994, F1macro 0.6825 con umbral 0.50.

| Clase | AP50-95 | AP50 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| small_boat | 0.406 | 0.687 | 0.663 | 0.709 | 0.685 |
| cargo_ship | 0.558 | 0.704 | 0.668 | 0.650 | 0.659 |
| fishing_vessel | 0.314 | 0.489 | 0.502 | 0.574 | 0.535 |
| warship | 0.776 | 0.904 | 0.902 | 0.835 | 0.867 |
| work_vessel | 0.508 | 0.713 | 0.658 | 0.674 | 0.666 |

| Fuente | Puntaje | mAP50-95 | F1 |
| --- | --- | --- | --- |
| fair1m | 0.5248 | 0.4703 | 0.6521 |
| shiprsimagenet | 0.6079 | 0.5852 | 0.6609 |
| vhrships | 0.7651 | 0.7148 | 0.8824 |
| xview | 0.5158 | 0.4392 | 0.6945 |

**Frente a los intentos anteriores.** El puntaje publico paso de 0.4912 (primer envio) a 0.5658 (+0.075). Localmente, el flujo final esta 0.098 por encima de nuestro primer modelo entrenado (0.4654).

**Baseline.** No corrimos el baseline oficial sin cambios (YOLO26n, 30 epocas). La referencia mas cercana es el mismo script de baseline con YOLO26s a 1024, que obtuvo 0.4654 en local. El YOLO26s calibrado (0.4912 publico) es ademas la configuracion de un solo modelo y una sola pasada.

## Estructura del repositorio

```text
scripts/
  train.py               Entrenamiento con limite de tiempo, ajustado a corridas cortas
  cache_predict.py       Predice cada orientacion una vez por modelo y guarda las cajas crudas
  fuse_cached.py         Fusiona las predicciones guardadas con WBF y califica configuraciones
  tta_wbf.py             Prediccion con aumento en inferencia y WBF entre modelos en un paso
  calibrar_umbrales.py   Calibracion de umbrales por clase y ZIP de envio
  smoke_test_sahi.py     Prueba del entorno (PyTorch, Ultralytics, SAHI, rasterio, GPU)
tests/                   Pruebas unitarias de transformaciones de cajas y calibracion
experiments.md           Una fila por experimento, con su motivo
submissions.md           Una fila por envio al leaderboard
refs/README.md           Indice del codigo fuente de referencia usado durante el desarrollo
requirements.txt         Paquetes de Python con version fija
pyproject.toml           Configuracion de lint y pruebas
```

No estan en el repositorio: los datos y el kit de los organizadores (se esperan en `Entrega_Context/`), los pesos de los modelos (`weights/`, `runs/`) ni los envios.

## Instalacion

Requisitos: Python 3.11 y `uv`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

En Linux con GPU NVIDIA, instalar primero PyTorch desde el indice de CUDA 12.6, porque la version por defecto de torch 2.14.1 para Linux usa CUDA 13 y exige el driver 580 o superior:

```bash
uv pip install --python .venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements.txt
```

No instalar `opencv-python-headless` junto a `opencv-python`. En un servidor Ubuntu minimo, OpenCV tambien necesita el paquete del sistema `libgl1`.

Ubicar los archivos de los organizadores asi:

```text
Entrega_Context/
  kit_participantes/   evaluar.py, formato_envio.py, entrenar_baseline.py
  spacelab_dataset/    data.yaml, images/{train,test,oculto}, labels/{train,test}
```

Los pesos de COCO (`yolo26s.pt`, `yolo26m.pt`) se descargan solos al usarlos por primera vez. Los pesos del YOLO26s de Kaggle se esperan en `weights/team_s1024.pt`; no estan incluidos en este repositorio y se pueden solicitar al equipo.

Verificar la instalacion:

```bash
.venv/bin/python scripts/smoke_test_sahi.py
.venv/bin/pytest -q
```

## Uso

**Calificar cualquier carpeta o ZIP de predicciones sobre test:**

```bash
.venv/bin/python Entrega_Context/kit_participantes/evaluar.py --gt Entrega_Context/spacelab_dataset/labels/test --envio <carpeta-o-zip> --barrer-umbral
```

**Un solo modelo, una pasada por imagen** (predice test y oculto, califica test y arma el ZIP de envio), y luego la calibracion por clase:

```bash
.venv/bin/python Entrega_Context/kit_participantes/entrenar_baseline.py --datos Entrega_Context/spacelab_dataset --pesos runs/exp01/yolo26s_1024/weights/best.pt --imgsz 1024 --proyecto runs/exp01 --nombre yolo26s_1024
.venv/bin/python scripts/calibrar_umbrales.py --test-preds runs/exp01/yolo26s_1024/pred_test --oculto-preds runs/exp01/yolo26s_1024/envio --out runs/exp01/calib
```

**Reproducir el mejor envio:**

```bash
for split in test oculto; do
  src=Entrega_Context/spacelab_dataset/images/$split
  .venv/bin/python scripts/cache_predict.py --weights runs/exp05/yolo26m_1280_ft/weights/best.pt --imgsz 1280 --source $src --out runs/cache/m_$split.pkl
  .venv/bin/python scripts/cache_predict.py --weights runs/exp01/yolo26s_1024/weights/best.pt --imgsz 1024 --source $src --out runs/cache/s_$split.pkl
  .venv/bin/python scripts/cache_predict.py --weights weights/team_s1024.pt --imgsz 1024 --source $src --out runs/cache/k_$split.pkl
done
.venv/bin/python scripts/fuse_cached.py --test-caches runs/cache/m_test.pkl runs/cache/s_test.pkl runs/cache/k_test.pkl --oculto-caches runs/cache/m_oculto.pkl runs/cache/s_oculto.pkl runs/cache/k_oculto.pkl --configs d8:1.5,0.5,1.5:0.70:avg --write runs/fusion_best
.venv/bin/python scripts/calibrar_umbrales.py --test-preds runs/fusion_best/test --oculto-preds runs/fusion_best/oculto --out runs/fusion_best/calib
.venv/bin/python Entrega_Context/kit_participantes/formato_envio.py verificar --envio runs/fusion_best/calib/envio_calibrado.zip --imagenes Entrega_Context/spacelab_dataset/images/oculto
```

Una configuracion de fusion se escribe como `orientaciones:pesos:iou:confianza`. Los conjuntos de orientaciones son `none`, `h`, `flips4`, `team4` y `d8`; los pesos siguen el orden de los archivos guardados, y el modo de confianza es `avg`, `max` o `box_and_model_avg`. Se pueden pasar varias configuraciones a `--configs` para compararlas sobre test; `--write` usa la primera.

## Equipo y creditos

### Equipo DualCore

- Sergio Lopez: concibio y dirigio todo el flujo de trabajo.
- Santiago Rodriguez: entreno el modelo YOLO26s en Kaggle y construyo su flujo de inferencia, cuyas orientaciones con rotacion nos llevaron a agregarlas al ensamble final.
- Ambos integrantes trabajaron juntos en la mayor parte del proyecto.

### Creditos

- Organizadores: Facultad de Ingenieria de la Pontificia Universidad Javeriana y Fuerza Aeroespacial Colombiana (FAC), GeoVision Colombia.
- Datos: FAIR1M, ShipRSImageNet, VHRShips y xView, reunidos y reetiquetados por los organizadores.
- Software: Ultralytics YOLO26, PyTorch, ensemble-boxes (Weighted Boxes Fusion) y SAHI (solo en la prueba del entorno).
- Computo: AWS EC2 g5.xlarge y Kaggle.
