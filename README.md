# procesamiento-datos-secuenciales-donut-transformer
Implementación de un Transformer encoder–decoder (Donut) para comprensión de documentos sin OCR, con visualización interactiva de atención, embeddings, QKV y proceso de inferencia mediante Streamlit.

## 1. Resumen 
Este proyecto presenta el desarrollo e implementación de un sistema de Visual Document Understanding basado en el modelo Transformer Donut (Document Understanding Transformer), el cual permite realizar tareas de Document Visual Question Answering (DocVQA) directamente sobre imágenes de documentos.
A diferencia de los enfoques tradicionales que dependen de sistemas de reconocimiento óptico de caracteres (OCR), Donut propone una arquitectura end-to-end que aprende una representación conjunta de la información visual y textual a partir de la imagen, eliminando la necesidad de módulos intermedios. Este enfoque reduce la complejidad del sistema, evita la propagación de errores del OCR y mejora la eficiencia computacional.
En este trabajo se implementa una aplicación interactiva utilizando Streamlit que integra el modelo preentrenado naver-clova-ix/donut-base-finetuned-docvqa, permitiendo al usuario cargar imágenes de documentos, formular preguntas y obtener respuestas generadas automáticamente. Adicionalmente, se incluye un módulo de análisis visual que permite explorar internamente el funcionamiento del modelo, incluyendo embeddings, división en patches, mecanismos de atención y representación de los vectores Q, K y V.
Los resultados evidencian que el modelo es capaz de comprender la estructura semántica de los documentos y generar respuestas coherentes, manteniendo tiempos de inferencia adecuados incluso en entornos sin GPU. Asimismo, la visualización de los componentes internos del Transformer facilita la interpretación del modelo, aportando valor tanto educativo como investigativo.

## 2. Introducción
El análisis automático de documentos es una tarea fundamental en múltiples aplicaciones del mundo real, como la digitalización de facturas, procesamiento de recibos, extracción de información y sistemas de atención automatizada. Tradicionalmente, estos sistemas se han basado en pipelines compuestos por múltiples etapas, donde el reconocimiento óptico de caracteres (OCR) juega un papel central al convertir el contenido visual en texto procesable.
Sin embargo, los enfoques basados en OCR presentan varias limitaciones importantes. En primer lugar, implican un alto costo computacional debido a la necesidad de ejecutar modelos adicionales. En segundo lugar, son sensibles a variaciones en el idioma, formato o calidad del documento. Finalmente, los errores generados en la etapa de OCR se propagan a las fases posteriores, afectando negativamente el rendimiento global del sistema.
En respuesta a estas limitaciones, han surgido modelos basados en arquitecturas Transformer que abordan el problema de manera integral. Entre ellos, Donut (Document Understanding Transformer) propone un enfoque end-to-end que elimina completamente la dependencia del OCR, aprendiendo directamente a mapear imágenes de documentos a representaciones estructuradas o respuestas textuales.
Este proyecto se enfoca en la implementación práctica de dicho enfoque mediante el desarrollo de una aplicación interactiva que permite explorar tanto el rendimiento del modelo como su funcionamiento interno. A través de una interfaz construida con Streamlit, el usuario puede realizar consultas sobre documentos y visualizar diferentes componentes del modelo, como embeddings, patches y mecanismos de atención.

## 3. Marco teórico
### 3.1 Arquitectura Transformer Encoder–Decoder

---

El modelo utilizado en este proyecto se basa en una arquitectura Transformer de tipo encoder–decoder, la cual ha demostrado un alto desempeño en tareas de procesamiento de datos secuenciales. En este caso, se emplea el modelo Donut (Document Understanding Transformer), diseñado específicamente para la comprensión de documentos visuales.

La arquitectura está compuesta por dos bloques principales:

Encoder (Swin Transformer): encargado de procesar la imagen de entrada y convertirla en una representación latente.
Decoder (basado en Transformer tipo BART): encargado de generar la secuencia de salida en forma de texto estructurado o respuesta a una pregunta.

El flujo general del modelo es el siguiente:

La imagen de entrada es redimensionada y dividida en pequeños bloques llamados patches.
Cada patch es transformado en un vector numérico (embedding).
Estos embeddings son procesados por el encoder, generando representaciones contextualizadas.
El decoder recibe estas representaciones junto con un prompt de entrada y genera la secuencia de salida token por token.

Este enfoque permite tratar la imagen como una secuencia, de manera similar a como se procesan las palabras en tareas de lenguaje natural.

### 3.2 Representación mediante embeddings

---

Una vez dividida la imagen en patches (por ejemplo, una imagen de 384×384 en bloques de 16×16 genera 576 tokens), cada patch es proyectado a un espacio vectorial de alta dimensión (en este caso, 1024 dimensiones).

Estos embeddings contienen información relevante sobre:

Bordes y formas
Patrones visuales
Estructura del documento
Posición espacial

Posteriormente, el encoder transforma estos vectores iniciales en representaciones más complejas que incorporan contexto global, permitiendo que cada token tenga información de toda la imagen.

### 3.3 Mecanismo de atención

---

El componente central del Transformer es el mecanismo de self-attention, el cual permite que cada token interactúe con todos los demás tokens de la secuencia.

Este mecanismo se basa en tres matrices fundamentales:

- Query (Q)
- Key (K)
- Value (V)

Estas matrices se obtienen a partir de los embeddings mediante transformaciones lineales:
Donde:

- X representa los embeddings de entrada  
- Wq, Wk, Wv son matrices de pesos aprendidas  

El cálculo de la atención se realiza de la siguiente forma:
Donde:

- Kᵀ es la transpuesta de K  
- d es la dimensión de los embeddings  
- softmax normaliza los valores de atención  

Este mecanismo permite:

- Identificar relaciones entre diferentes regiones de la imagen  
- Enfocar la atención en zonas relevantes  
- Generar representaciones contextuales dinámicas  

En la implementación desarrollada, este comportamiento se visualiza mediante mapas de atención que muestran en qué partes de la imagen se enfoca el modelo durante la inferencia.

### 3.4 Generación de secuencias (Decoder)

---

El decoder funciona de manera autoregresiva, generando la salida token por token. En cada paso:

Recibe los tokens previamente generados
Atiende a la salida del encoder
Predice el siguiente token más probable

En este proyecto, se utiliza un prompt estructurado como:

<s_docvqa><s_question>Pregunta</s_question>

Esto guía al modelo para realizar tareas de Document Visual Question Answering.

El resultado final es una secuencia de texto que puede representar:

Respuestas directas
Información estructurada (tipo JSON)
Campos específicos del documento


### 3.5 Innovaciones del modelo Donut

---

El modelo Donut introduce varias innovaciones importantes frente a enfoques tradicionales:

Eliminación del OCR: no requiere extracción explícita de texto, lo que reduce errores acumulativos.
Arquitectura end-to-end: aprende directamente desde la imagen hasta la salida textual.
Uso de Vision Transformer (Swin): mejora la captura de patrones visuales locales y globales.
Flexibilidad en tareas: puede adaptarse a diferentes problemas como clasificación, extracción de información y VQA.
Representación unificada: combina visión y lenguaje en un solo modelo. Estas características lo convierten en un modelo eficiente y robusto para tareas de comprensión de documentos.

## 4. Metodología

El desarrollo de este proyecto se basó en la implementación de un modelo Transformer encoder–decoder preentrenado para la tarea de Document Visual Question Answering (DocVQA), utilizando el modelo Donut.

### 4.1 Enfoque general

---

El enfoque adoptado consiste en utilizar un modelo preentrenado, evitando el entrenamiento desde cero, con el objetivo de enfocarse en:

- Comprender la arquitectura del modelo  
- Implementar el proceso de inferencia  
- Analizar el comportamiento interno del Transformer  
- Visualizar los componentes clave del modelo  

Se desarrolló una aplicación interactiva que permite ejecutar el modelo sobre imágenes de documentos y explorar sus representaciones internas.

### 4.2 Herramientas utilizadas

---

Para la implementación del sistema se utilizaron las siguientes tecnologías:

- Python: lenguaje principal de desarrollo  
- Streamlit: construcción de la interfaz interactiva  
- PyTorch (torch): ejecución del modelo y manejo de tensores  
- Transformers (Hugging Face): carga del modelo y procesamiento  
- PIL (Pillow): manipulación de imágenes  
- NumPy y Pandas: procesamiento de datos  
- Matplotlib, Seaborn y Plotly: visualización de resultados  
- RapidFuzz: cálculo de métricas de evaluación  

Estas herramientas permiten integrar tanto la inferencia del modelo como la visualización de sus componentes internos.

### 4.3 Uso de pesos preentrenados

---

En este proyecto se utilizó el modelo preentrenado: "naver-clova-ix/donut-base-finetuned-docvqa"
Este modelo ya ha sido entrenado para tareas de comprensión de documentos, específicamente para responder preguntas sobre imágenes.

La carga del modelo se realiza mediante la librería Transformers:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa"
)
```
DonutProcessor: se encarga del preprocesamiento de la imagen y tokenización del texto
VisionEncoderDecoderModel: contiene la arquitectura completa encoder–decoder
El modelo se configura en modo evaluación para evitar el cálculo de gradientes: model.eval()

### 4.4 Proceso de inferencia

---

El proceso de inferencia implementado en el proyecto sigue los siguientes pasos:
  4.4.1. Carga de la imagen
      a. El usuario sube una imagen mediante la interfaz
      b. La imagen se convierte a formato RGB
  4.4.2. Preprocesamiento
      a. La imagen es redimensionada a 384×384 píxeles
      b. Se convierte en tensores utilizando el processor
      
```python
pixel_values = processor(image_resized, return_tensors="pt").pixel_values
 ```
        
  4.4.3. Construcción del prompt
      a. Se define un prompt estructurado para la tarea DocVQA
      
```python
task_prompt = "<s_docvqa><s_question>Pregunta</s_question>"
```
      
  4.4.4. Tokenización del prompt
  
```python
decoder_input_ids = processor.tokenizer(
task_prompt,
return_tensors="pt"
).input_ids
```
      
  4.4.5 Generación de la respuesta
  
```python
outputs = model.generate(
pixel_values,
decoder_input_ids=decoder_input_ids,
max_length=512
)
```
      
  4.4.6 Decodificación de la salida
  
```python
decoded = processor.batch_decode(outputs)[0]
```
      
  4.4.7 Postprocesamiento
      a. Se eliminan tokens especiales
      b. Se obtiene el texto final

### 4.5 Visualización del modelo

---

Adicionalmente, se implementaron módulos interactivos para analizar el comportamiento interno del modelo:
  - Visualización de embeddings
  - División de la imagen en patches
  - Mapas de atención
  - Representación de Q, K y V
  - Métricas de desempeño
Esto permite no solo ejecutar el modelo, sino también entender su funcionamiento interno de manera visual.

### 4.6 Consideraciones de implementación

---

  - El modelo se ejecuta en CPU, lo que garantiza compatibilidad en diferentes equipos
  - Se utiliza caché para evitar recargar el modelo múltiples veces
  - Se mantiene un historial de consultas para análisis posterior
  - Se optimiza el flujo para ejecución en tiempo real durante la sustentación


## 5. Desarrollo e implementación

En esta sección se describe detalladamente cómo ejecutar el proyecto, cómo se cargan los pesos del modelo y cómo se realiza el proceso completo de inferencia, desde la entrada de datos hasta la generación del resultado.

---

### 5.1 Estructura general del sistema

---

El sistema fue desarrollado como una aplicación interactiva utilizando Streamlit, permitiendo al usuario:

- Cargar una imagen de documento
- Ingresar una pregunta
- Ejecutar el modelo
- Visualizar resultados y análisis internos

El flujo general del sistema es:
Usuario → Interfaz (Streamlit) → Preprocesamiento → Modelo Transformer → Decodificación → Visualización

### 5.2 Ejecución del proyecto

---

Para ejecutar el proyecto se deben seguir los siguientes pasos:

1. Clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```Bash
streamlit run app.py
```

4. Abrir en el navegador:
```Bash
http://localhost:8501
```

5.3 Carga del modelo y pesos preentrenados

---

El modelo utilizado es:
```Bash
naver-clova-ix/donut-base-finetuned-docvqa
```

Este modelo se descarga automáticamente desde Hugging Face al ejecutarse por primera vez.

Implementación:
```Python
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    model.eval()
    return processor, model
```

Explicación:
- DonutProcessor:
        - Convierte imágenes en tensores
        - Tokeniza texto de entrada y salida
  
- VisionEncoderDecoderModel:
        - Contiene el encoder (Swin Transformer)
        - Contiene el decoder (BART)
  
- model.eval():
        - Desactiva entrenamiento
        - Optimiza inferencia
  
- @st.cache_resource:
        - Evita recargar el modelo en cada ejecución
        - Mejora el rendimiento significativamente
  
5.4 Configuración del dispositivo

---

El modelo se ejecuta en CPU:
```Python
device = "cpu"
model.to(device)
```
Esto garantiza que el sistema funcione en cualquier computador sin necesidad de GPU.

5.5 Entrada de datos

---

El usuario proporciona dos entradas:
    1. Imagen del documento
    2. Pregunta en lenguaje natural
    
Implementación:
```Python
uploaded_file = st.file_uploader("Subir imagen", type=["jpg","png","jpeg"])
question = st.text_input("Pregunta")
```
Procesamiento de imagen:
```Python
image = Image.open(uploaded_file).convert("RGB")
```
5.6 Preprocesamiento

---

El modelo requiere imágenes de tamaño fijo:
```Python
image_resized = image.resize((384,384))
```
Luego se convierte a tensor:
```Python
pixel_values = processor(
    image_resized,
    return_tensors="pt"
).pixel_values.to(device)
```
Explicación:
    - Se transforma la imagen en un tensor
    - Se normaliza automáticamente
    - Se prepara como entrada del encoder
    
5.7 Construcción del prompt

---

El modelo utiliza un formato estructurado:
```Python
task_prompt = f"<s_docvqa><s_question>{question}</s_question>"
```
Explicación:
    - <s_docvqa> indica la tarea
    - <s_question> contiene la pregunta
    - Este formato guía al decoder
    
5.8 Tokenización del prompt

---

```Python
decoder_input_ids = processor.tokenizer(
    task_prompt,
    return_tensors="pt"
).input_ids.to(device)
```
Esto convierte el texto en tokens numéricos para el decoder.

5.9 Proceso de inferencia

---

El modelo genera la respuesta:
```Python
with torch.no_grad():
    gen = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512
    )
```
Explicación:
    - pixel_values: entrada visual
    - decoder_input_ids: entrada textual
    - generate(): produce la secuencia de salida
    - torch.no_grad(): optimiza memoria y velocidad
    
5.10 Decodificación de la salida

---

```Python
decoded = processor.batch_decode(gen, skip_special_tokens=False)[0]
```
Luego se limpia el texto:

```Python
import re
text = re.sub(r"<.*?>", "", decoded).strip()
```

Explicación:
    - Se eliminan tokens especiales
    - Se obtiene texto legible
    
5.11 Almacenamiento de resultados

---

Los resultados se guardan para análisis posterior:
```Python
st.session_state.result = {
    "text": text,
    "pixel_values": pixel_values.cpu(),
    "tokens": len(gen[0]),
    "chars": len(text),
    "latency": latency
}
```

También se guarda historial:
```Python
st.session_state.history.append({
    "question": question,
    "answer": text,
    "latency": latency,
    "tokens": len(gen[0]),
    "chars": len(text)
})
```

5.12 Visualización del modelo

---

Se implementaron múltiples módulos interactivos:
Embeddings

    - Visualización de vectores de 1024 dimensiones
Patches

    - División en 576 tokens visuales (24x24)
Attention

    - Mapas de atención entre tokens
    - Visualización sobre la imagen
Q, K, V

    - Simulación del mecanismo interno del Transformer
Resultados

    - Métricas:
        - Exact Match
        - F1 Score
        - ANLS
        
5.13 Flujo completo del sistema

---

Imagen → Resize → Tensor → Encoder (Swin)
        ↓
Embeddings → Attention → Representación contextual
        ↓
Decoder (BART) + Prompt
        ↓
Generación de tokens
        ↓
Texto final

5.14 Consideraciones importantes

---

    - No se realiza entrenamiento desde cero
    - Se utilizan pesos preentrenados
    - El sistema es completamente end-to-end (sin OCR)
    - La inferencia es en tiempo real
    - Se incluye visualización interna del modelo
    
5.15 Conclusión de la implementación

---

La implementación permite no solo ejecutar el modelo Donut, sino también comprender en profundidad su funcionamiento interno, cumpliendo con los objetivos del proyecto al integrar inferencia, visualización y análisis del modelo Transformer encoder–decoder.

## 6. Resultados y análisis

En esta sección se presentan los resultados obtenidos al ejecutar el modelo Donut sobre diferentes imágenes de documentos, junto con un análisis detallado de su desempeño, visualizaciones internas y métricas de evaluación.

---

### 6.1 Resultados de inferencia

---

El modelo fue evaluado utilizando imágenes de documentos junto con preguntas en lenguaje natural. El sistema genera respuestas automáticamente mediante inferencia.

Ejemplo de resultado:

- Pregunta: What is the total?
- Predicción del modelo: 154.32

El tiempo de respuesta promedio observado fue bajo, permitiendo inferencia en tiempo real en CPU.

### 6.2 Visualización de la entrada y salida

---

La aplicación permite visualizar:

- Imagen original cargada por el usuario
- Pregunta realizada
- Respuesta generada por el modelo

Esto facilita la validación cualitativa del desempeño del modelo.

### 6.3 Análisis del mecanismo de atención

---

Se implementó una visualización del mapa de atención del encoder.

El modelo genera una matriz de atención que representa cómo cada token visual interactúa con los demás.

#### Interpretación:

- Zonas con mayor intensidad indican mayor relevancia
- El token CLS resume la información global de la imagen
- Se puede observar qué regiones de la imagen influyen en la respuesta

Esto permite entender el comportamiento interno del modelo, evidenciando el funcionamiento del mecanismo de self-attention.

### 6.4 Análisis de embeddings

---

Cada imagen es transformada en una secuencia de embeddings de dimensión 1024.

Se visualizó:

- Distribución de valores del embedding
- Representación en forma de mapa de calor
- Activación por token visual

#### Observaciones:

- Los embeddings capturan información estructural del documento
- Regiones con texto presentan mayor activación
- Bordes y patrones visuales son representados en el espacio latente


### 6.5 Análisis de tokens visuales (patches)

---

El modelo divide la imagen en:

- 576 tokens (24 x 24)
- Cada token corresponde a un patch de 16 x 16 píxeles

Se analizó:

- Posición de cada token
- Importancia relativa (norma del embedding)
- Relación entre ubicación y relevancia

#### Observaciones:

- Los tokens con mayor importancia suelen corresponder a regiones con información clave (texto relevante)
- Las zonas vacías o fondo tienen menor impacto en la inferencia

### 6.6 Análisis del mecanismo Q, K y V

---

Se implementó una simulación del cálculo de Query, Key y Value a partir de los embeddings.

Esto permitió analizar:

- Magnitud de Q, K y V
- Distribución en el espacio latente
- Influencia de cada token en el cálculo de atención

#### Observaciones:

- Tokens con mayor norma en Q tienen mayor influencia
- La atención depende de la similitud entre Q y K
- El mecanismo permite modelar relaciones complejas entre regiones de la imagen

### 6.7 Métricas de evaluación

---

Se implementaron tres métricas principales:

#### Exact Match (EM)
Mide si la predicción coincide exactamente con la respuesta real.
```Bash
EM = 1 si predicción == ground truth, 0 en otro caso
```

#### F1 Score
Evalúa la similitud a nivel de palabras:
```Bash
F1 = 2 * (precisión * recall) / (precisión + recall)
```

#### ANLS (Approximate Normalized Levenshtein Similarity)
Mide similitud considerando errores parciales:
```Bash
ANLS = 1 - (distancia_levenshtein / longitud_máxima)
```

### 6.8 Resultados cuantitativos

---

A partir de las pruebas realizadas:

- El modelo logra buenas predicciones en documentos claros
- El desempeño disminuye en imágenes con:
  - Baja resolución
  - Texto pequeño
  - Ruido visual

Ejemplo de análisis:

- Exact Match: alto en respuestas cortas
- F1 Score: permite evaluar coincidencias parciales
- ANLS: útil para medir errores leves


### 6.9 Análisis de rendimiento

---

Se evaluaron:

- Tiempo de inferencia
- Número de tokens generados
- Longitud de la respuesta

#### Observaciones:

- Existe relación entre:
  - Mayor número de tokens → mayor tiempo de inferencia
- El modelo mantiene tiempos bajos incluso en CPU
- La generación es eficiente para aplicaciones en tiempo real


### 6.10 Visualizaciones adicionales

---

Se implementaron gráficas para analizar el comportamiento del modelo:

- Tokens vs tiempo de inferencia
- Longitud de respuesta vs tokens
- Tiempo por consulta

Estas visualizaciones permiten identificar patrones en el rendimiento del modelo.


### 6.11 Limitaciones observadas en los resultados

---

Durante las pruebas se identificaron las siguientes limitaciones:

- Sensibilidad a la calidad de la imagen
- Dificultad con texto pequeño o borroso
- Dependencia del formato del documento
- Posibles errores en respuestas largas

### 6.12 Conclusión del análisis

---

El modelo Donut demuestra un desempeño sólido en tareas de Document VQA, siendo capaz de generar respuestas coherentes sin necesidad de OCR.

Además, las visualizaciones implementadas permiten comprender en profundidad el funcionamiento interno del modelo, especialmente:

- El mecanismo de atención
- La representación mediante embeddings
- La interacción entre tokens visuales

## 7. Conclusiones

En este proyecto se implementó y analizó una arquitectura Transformer encoder–decoder aplicada a la tarea de Document Visual Question Answering (DocVQA), utilizando el modelo Donut con pesos preentrenados.

### 7.1 Aprendizajes

---

A lo largo del desarrollo del proyecto se obtuvieron los siguientes aprendizajes:

- Se comprendió en profundidad el funcionamiento de una arquitectura Transformer encoder–decoder aplicada a datos visuales y secuenciales.
- Se analizó cómo una imagen puede ser transformada en una secuencia de tokens visuales mediante la división en patches.
- Se entendió la importancia de los embeddings como representación numérica de la información visual.
- Se estudió el mecanismo de atención (self-attention), identificando cómo el modelo aprende relaciones entre diferentes regiones de la imagen.
- Se comprendió la generación de los tensores Q (queries), K (keys) y V (values), y su papel en el cálculo de la atención.
- Se analizó el proceso completo de inferencia, desde la entrada (imagen + prompt) hasta la salida (texto generado).
- Se evidenció cómo los modelos modernos pueden resolver tareas complejas sin necesidad de OCR explícito, utilizando enfoques end-to-end.

Además, se logró implementar una interfaz interactiva que permite visualizar el comportamiento interno del modelo, lo cual facilita su interpretación.


### 7.2 Limitaciones

---

A pesar de los buenos resultados obtenidos, el modelo presenta algunas limitaciones:

- Sensibilidad a la calidad de la imagen: imágenes con baja resolución o ruido afectan el desempeño.
- Dificultad para reconocer texto pequeño o con tipografías complejas.
- Dependencia del dominio de entrenamiento: el modelo funciona mejor en documentos similares a los usados durante su entrenamiento.
- Posibles errores en respuestas largas o complejas.
- Limitaciones en la interpretabilidad completa del modelo, ya que algunas operaciones internas no son directamente accesibles (como las matrices reales de Q, K y V).
- Alto consumo de memoria en comparación con modelos tradicionales.


### 7.3 Posibles mejoras

---

Se identifican varias oportunidades de mejora para el sistema:

- Implementar ejecución en GPU para mejorar el rendimiento y reducir el tiempo de inferencia.
- Integrar modelos más recientes o versiones más grandes de Donut para mejorar la precisión.
- Incorporar técnicas de fine-tuning con datasets específicos para mejorar el desempeño en dominios particulares.
- Mejorar la interfaz gráfica para permitir análisis más avanzados del modelo.
- Implementar visualizaciones más precisas del mecanismo de atención, incluyendo atención por cabeza (multi-head attention).
- Optimizar el preprocesamiento de imágenes para mejorar la calidad de entrada.
- Integrar evaluación automática con datasets estandarizados para obtener métricas más robustas.

### 7.4 Conclusión general

---

El modelo Donut representa un avance significativo en el procesamiento de documentos, al eliminar la necesidad de sistemas OCR tradicionales y permitir un enfoque completamente end-to-end basado en Transformers.

La implementación desarrollada no solo permite ejecutar el modelo, sino también analizar su funcionamiento interno, cumpliendo con los objetivos del proyecto al integrar:

- Inferencia funcional
- Visualización interactiva
- Interpretación del modelo

## 8. Referencias

    [1] J. Kim, J. Lee, J. Park, and J. Kim, "OCR-free Document Understanding Transformer," arXiv preprint arXiv:2111.15664, 2021. [Online]. Available: https://arxiv.org/abs/2111.15664

    [2] NAVER Clova AI, "Donut: OCR-free Document Understanding Transformer," Hugging Face, 2023. [Online]. Available: https://huggingface.co/naver-clova-ix/donut-base

    [3] NAVER Clova AI, "Donut base finetuned on DocVQA," Hugging Face, 2023. [Online]. Available: https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa

    [4] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

    [5] PyTorch Team, "PyTorch: An imperative style, high-performance deep learning library," 2023. [Online]. Available: https://pytorch.org

    [6] Hugging Face, "Transformers Library Documentation," 2023. [Online]. Available: https://huggingface.co/docs/transformers

    [7] Streamlit Inc., "Streamlit Documentation," 2023. [Online]. Available: https://docs.streamlit.io

    [8] Plotly Technologies Inc., "Plotly Python Graphing Library," 2023. [Online]. Available: https://plotly.com/python/

    [9] RapidFuzz Library, "RapidFuzz Documentation," 2023. [Online]. Available: https://maxbachmann.github.io/RapidFuzz/

    [10] NumPy Developers, "NumPy: Fundamental package for scientific computing with Python," 2023. [Online]. Available: https://numpy.org

    [11] Pandas Development Team, "Pandas Documentation," 2023. [Online]. Available: https://pandas.pydata.org

[12] Matplotlib Development Team, "Matplotlib: Visualization with Python," 2023. [Online]. Available: https://matplotlib.org

