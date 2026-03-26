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

Una vez dividida la imagen en patches (por ejemplo, una imagen de 384×384 en bloques de 16×16 genera 576 tokens), cada patch es proyectado a un espacio vectorial de alta dimensión (en este caso, 1024 dimensiones).

Estos embeddings contienen información relevante sobre:

Bordes y formas
Patrones visuales
Estructura del documento
Posición espacial

Posteriormente, el encoder transforma estos vectores iniciales en representaciones más complejas que incorporan contexto global, permitiendo que cada token tenga información de toda la imagen.

### 3.3 Mecanismo de atención

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

El modelo Donut introduce varias innovaciones importantes frente a enfoques tradicionales:

Eliminación del OCR: no requiere extracción explícita de texto, lo que reduce errores acumulativos.
Arquitectura end-to-end: aprende directamente desde la imagen hasta la salida textual.
Uso de Vision Transformer (Swin): mejora la captura de patrones visuales locales y globales.
Flexibilidad en tareas: puede adaptarse a diferentes problemas como clasificación, extracción de información y VQA.
Representación unificada: combina visión y lenguaje en un solo modelo. Estas características lo convierten en un modelo eficiente y robusto para tareas de comprensión de documentos.

## 4. Metodología

El desarrollo de este proyecto se basó en la implementación de un modelo Transformer encoder–decoder preentrenado para la tarea de Document Visual Question Answering (DocVQA), utilizando el modelo Donut.

### 4.1 Enfoque general

El enfoque adoptado consiste en utilizar un modelo preentrenado, evitando el entrenamiento desde cero, con el objetivo de enfocarse en:

- Comprender la arquitectura del modelo  
- Implementar el proceso de inferencia  
- Analizar el comportamiento interno del Transformer  
- Visualizar los componentes clave del modelo  

Se desarrolló una aplicación interactiva que permite ejecutar el modelo sobre imágenes de documentos y explorar sus representaciones internas.

### 4.2 Herramientas utilizadas

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
DonutProcessor: se encarga del preprocesamiento de la imagen y tokenización del texto
VisionEncoderDecoderModel: contiene la arquitectura completa encoder–decoder
El modelo se configura en modo evaluación para evitar el cálculo de gradientes: model.eval()
```
### 4.4 Proceso de inferencia
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
Adicionalmente, se implementaron módulos interactivos para analizar el comportamiento interno del modelo:
  - Visualización de embeddings
  - División de la imagen en patches
  - Mapas de atención
  - Representación de Q, K y V
  - Métricas de desempeño
Esto permite no solo ejecutar el modelo, sino también entender su funcionamiento interno de manera visual.

### 4.6 Consideraciones de implementación
  - El modelo se ejecuta en CPU, lo que garantiza compatibilidad en diferentes equipos
  - Se utiliza caché para evitar recargar el modelo múltiples veces
  - Se mantiene un historial de consultas para análisis posterior
  - Se optimiza el flujo para ejecución en tiempo real durante la sustentación
  
      

