# Sistema de Verificación de Hechos (RAG Local) Grupo 6

Este proyecto implementa un sistema de **Verificación de Hechos (Fact-Checking)** que opera de manera local, utilizando **RAG (Retrieval-Augmented Generation)**. Combina una base de conocimientos creada a partir de Wikipedia con la inteligencia de **Llama 3**/**Qwen 3** para verificar afirmaciones.

## Características
- Cita las fuentes exactas (chunks de Wikipedia) utilizadas para el veredicto.
- Usa **ChromaDB** para búsquedas vectoriales rápidas.
- Compatibilidad con diferentes idiomas
- Muestra la confianza del sistema con su veredicto
- GUI intuitiva y moderna

## Requisitos Previos

1.  Entorno **Python** en Conda o Venv 
2.  **Ollama** instalado y corriendo.
3.  Configuracion de un archivo .env que contenga las siguientes variables:
    ```bash
    OLLAMA_HOST=https://*Host-de-la-universidad*.uc3m.es/
    OLLAMA_API_KEY=*Key-proporcionada*
    WIKI_USER_AGENT=FactCheckerRAG/1.0 (contact: tucorreo@correo.com)
    ```

## Instalación

1.  **Clonar/Abrir el proyecto**.
2.  **Crear un entorno virtual (o conda)**:
    ```bash
    python -m venv venv
    # Activar en Windows:
    .\venv\Scripts\activate
    # Activar en Mac/Linux:
    source venv/bin/activate
    ```
3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Preparar Ollama** (Llama 3):
    ```bash
    ollama pull llama3
    ollama serve
    ```

## Ejecución

### 1. Ingesta de Datos (ETL)
Descarga automaticamente artículos de Wikipedia y crea la base de datos vectorial local.
```bash
python ingest.py
```

### 2. Iniciar la App
Lanza la interfaz web local.
```bash
streamlit run app.py
```
Abre el navegador en la dirección que aparece (ej. `http://localhost:8501`).

## Estructura
- `ingest.py`: Script para descargar y procesar datos de Wikipedia.
- `rag_engine.py`: Lógica del validador y conexión con el LLM.
- `app.py`: Interfaz de usuario (Streamlit).
- `config.py`: Configuraciones globales.
