# 🕵️‍♂️ Sistema de Verificación de Hechos (RAG Local)

Este proyecto implementa un sistema de **Verificación de Hechos (Fact-Checking)** que opera de manera local, utilizando **RAG (Retrieval-Augmented Generation)**. Combina una base de conocimientos creada a partir de Wikipedia con la inteligencia de **Llama 3** para verificar afirmaciones.

## 🚀 Características
- **100% Privado y Local**: Inferencia mediante **Llama 3** corriendo en tu máquina vía Ollama.
- **Transparente**: Cita las fuentes exactas (chunks de Wikipedia) utilizadas para el veredicto.
- **Eficiente**: Usa **ChromaDB** para búsquedas vectoriales rápidas.

## 🛠️ Requisitos Previos

1.  **Python 3.11** instalado.
2.  **Ollama** instalado y corriendo. Descárgalo en [ollama.com](https://ollama.com).

## ⚙️ Instalación

1.  **Clonar/Abrir el proyecto** en tu terminal.
2.  **Crear un entorno virtual**:
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
    Abre una terminal nueva (fuera de VS Code si es necesario) y ejecuta:
    ```bash
    ollama pull llama3
    ollama serve
    ```

## 🏃‍♂️ Ejecución

### 1. Ingesta de Datos (ETL)
Descarga artículos de Wikipedia y crea la base de datos vectorial local. (Requiere Internet solo para esta fase).
```bash
python ingest.py
```

### 2. Iniciar la App
Lanza la interfaz web local.
```bash
streamlit run app.py
```
Abre tu navegador en la dirección que aparece (ej. `http://localhost:8501`).

## 📁 Estructura
- `ingest.py`: Script para descargar y procesar datos de Wikipedia.
- `rag_engine.py`: Lógica del validador y conexión con el LLM.
- `app.py`: Interfaz de usuario (Streamlit).
- `config.py`: Configuraciones globales.
