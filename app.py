import streamlit as st
from rag_engine import FactChecker
import re

# Configuración visual de la aplicación
st.set_page_config(
    page_title="Fact-Check AI - Grupo 6",
    page_icon="🔍",
    layout="wide"
)

# Estilos personalizados para el chat y la barra de progreso
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_fact_checker():
    return FactChecker()

def main():
    st.title("🔍 Fact-Check AI Assistant")
    st.caption("Verificación de hechos basada en evidencia de Wikipedia")

    # Configuración de idioma en la barra lateral
    with st.sidebar:
        st.header("Configuración")
        language_options = {
            "Auto (Detectar)": "auto", "Español": "es", "English": "en",
            "Français": "fr", "Italiano": "it", "Deutsch": "de"
        }
        target_lang = st.selectbox("Idioma de respuesta:", options=list(language_options.keys()))
        lang_code = language_options[target_lang]
        st.divider()
        st.info("Sistema RAG: Las respuestas se limitan estrictamente a la evidencia recuperada.")

    # Gestión del historial de conversación
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "extra" in message:
                with st.expander("📚 Ver fuentes originales"):
                    st.markdown(message["extra"])

    # Lógica principal del Chat
    if prompt := st.chat_input("Escribe una afirmación para verificar..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando fuentes y verificando..."):
                checker = get_fact_checker()
                verdict, confidence, sources = checker.verify_claim(prompt, lang_code)
                
                # Limpieza de prefijos repetitivos (Verdict, Veredicto, etc.)
                patterns = [
                    r'^(Verdict|Veredicto|Veredict|Verdict explanation|True Explanation|False Explanation|Verdadera explicación|Explicación verdadera|Explicación falsa):\s*',
                    r'^\*\*.*?\*\*:\s*' 
                ]
                
                cleaned_text = verdict.strip()
                for p in patterns:
                    cleaned_text = re.sub(p, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE).strip()

                # Separación de título y cuerpo
                lines = cleaned_text.splitlines()
                first_line = lines[0] if lines else ""
                rest_of_text = "\n".join(lines[1:]) if len(lines) > 1 else ""

                # Asignación dinámica de colores según el idioma
                v_lower = first_line.lower()
                if any(w in v_lower for w in ["true", "verdadero", "vrai", "vero", "wahr", "verdadera"]):
                    color = "green"
                elif any(w in v_lower for w in ["insufficient", "insuficiente", "insuffisante", "insufficiente", "unzureichend"]):
                    color = "orange"
                else:
                    color = "red" 

                # Renderizado de métricas de confianza
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Confianza", f"{confidence}%")
                with col2:
                    st.write("") 
                    st.progress(confidence / 100)

                # Renderizado del veredicto
                st.markdown(f"### :{color}[{first_line}]")
                if rest_of_text:
                    st.markdown(rest_of_text)

                # Formateo de fuentes bibliográficas
                source_text = ""
                if sources:
                    source_text = "#### Fuentes consultadas:\n"
                    for i, meta in enumerate(sources, 1):
                        source_text += f"{i}. **{meta.get('topic')}** - [Enlace]({meta.get('source')})\n"
                
                with st.expander("📚 Ver fuentes originales"):
                    st.markdown(source_text)

                # Guardado en sesión para persistencia
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"### :{color}[{first_line}]\n{rest_of_text}",
                    "extra": source_text
                })

if __name__ == "__main__":
    main()