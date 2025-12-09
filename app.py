import streamlit as st
from rag_engine import FactChecker

# Page Configuration
st.set_page_config(
    page_title="🔍 Fact-Checker Local",
    page_icon="✅",
    layout="centered"
)

# Initialize FactChecker (Cached)
@st.cache_resource
def get_fact_checker():
    return FactChecker()

def main():
    st.title("🔍 Sistema de Verificación de Hechos")
    st.markdown("---")
    
    st.write(
        "Este sistema utiliza **RAG (Retrieval-Augmented Generation)** con **Llama 3** "
        "para verificar afirmaciones basándose en artículos de Wikipedia."
    )
    
    # Input
    query = st.text_area("✍️ Ingresa una afirmación para verificar:", height=100)
    
    if st.button("Verificar Afirmación", type="primary"):
        if not query.strip():
            st.warning("Por favor ingresa un texto válido.")
            return
            
        with st.spinner("⏳ Consultando base de conocimientos y analizando..."):
            checker = get_fact_checker()
            verdict, sources = checker.verify_claim(query)
            
        # Display Verdict
        st.subheader("💡 Veredicto")
        st.markdown(verdict)
        
        # Sources Expander
        with st.expander("📚 Fuentes Recuperadas (Evidencia)"):
            if sources:
                for i, metadata in enumerate(sources, 1):
                    st.markdown(f"**Fuente {i}:**")
                    st.markdown(f"- **Tema:** {metadata.get('topic', 'N/A')}")
                    st.markdown(f"- **URL:** {metadata.get('source', '#')}")
                    st.divider()
            else:
                st.write("No se encontraron fuentes relevantes.")

if __name__ == "__main__":
    main()
