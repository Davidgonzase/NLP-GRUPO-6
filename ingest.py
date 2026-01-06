import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import wikipediaapi
import config
import tqdm

# Funcion para obtener contenido de Wikipedia
def get_wikipedia_content(topic, lang="en"):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent=config.WIKI_USER_AGENT, 
        language=lang
    )
    page = wiki_wiki.page(topic)
    
    if not page.exists():
        print(f"Page '{topic}' does not exist.")
        return None, None
    
    return page.text, page.fullurl

# Division del texto en chunks
def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

def main():
    print("Iniciando proceso de ingesta de datos...\n")
    
    # Inicializacion de ChormaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    
    # Creacion de la coleccion y borrado si existe
    collection_name = "fact_checking_knowledge_base"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    
    # Iniciado de modelo de embeddings
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    print(f"Modelo {config.EMBEDDING_MODEL_NAME} cargado\n")
    
    # Proceso de ingesta
    total_chunks = 0
    for topic in tqdm.tqdm(config.WIKI_TOPICS, desc="Procesando temas"):
        print(f"\nTema: {topic}")
        content, url = get_wikipedia_content(topic, lang=config.WIKI_LANG)
        if not content:
            continue
        chunks = chunk_text(content, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        if not chunks:
            continue
        
        print(f"{len(chunks)} chunks generados")
        
        # Generar embeddings y agregar a ChromaDB
        embeddings = embedding_model.encode(chunks)
        ids = [f"{topic}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": url, "topic": topic, "chunk_id": i} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        total_chunks += len(chunks)
        
    print(f"Ingesta completada. Total chunks indexados: {total_chunks}")

if __name__ == "__main__":
    main()
