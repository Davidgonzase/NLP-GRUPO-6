import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import wikipediaapi
import config
import tqdm

def get_wikipedia_content(topic, lang="en"):
    """
    Fetches the content of a Wikipedia page.
    """
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='RAGFactChecker/1.0 (contact: jrodriguez4013@gmail.com)',
        language=lang
    )
    page = wiki_wiki.page(topic)
    
    if not page.exists():
        print(f"Page '{topic}' does not exist.")
        return None, None
    
    return page.text, page.fullurl

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks with overlap.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

def main():
    print("🚀 Iniciando proceso de ingesta...")
    
    # Initialize ChromaDB
    print(f"📂 Conectando a ChromaDB en: {config.CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    
    # Create or get collection
    collection_name = "fact_checking_knowledge_base"
    # Delete if exists to start fresh (optional, but good for idempotent runs in this demo)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)
    
    # Initialize Embedding Model
    print(f"🧠 Cargando modelo de embeddings: {config.EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    
    # Process Topics
    total_chunks = 0
    
    for topic in tqdm.tqdm(config.WIKI_TOPICS, desc="Procesando temas"):
        print(f"\n📥 Descargando: {topic}")
        content, url = get_wikipedia_content(topic, lang=config.WIKI_LANG)
        
        if not content:
            continue
            
        chunks = chunk_text(content, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        
        if not chunks:
            continue
            
        print(f"   ✂️ Generando {len(chunks)} chunks...")
        
        # Generate Embeddings
        embeddings = embedding_model.encode(chunks)
        
        # Prepare data for Chroma
        ids = [f"{topic}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": url, "topic": topic, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to collection
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        total_chunks += len(chunks)
        
    print(f"\n✅ Ingesta completada. Total chunks indexados: {total_chunks}")

if __name__ == "__main__":
    main()
