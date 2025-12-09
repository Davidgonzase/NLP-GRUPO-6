import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import config

class FactChecker:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="fact_checking_knowledge_base")
        
        # Initialize Embedding Model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
    def retrieve_context(self, query, n_results=3):
        """
        Retrieves relevant documents from ChromaDB.
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        return documents, metadatas

    def verify_claim(self, claim):
        """
        Verifies a claim using RAG with Llama 3.
        """
        # 1. Retrieve Context
        documents, metadatas = self.retrieve_context(claim)
        
        if not documents:
            return "INFORMACIÓN INSUFICIENTE", []

        context_str = "\n\n".join(documents)
        
        # 2. Construct Prompt
        prompt = f"""
You are an expert fact-checker. Your task is to verify a claim based ONLY on the provided context.

CONTEXT:
{context_str}

CLAIM TO VERIFY:
"{claim}"

INSTRUCTIONS:
1. Analyze the claim and the context.
2. If the context supports the claim, verdict is "SUPPORTED".
3. If the context contradicts the claim, verdict is "REFUTED".
4. If there is no relevant info, verdict is "INSUFFICIENT INFO".

OUTPUT FORMAT (STRICT):
- Verdict: [SUPPORTED | REFUTED | INSUFFICIENT INFO] (Translated to the language of the CLAIM)
- Confidence Score: [0-100]%
- Explanation: A brief reasoning (1 sentence) explaining why. (Translated to the language of the CLAIM)
- Quote: The original text from the context used as evidence.

CRITICAL RULE:
You MUST answer in the SAME LANGUAGE as the "CLAIM TO VERIFY".
If the claim is in French, the entire response (Verification, Explanation) MUST be in French.
If the claim is in Spanish, use Spanish.
"""

        # 3. Query LLM (Ollama)
        try:
            response = ollama.chat(model=config.LLM_MODEL_NAME, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            
            result_text = response['message']['content']
            return result_text, metadatas
            
        except Exception as e:
            return f"Error al conectar con Ollama: {str(e)}", []
