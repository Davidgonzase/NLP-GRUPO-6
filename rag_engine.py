import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import config
from langdetect import detect, LangDetectException
LANGUAGE_MAP = {
    'es': 'Spanish',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ru': 'Russian',
    'zh-cn': 'Chinese',
    'ja': 'Japanese'
}

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
        
        # 2. Detect Language
        try:
            detected_lang_code = detect(claim)
            detected_lang = LANGUAGE_MAP.get(detected_lang_code, "English")
            print(detected_lang)
        except LangDetectException:
            detected_lang = "English" # Fallback to English

        # 3. Construct Prompt
        prompt = f"""
You are an expert fact-checker. Your task is to verify a claim based ONLY on the provided context.

CONTEXT (Information Source):
{context_str}

CLAIM TO VERIFY (User Query):
"{claim}"

INSTRUCTIONS:
1. Analyze the claim and the context.
2. **CRITICAL:** The language of the claim is '{detected_lang}'. You MUST answer in '{detected_lang}'.
   - Do NOT answer in English unless '{detected_lang}' is 'en'.

VERDICT CRITERIA:
- SUPPORTED:
    * The context provides strong evidence that explicitly confirms the claim.
    * Minor details may differ if the core claim is accurate (e.g., "around 100 people" vs "102 people").
    * If the claim is a generalization effectively supported by specific examples in the context.
- REFUTED:
    * The context explicitly contradicts the core assertion of the claim.
    * The context provides mutually exclusive information (e.g., Claim: "X is red", Context: "X is blue").
    * Significant numerical or factual discrepancies exist.
- INSUFFICIENT INFO:
    * The context is unrelated to the claim.
    * The context mentions the subject but doesn't address the specific assertion in the claim.
    * The evidence is ambiguous or too vague to make a definitive judgment.
    * Do NOT hallucinate info not in the context to force a verdict.

OUTPUT FORMAT (STRICT):
The following output format MUST be written in the SAME LANGUAGE as the claim. This is a STRICT CRITERIA, you have to strictily respond in the same language as the claim.
This is the output format,translated to the claim language, you should follow:
- Verdict: [SUPPORTED | REFUTED | INSUFFICIENT INFO]
- Explanation: A brief reasoning based on the context.
- Quote: The original text from the context used as evidence. The quote extracted from the context must be in the same language as the claim.
NEVER INCLUDE in your output reasoning, opinions, chain of thinkings or any other text that is not the output format.

EXAMPLES OF CLAIMS AND RESPONSES:
(CLAIM with language english)
Claim: "The Eiffel Tower is located in Berlin."
Response:
Verdict: REFUTED
Explanation: The context states that the Eiffel Tower is in Paris, France, not Berlin.
Quote: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."

(CLAIM with language spanish)
Claim: "La inteligencia artificial puede superar a los humanos en tareas específicas."
Response:
Veredicto: RESPALDADO
Explicación: El contexto menciona que la inteligencia artificial ha superado a los humanos en juegos como el ajedrez y Go.
Cita:"La inteligencia artificial ha demostrado ser capaz de superar a los humanos en juegos complejos
Note how the output is translated to the claim language, the response is in the same language as the claim, including the Verdict,Explanation and Quote titles. This is the procedure you must follow strictly.
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
