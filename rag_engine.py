import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import config
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

class FactChecker:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="fact_checking_knowledge_base")
        
        # Initialize Embedding Model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
    def retrieve_context(self, query, n_results=5):
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
        Verifies a claim using a Translation -> RAG -> Translation pipeline.
        """
        try:
            detected_lang = detect(claim)
        except LangDetectException:
            detected_lang = 'en'
            
        print(f"🌍 Idioma detectado: {detected_lang}")

        # Si no es inglés, traducimos la consulta para el sistema
        if detected_lang != 'en':
            try:
                claim_english = GoogleTranslator(source=detected_lang, target='en').translate(claim)
            except Exception as e:
                print(f"⚠️ Error traduciendo input: {e}")
                claim_english = claim # Fallback
        else:
            claim_english = claim

        documents, metadatas = self.retrieve_context(claim_english)
        
        if not documents:
            return "INFORMACIÓN INSUFICIENTE (No se encontraron documentos relevantes)", []

        context_str = "\n\n".join(documents)
        print(context_str)
        prompt = f"""You are a precise fact-checking system. Verify the claim using ONLY the provided context.

CONTEXT:
{context_str}

CLAIM:
"{claim}"

VERIFICATION PROCESS:

1. ENTITY EXTRACTION
   - Identify key entities in the claim: names, numbers, dates, locations, technologies, organizations
   - Extract corresponding entities from the context
   
2. LOGICAL COMPARISON
   - Compare extracted facts directly
   - Named entities are mutually exclusive: "GPT" ≠ "BERT", "Paris" ≠ "Berlin", "2020" ≠ "2021"
   - Check if the relationship/assertion matches between claim and context
   
3. EVIDENCE ASSESSMENT
   Critical distinctions:
   - Core facts vs. peripheral details (e.g., "approximately 100" vs "102" is acceptable variance)
   - Explicit statements vs. implications
   - Direct contradictions vs. missing information

VERDICT RULES:

SUPPORTED - Use when:
- Context explicitly confirms the claim's core assertion
- Facts exposed in the context mathces the entities(names, numbers, relationships) or affirmationes stablished on the claim
- Minor stylistic differences are acceptable (e.g., "CEO" vs "Chief Executive Officer")

REFUTED - Use when:
- Context contradicts the claim with different facts
- Key entities mismatch (different names, numbers, dates between the facts in the context and the claim)
- Context explicitly states the opposite

INSUFFICIENT INFO - Use when:
- Context doesn't address the specific claim
- Context is too vague or ambiguous
- Subject mentioned but assertion not covered
- Never infer beyond what's explicitly stated

OUTPUT FORMAT. You MUST follow the following format in your output. You MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the verdict, explanation and quote:

Verdict: [SUPPORTED | REFUTED | INSUFFICIENT INFO]\n
Explanation: [1-2 sentences comparing claim facts to context facts, highlighting matches or mismatches]\n
Quote: "[Exact text from context supporting your verdict. This quote must be valuable for evidencing your decision, so avoid including non-informative text]"

OUTPUT EXAMPLES:

Example 1: SUPPORTED:
Claim: "BERT was introduced by Google in 2018."
Context: "Google released BERT in 2018 as a breakthrough in NLP."
OUTPUT:
Verdict: SUPPORTED
Explanation: The context confirms BERT was released by Google in 2018, matching all key entities in the claim.
Quote: "Google released BERT in 2018 as a breakthrough in NLP."

Example 2 - REFUTED:
Claim: "The Eiffel Tower is in Berlin."
Context: "The Eiffel Tower is located in Paris, France."
OUTPUT:
Verdict: REFUTED
Explanation: The claim states Berlin as the location, but the context explicitly states Paris—these are mutually exclusive cities.
Quote: "The Eiffel Tower is located in Paris, France."

Example 3 - INSUFFICIENT INFO:
Claim: "The company was founded by Sarah Chen."
Context: "The company has grown significantly since its founding."
OUTPUT:
Verdict: INSUFFICIENT INFO
Explanation: The context mentions the company's founding but doesn't specify who founded it.
Quote: "The company has grown significantly since its founding."
"""
        try:
            response = ollama.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'user', 'content': prompt},
            ])
            result_english = response['message']['content']
            
        except Exception as e:
            return f"Error en LLM: {str(e)}", []

        if detected_lang != 'en':
            try:
                # Traducimos todo el bloque de respuesta
                final_response = GoogleTranslator(source='en', target=detected_lang).translate(result_english)
            except Exception as e:
                final_response = result_english + f"\n(Error traduciendo respuesta: {e})"
        else:
            final_response = result_english

        return final_response, metadatas