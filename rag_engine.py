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
        
    def retrieve_context(self, query, n_results=6):
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

    def verify_claim(self, claim, target_lang="auto"):
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
            return "INFORMACIÓN INSUFICIENTE (No se encontraron documentos relevantes)", 0, []

        context_str = "\n\n".join(documents)
        print(context_str)
        prompt = f"""You are a precise fact-checking system. Verify the claim using ONLY the provided context.

CONTEXT:
{context_str}

CLAIM:
"{claim}"

FOLLOW THESE INSTRUCTIONS:
1. VERDICT RULES:
Define a veredict of TRUE when:
- Context explicitly confirms the claim's core assertion
- Facts exposed in the context mathces the entities(names, numbers, relationships) or affirmationes stablished on the claim
- Minor stylistic differences are acceptable (e.g., "CEO" vs "Chief Executive Officer")

Define a veredict of FALSE when:
- Context contradicts the claim with different facts
- Context explicitly states the opposite
- **CRITICAL:** Do NOT use FALSE simply because there in no information about the claim in the context. You must point to a specific sentence that makes the claim **impossible** to be true.

Define a veredict of INSUFFICIENT INFO when:
- Context doesn't address the specific claim
- Context is too vague or ambiguous
- Subject mentioned but assertion not covered
- Never infer beyond what's explicitly stated

2. QUOTE EXTRACTION CRITERIA:
- The quote must be valuable for evidencing your decision
- The quote must be extracted from the context
- The quote must not include information that is not relevant for supporting the verdict

3. OUTPUT FORMAT. You MUST follow the following format in your output. You MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the verdict, explanation and quote:
Verdict: [TRUE | FALSE | INSUFFICIENT INFO]\n
Explanation: [1-2 sentences comparing claim facts to context facts, highlighting matches or mismatches]\n
Quote: "[Quote extracted from the context supporting your verdict]"

4. OUTPUT EXAMPLES:
Example 1: TRUE:
Claim: "BERT was introduced by Google in 2018."
Context: "Google released BERT in 2018 as a breakthrough in NLP."
OUTPUT:
Verdict: TRUE
Explanation: The context confirms BERT was released by Google in 2018, matching all key entities in the claim.
Quote: "Google released BERT in 2018 as a breakthrough in NLP."

Example 2 - FALSE:
Claim: "The Eiffel Tower is in Berlin."
Context: "The Eiffel Tower is located in Paris, France."
OUTPUT:
Verdict: FALSE
Explanation: The claim states Berlin as the location, but the context explicitly states Paris—these are mutually exclusive cities.
Quote: "The Eiffel Tower is located in Paris, France."

Example 3 - INSUFFICIENT INFO:
Claim: "The company was founded by Sarah Chen."
Context: "The company has grown significantly since its founding."
OUTPUT:
Verdict: INSUFFICIENT INFO
Explanation: The context mentions the company's founding but doesn't specify who founded it.
Quote: "There is no information about the company's founders in the context."
"""
        try:
            response = ollama.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'user', 'content': prompt},
            ])
            result_english = response['message']['content']
            
        except Exception as e:
            return f"Error en LLM: {str(e)}", 0, []

        # --- Secondary Prompt: Confidence Calculation ---
        confidence_prompt = f"""
You are evaluating how certain a fact-checking verdict is based on the evidence.

CLAIM: "{claim_english}"
EVIDENCE: "{context_str}"
VERDICT: "{result_english}"

Assign a confidence score (0-100) representing how strongly the evidence supports this specific verdict.

HIGH CONFIDENCE (80-100):
TRUE verdict: Evidence explicitly confirms the claim with specific facts, dates, figures, or authoritative sources. Direct match with no ambiguity.
FALSE verdict: Evidence explicitly contradicts the claim with clear counter-evidence. The refutation is unambiguous.
INSUFFICIENT INFO verdict: Evidence clearly lacks any relevant information about the claim. It's obvious the context doesn't address this topic at all.

MODERATE CONFIDENCE (50-79):
TRUE verdict: Evidence supports the claim but requires reasonable inference or context interpretation. Partial information that points toward truth.
FALSE verdict: Evidence suggests the claim is false but doesn't completely refute it. Strong indicators of falsehood but with minor gaps.
INSUFFICIENT INFO verdict: Evidence mentions related topics but doesn't directly address the specific claim. Unclear if information is truly absent or just not explicitly stated.

LOW CONFIDENCE (20-49):
TRUE verdict: Evidence only tangentially supports the claim. Requires significant assumptions or logical leaps.
FALSE verdict: Evidence weakly contradicts the claim. Counter-evidence is vague or indirect.
INSUFFICIENT INFO verdict: Evidence might contain relevant information but it's ambiguous or incomplete. Hard to determine if context truly lacks information.

VERY LOW CONFIDENCE (0-19):
Any verdict where the evidence-to-verdict connection is highly questionable, contradictory, or the reasoning is fundamentally flawed.

Key principle: High confidence means you're CERTAIN about the verdict, not that the claim is true. You can be 95% confident that information is insufficient.

You MUST output only a number between 0-100 representing your estimated confidence in the verdict. You MUST NOT output any additional text.
        """
        
        try:
            conf_response = ollama.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'user', 'content': confidence_prompt},
            ])
            confidence_str = conf_response['message']['content'].strip()
            # Extract number even if there is text
            import re
            match = re.search(r'\d+', confidence_str)
            confidence_score = int(match.group()) if match else 0
        except Exception:
            confidence_score = 0


        # Determine final language
        final_lang = detected_lang if target_lang=="auto" else target_lang

        if final_lang != 'en':
            try:
                # Traducimos todo el bloque de respuesta
                final_response = GoogleTranslator(source='en', target=final_lang).translate(result_english)
            except Exception as e:
                final_response = result_english + f"\n(Error traduciendo respuesta: {e})"
        else:
            final_response = result_english

        return final_response, confidence_score, metadatas