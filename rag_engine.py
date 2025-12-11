import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from ollama import Client
import config
from langdetect import detect, LangDetectException
import re
from deep_translator import GoogleTranslator


class FactChecker:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="fact_checking_knowledge_base")
        
        # Initialize Embedding Model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # Initialize Ollama Client
        self.llm_client = Client(
            host='esto_no_existe',
            headers={'X-API-KEY': 'api_key_123'} 
        )
        
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
        system_prompt = """You are an expert system that is specialized in classifying a given claim into one of the following three categories: TRUE,FALSE or INSUFFICIENT INFORMATION(used when the claim is not verifiable given the provided context). Verify the claim using ONLY the provided context.

FOLLOW THESE INSTRUCTIONS:
1. VERDICT RULES:
Your first step is determine wheter the claim can be refuted or confirmed using the provided context.If the context does not contain relevant information about the claim, you MUST classify it as INSUFFICIENT INFORMATION.
Here are some examples of claims and contexts that should be classified as INSUFFICIENT INFORMATION:
EXAMPLE 1 - INSUFFICIENT INFORMATION(information in the context isn't enough to stablish a veredict):
Context: "The city council approved a new zoning law to encourage mixed-use development downtown."
Claim: "The new zoning law includes provisions for affordable housing"
OUTPUT
Verdict: INSUFFICIENT INFORMATION\n
Explanation: The context does not provide any information about affordable housing provisions in the zoning law.

EXAMPLE 2 - INSUFFICIENT INFORMATION(claim information is not present in the context):
Context: "Dogs are known for their loyalty and companionship to humans."
Claim: "The global conference focused on cybersecurity advancements took place in Berlin"
OUTPUT
Verdict: INSUFFICIENT INFORMATION\n
Explanation: The context does not contain information about the conference's focus on cybersecurity.

In your output you MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the verdict and explanation. In this scenario, your output format must be the following. :
VERDICT: INSUFFICIENT INFO\n
EXPLANATION: The context does not contain enough information to verify (user's claim)  

If the context contains relevant information to verify or refute the claim, proceed to classify the claim as TRUE or FALSE based on the criteria below.
Define a veredict of TRUE when when at least one of the following criteria is met:
- Context explicitly confirms the claim's core assertion
- Facts exposed in the context mathces the entities(names, numbers, relationships) or affirmationes stablished on the claim
- Temporal statements in the claim aligns with the temporal statements in the context(dates, durations, sequences)
- The context contains the same information expressed in the claim with different words. Some paraphrasing is acceptable as long as the core facts align.

Define a veredict of FALSE when at least one of the following criteria is met:
- Context contains explicit information that makes the claim invalid.
- Context contains information that is mutually exclusive with the claim's assertion(X is of type A, but the claim states X is of type B).
- If the context provides the "true" version of a fact that is incorrectly stated in the claim, it is FALSE.
- Context contains temporal statements (dates, durations, or sequences) that prove the falsity of the claim(e.g., the claim states an event happened in october but the context says it happened in june).

Here are some expamples to illustrate the criteria for classifying a claim as TRUE OR FALSE:
Example 1 - TRUE (paraphrased information):
Context: "The merger was finalized on March 15, bringing together two industry leaders."
Claim: "The merger was completed in March"
OUTPUT
Verdict: TRUE\n
Explanation: "Finalized on March 15" confirms the merger was completed in March, matching the claim's core assertion.

Example 2 - TRUE (implied confirmation):
Context: "After five years as VP of Sales, Martinez was promoted to the executive suite as Chief Revenue Officer."
Claim: "Martinez is the Chief Revenue Officer"
OUTPUT
Verdict: TRUE\n
Explanation: The context explicitly states Martinez was promoted to Chief Revenue Officer, confirming the claim.

Example 3 - FALSE:
Context: "The company reported 450 employees across all locations."
Claim: "The company has 380 employees"
OUTPUT
Verdict: FALSE\n
Explanation: The context states 450 employees, which directly invalidates the claim of 380 employees.

Example 4 - FALSE (temporal contradiction):
Context: "The policy was announced on May of 2020."
Claim: "The policy was announced on September of 2020"
OUTPUT
Verdict: FALSE\n
Explanation: The context explicitly states the policy was announced on May of 2020, not in September.

Example 5 - FALSE (mutually exclusive information):
Context: "The 'Summit' supercomputer is powered by IBM Power9 CPUs and NVIDIA V100 GPUs, designed specifically for AI workloads."
Claim: "The Summit supercomputer runs on Intel Xeon processors"
OUTPUT
Verdict: FALSE\n
Explanation: The context specifies IBM Power9 CPUs, which invalidates the claim that it runs on Intel Xeon processors.

3. OUTPUT FORMAT. You MUST follow the following format in your output. You MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the verdict and explanation:
Verdict: [TRUE|FALSE]\n
Explanation: [Brief explanation comparing claim facts to context facts, highlighting matches or mismatches].
"""
        user_prompt = f"""
CONTEXT:
{context_str}

CLAIM:
"{claim}"
"""
        try:
            response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ], options={'temperature': 0.0})
            verdict_explanation = response['message']['content']

            # --- Secondary Prompt: Quote Extraction ---
            quote_system_prompt = """You are an expert on the task of quote extraction for supporting a verdict and explanation about a claim. 
If the verdict is TRUE OR FALSE,EXTRACT A QUOTE from the context that supports the following verdict and explanation for the claim. IF the verdict is INSUFFICIENT INFORMATION, JUST STATE THAT THE CONTEXT DOES NOT CONTAIN INFORMATION ABOUT THE CLAIM.

QUOTE EXTRACTION CRITERIA:
- The quote must contain valuable information for evidencing your decision
- The quote must be extracted from the context
- The quote must not include information that is not relevant for supporting the verdict
- If the verdict was INSUFFICIENT just state that the content of the context does not contain information about the claim

CONTEXT FIDELITY CONSTRAINT:
- The quote must be extracted from the context literally, without any paraphrasing or modification.
- It is forbidden to add information that is not present in the context into the quote.

EXAMPLES OF DESIRED QUOTE EXTRACTION:
Example 1 - TRUE verdict:
Context: "Apple Inc. announced its Q4 earnings, with CEO Tim Cook reporting revenue of $89.5 billion."
Claim: "Tim Cook is the CEO of Apple"
OUTPUT
Quote: "Apple Inc. announced its Q4 earnings, with CEO Tim Cook reporting ..."

Example 2 - FALSE verdict:
Context: "John Smith resigned from his position as CFO in March 2023, and was replaced by Maria Garcia."
Claim: "John Smith currently serves as CFO"
OUTPUT
Quote: "John Smith resigned from his position as CFO in March 2023, and was replaced by Maria Garcia."

Example 3 - INSUFFICIENT INFORMATION verdict:
Context: "The company launched three new products this quarter, focusing on sustainability."
Claim: "The company's revenue increased by 15% this quarter"
OUTPUT
Quote: "The context does not contain information about revenue or percentage increases"

3. OUTPUT FORMAT. You MUST follow the following format in your output. You MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the quote. Never include the previous claim or explanation in your output:
Quote: "[Quote literally extracted from the context supporting your verdict]"
"""
            quote_user_prompt = f"""
CONTEXT:
{context_str}

CLAIM:
"{claim}"

VERDICT AND EXPLANATION:
{verdict_explanation}
"""
            quote_response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': quote_system_prompt},
                {'role': 'user', 'content': quote_user_prompt},
            ], options={'temperature': 0.0})
            quote_result = quote_response['message']['content']
            
            # Regex to clean quote result (handling variations and extra text)
            quote_match = re.search(r'Quote:\s*["\']?(.*?)["\']?\s*$', quote_result, re.IGNORECASE | re.DOTALL)
            if quote_match:
                # If match found, use the extracted content formatted cleanly
                quote_cleaned = quote_match.group(1).strip()
                quote_result = f'Quote: "{quote_cleaned}"'
            
            print(quote_result)

            result_english = f"{verdict_explanation}\n{quote_result}"
            
        except Exception as e:
            return f"Error en LLM: {str(e)}", 0, []

        # --- Secondary Prompt: Confidence Calculation ---
        confidence_system_prompt = """
You are evaluating how certain a fact-checking verdict is based on the evidence.

Assign a confidence score (0-100) representing how strongly the evidence supports this specific verdict.

HIGH CONFIDENCE (80-100):
TRUE verdict: Evidence explicitly confirms the claim with specific facts, dates, figures, or authoritative sources. Direct match with no ambiguity.
FALSE verdict: Evidence explicitly contradicts the claim with clear counter-evidence. The refutation is unambiguous.
INSUFFICIENT INFORMATION verdict: Evidence clearly lacks any relevant information about the claim. It's obvious the context doesn't address this topic at all.

MODERATE CONFIDENCE (50-79):
TRUE verdict: Evidence supports the claim but requires reasonable inference or context interpretation. Partial information that points toward truth.
FALSE verdict: Evidence suggests the claim is false but doesn't completely refute it. Strong indicators of falsehood but with minor gaps.
INSUFFICIENT INFORMATION verdict: Evidence mentions related topics but doesn't directly address the specific claim. Unclear if information is truly absent or just not explicitly stated.

LOW CONFIDENCE (20-49):
TRUE verdict: Evidence only tangentially supports the claim. Requires significant assumptions or logical leaps.
FALSE verdict: Evidence weakly contradicts the claim. Counter-evidence is vague or indirect.
INSUFFICIENT INFORMATION verdict: Evidence might contain relevant information but it's ambiguous or incomplete. Hard to determine if context truly lacks information.

VERY LOW CONFIDENCE (0-19):
Any verdict where the evidence-to-verdict connection is highly questionable, contradictory, or the reasoning is fundamentally flawed.

Key principle: High confidence means you're CERTAIN about the verdict, not that the claim is true. You can be 95% confident that information is insufficient.

You MUST output only a number between 0-100 representing your estimated confidence in the verdict. You MUST NOT output any additional text.

OUTPUT EXAMPLES:

Example 1 (HIGH CONFIDENCE - TRUE):
Claim: "The Eiffel Tower was completed in 1889"
Evidence: "The Eiffel Tower, built for the 1889 World's Fair in Paris, was completed on March 31, 1889."
Verdict: TRUE
Output: 95

Example 2 (HIGH CONFIDENCE - FALSE):
Claim: "The Great Wall of China is visible from the Moon with the naked eye"
Evidence: "NASA astronauts have confirmed that the Great Wall of China is not visible from the Moon without aid. No human-made structures are visible from lunar distance with the naked eye."
Verdict: FALSE
Output: 98

Example 3 (HIGH CONFIDENCE - INSUFFICIENT INFORMATION):
Claim: "The mayor of Springfield announced a new recycling program in 2023"
Evidence: "Springfield's economic development has focused on attracting tech companies. The downtown area has seen significant retail growth."
Verdict: INSUFFICIENT INFORMATION
Output: 92
"""
        confidence_user_prompt = f"""
CLAIM: "{claim_english}"
EVIDENCE: "{context_str}"
VERDICT: "{result_english}"
"""
        
        try:
            conf_response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': confidence_system_prompt},
                {'role': 'user', 'content': confidence_user_prompt},
            ], options={'temperature': 0.0})
            confidence_str = conf_response['message']['content'].strip()
            # Extract number even if there is text
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