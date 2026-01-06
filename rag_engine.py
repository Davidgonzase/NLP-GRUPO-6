import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from ollama import Client
import config
from langdetect import detect, LangDetectException
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Clase principal de verificación de hechos
class FactChecker:
    # Mapeo de códigos de idioma ISO a NLLB
    ISO_TO_NLLB = {
        'es': 'spa_Latn',
        'en': 'eng_Latn',
        'fr': 'fra_Latn',
        'it': 'ita_Latn',
        'de': 'deu_Latn',
        'pt': 'por_Latn',
        'ru': 'rus_Cyrl',
        'zh': 'zho_Hans',
        'zh-CN': 'zho_Hans',
        'ja': 'jpn_Jpan',
        'nl': 'nld_Latn',
        'pl': 'pol_Latn',
        'ar': 'arb_Arab',
        'tr': 'tur_Latn',
        'ko': 'kor_Hang'
    }

    def __init__(self):
        # Inicializacion de ChromaDB, modelo de embeddings, LLM y traductor
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="fact_checking_knowledge_base")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.llm_client = Client(
            host=config.OLLAMA_HOST,
            headers={'X-API-KEY': config.OLLAMA_API_KEY}
        )
        
        # Iniciado del modelo de traducción NLLB usado para traducciones
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.translator = pipeline("translation", model=model, tokenizer=tokenizer)

        # Detector de idioma usando xlm-roberta-base-language-detection
        self.detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    
    # Recuperación de contexto relevante desde ChromaDB
    def retrieve_context(self, query, n_results=6):
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        return documents, metadatas

    # Verificación de afirmaciones usando pipeline Traducción -> RAG -> Traducción
    def verify_claim(self, claim, target_lang="auto"):
        # Detección de idioma del input
        try:
            detected_lang = self.detector(claim)[0]['label']
        except LangDetectException:
            detected_lang = 'en'
            
        print(f"Idioma detectado: {detected_lang}")

        # Si no es inglés, traducimos la consulta para el sistema
        if detected_lang != 'en':
            try:
                nllb_source = self.ISO_TO_NLLB.get(detected_lang, 'eng_Latn')
                claim_english = self.translator(claim, src_lang=nllb_source, tgt_lang="eng_Latn", max_length=512)[0]['translation_text']

                print(f"Traducción al inglés: {claim_english}")
                
            except Exception as e:
                print(f"Error traduciendo input: {e}")
                claim_english = claim # Fallback
        else:
            claim_english = claim

        # Recuperación de contexto relevante en base al texto original o traducido en inglés
        documents, metadatas = self.retrieve_context(claim_english)
        
        # En caso de que la afirmación no se encuentre en la base de conocimiento informamos en la APP
        if not documents:
            return "INFORMACIÓN INSUFICIENTE (No se encontraron documentos relevantes)", 0, []

        # Construcción del prompt para el LLM
        context_str = "\n\n".join(documents)
        print(context_str)
        # En este prompt se incluyen reglas estrictas para la verificación de hechos, junto con ejemplos detallados y el formato de salida esperado
        # Su propósito es guiar al modelo para que evalúe la veracidad de una afirmación basándose únicamente en el contexto proporcionado y que devuelva un veredicto False, True o Insufficient Information
        # Para ello hemos indicado al modelo que debe actuar como si no tuviera conocimiento previo del mundo, basándose únicamente en el contexto proporcionado
        # Esto es crucial para evitar que el modelo utilice "conocimiento general" que no esté presente en el contexto recuperado y que se adapte a las reglas y ejemplos dados
        system_prompt = """You are an expert system specialized in verifying claims using ONLY provided textual evidence. You must classify a given claim into one of three categories: TRUE, FALSE, or INSUFFICIENT INFORMATION.

*** CRITICAL INSTRUCTION ***
You must act as if you have NO prior knowledge of the world. You must ignore all facts, history, science, or common knowledge that is not explicitly written in the provided CONTEXT. 
- Even if a claim is a universally known fact (e.g., "The Earth is round"), if it is not mentioned in the CONTEXT, you MUST classify it as INSUFFICIENT INFORMATION.
- Even if a claim is physically impossible or universally false (e.g., "Humans can fly"), if the CONTEXT does not contradict it, you MUST classify it as INSUFFICIENT INFORMATION.

1. VERDICT RULES:

Determine the verdict based strictly on the following logic:

VERDICT: INSUFFICIENT INFORMATION
Use this when the context does not contain the necessary information to prove or disprove the claim.
- If the claim is about a topic not mentioned in the context.
- If the claim relies on external knowledge (common sense, geography, history) not present in the text.
- If the claim relies on a relation with an element from the text but the relation is not in the text.

VERDICT: TRUE
Use this ONLY when the context provides explicit evidence supporting the claim.
- The context explicitly states the information in the claim.
- The context implies the claim through synonymous phrasing or logical consequence of the text provided.

VERDICT: FALSE
Use this ONLY when the context explicitly contradicts the claim.
- The context contains information that is mutually exclusive to the claim.
- The context contains information that makes the claim invalid directly or by inference.
- The context contains information that contradicts the claim.
- The context contains information with which the claim's falsity can be inferred.
- The context provides a specific value/date/name that differs from the claim.

2. EXAMPLES:

EXAMPLE 1 - INSUFFICIENT INFO (Topic missing):
Context: "The city council approved a new zoning law to encourage mixed-use development."
Claim: "The new zoning law includes provisions for affordable housing"
OUTPUT:
Verdict: Insufficient information
Explanation: The context does not provide any information about affordable housing provisions.

EXAMPLE 2 - INSUFFICIENT INFO (Universal Truth Trap - CRITICAL):
Context: "The software update v2.0 fixed several bugs in the login module."
Claim: "The sun rises in the east"
OUTPUT:
Verdict: Insufficient information
Explanation: While factually true in the real world, the provided context does not mention the sun or its movement.

EXAMPLE 3 - INSUFFICIENT INFO (Universal Falsehood Trap):
Context: "John went to the grocery store to buy milk."
Claim: "The moon is made of green cheese"
OUTPUT:
Verdict: Insufficient information
Explanation: The context describes John's shopping trip and does not contain information to refute the composition of the moon.

EXAMPLE 4 - INSUFFICIENT INFO (Relation not present):
Context: "The calculations for the trajectory of the rocket are mathematically correct"
Claim: "2+2=3"
OUTPUT:
Verdict: Insufficient information
Explanation: The context mentions correct mathematic operations, but does not contain concrete information about concrete operations.

ECAMPLE 4 - TRUE (paraphrased information):
Context: "The merger was finalized on March 15, bringing together two industry leaders."
Claim: "The merger was completed in March"
OUTPUT
Verdict: True
Explanation: "Finalized on March 15" confirms the merger was completed in March, matching the claim's core assertion.

EXAMPLE 5 - TRUE (implied confirmation):
Context: "After five years as VP of Sales, Martinez was promoted to the executive suite as Chief Revenue Officer."
Claim: "Martinez is the Chief Revenue Officer"
OUTPUT
Verdict: True
Explanation: The context explicitly states Martinez was promoted to Chief Revenue Officer, confirming the claim.

EXAMPLE 6 - FALSE:
Context: "The company reported 450 employees across all locations."
Claim: "The company has 380 employees"
OUTPUT
Verdict: False
Explanation: The context states 450 employees, which directly invalidates the claim of 380 employees.

EXAMPLE 7 - FALSE (temporal contradiction):
Context: "The policy was announced on May of 2020."
Claim: "The policy was announced on September of 2020"
OUTPUT
Verdict: False
Explanation: The context explicitly states the policy was announced on May of 2020, not in September.

EXAMPLE 8 - FALSE (mutually exclusive information):
Context: "The 'Summit' supercomputer is powered by IBM Power9 CPUs and NVIDIA V100 GPUs, designed specifically for AI workloads."
Claim: "The Summit supercomputer runs on Intel Xeon processors"
OUTPUT
Verdict: False
Explanation: The context specifies IBM Power9 CPUs, which invalidates the claim that it runs on Intel Xeon processors.

EXAMPLE 9 - FALSE (falsity inferred):
Context: A phone can be used to open applications.
Claim: Instagram, an application, cannot be used by a phone.
OUTPUT
Verdict: False
Explanation: The context specifies that phones can open applications, which contradicts the claim that Instagram cannot be used by a phone.

3. OUTPUT FORMAT:
You must strictly follow this format. Do not include internal reasoning or preamble.
Verdict: [True|False|Insufficient information]
Explanation: [Brief justification based ONLY on the text]\n
"""
        user_prompt = f"""
CONTEXT:
{context_str}

CLAIM:
"{claim}"
"""
        # LLamada al LLM para obtener veredicto y explicación
        try:
            response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ], options={'temperature': 0.0})
            verdict_explanation = response['message']['content']

            # Segunda llamada al LLM para extracción de cita relevante
            # Al igual que antes, se incluyen instrucciones detalladas y ejemplos para guiar al modelo
            # Su propósito es obterner una cita textual del contexto que respalde el veredicto dado 
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
            # Segunda llamada al LLM para extracción de cita relevante
            quote_response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': quote_system_prompt},
                {'role': 'user', 'content': quote_user_prompt},
            ], options={'temperature': 0.0})
            quote_result = quote_response['message']['content']
            
            # Limpieza del resultado para asegurar formato correcto
            quote_match = re.search(r'Quote:\s*["\']?(.*?)["\']?\s*$', quote_result, re.IGNORECASE | re.DOTALL)
            if quote_match:
                # En caso de encontrar el formato correcto, extraemos la cita
                quote_cleaned = quote_match.group(1).strip()
                quote_result = f'Quote: "{quote_cleaned}"'
            
            print(quote_result)
            result_english_inter = f"{verdict_explanation}\n{quote_result}"
            print(f"RESULT INTER: {result_english_inter}")
            
        except Exception as e:
            return f"Error en LLM: {str(e)}", 0, []

        # Tercer LLM Prompt: Resumen de la evidencia
        # Este prompt tiene como objetivo generar un resumen conciso de la evidencia presentada en el contexto
        # El resumen ayuda a sintetizar la información clave que respalda el veredicto, facilitando su comprensión
        summary_system_prompt = """You are an expert on the task of making summaries about a set of paragraphs. 
This set of paragraphs is the EVIDENCE given. The summary MUST have between 40 and 100 words.

OUTPUT FORMAT. You MUST NOT include reasoning, chain of thoughts, explanations, etc. just limit yourself to return the summary . Never include the previous claim or explanation in your output.
"""

        summary_user_prompt = f"""
EVIDENCE: "{context_str}"
"""
        # Tercera llamada al LLM para resumen de evidencia
        try:
            summary_response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': summary_system_prompt},
                {'role': 'user', 'content': summary_user_prompt},
            ], options={'temperature': 0.2})
            summary_str = summary_response['message']['content'].strip()
            summary_str = f"Summary: {summary_str}"

            result_english = f"{result_english_inter}\n{summary_str}"

            print(f"RESULT ENG: {result_english}")
        except Exception:
            return f"Error en LLM: {str(e)}", 0, []

        # Cuarto LLM Prompt: Evaluación de confianza
        # Este prompt está diseñado para que el modelo evalúe cuán confiable es el veredicto dado el contexto
        # Se proporcionan criterios detallados para asignar una puntuación de confianza entre 0 y 100
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
Verdict: True
Output: 95

Example 2 (HIGH CONFIDENCE - FALSE):
Claim: "The Great Wall of China is visible from the Moon with the naked eye"
Evidence: "NASA astronauts have confirmed that the Great Wall of China is not visible from the Moon without aid. No human-made structures are visible from lunar distance with the naked eye."
Verdict: False
Output: 98

Example 3 (HIGH CONFIDENCE - INSUFFICIENT INFORMATION):
Claim: "The mayor of Springfield announced a new recycling program in 2023"
Evidence: "Springfield's economic development has focused on attracting tech companies. The downtown area has seen significant retail growth."
Verdict: Insufficient information
Output: 92
"""
        confidence_user_prompt = f"""
CLAIM: "{claim_english}"
EVIDENCE: "{context_str}"
VERDICT: "{result_english}"
"""
        # Cuarta y última llamada al LLM para evaluación de confianza
        try:
            conf_response = self.llm_client.chat(model=config.LLM_MODEL_NAME, messages=[
                {'role': 'system', 'content': confidence_system_prompt},
                {'role': 'user', 'content': confidence_user_prompt},
            ], options={'temperature': 0.0})
            confidence_str = conf_response['message']['content'].strip()
            # Extraemos el número incluso si hay texto adicional
            match = re.search(r'\d+', confidence_str)
            confidence_score = int(match.group()) if match else 0
        except Exception:
            confidence_score = 0


        # Determinamos el idioma final de la respuesta
        final_lang = detected_lang if target_lang=="auto" else target_lang

        if final_lang != 'en':
            try:
                # Traducimos todo el bloque de respuesta
                nllb_dest = self.ISO_TO_NLLB.get(final_lang, 'eng_Latn')

                final_verdict_explanation = self.translator(verdict_explanation, src_lang='eng_Latn', tgt_lang=nllb_dest, max_length=512)[0]['translation_text']
                final_quote_result = self.translator(quote_result, src_lang='eng_Latn', tgt_lang=nllb_dest, max_length=512)[0]['translation_text']
                final_summary_str = self.translator(summary_str, src_lang='eng_Latn', tgt_lang=nllb_dest, max_length=512)[0]['translation_text']

                final_response = f"{final_verdict_explanation}\n{final_quote_result}\n{final_summary_str}"

                print(f"ESTA ES LA RESPUESTA: {final_response}")
                
            except Exception as e:
                final_response = result_english + f"\n(Error traduciendo respuesta: {e})"
        else:
            final_response = result_english

        # Devolvemos la respuesta final, la puntuación de confianza y los metadatos de las fuentes
        return final_response, confidence_score, metadatas