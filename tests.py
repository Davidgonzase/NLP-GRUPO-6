import pandas as pd
from rag_engine import FactChecker
from sklearn.metrics import classification_report, confusion_matrix
import time

def system_eval(frases_csv):
    # Inicializar el motor
    checker = FactChecker()
    df = pd.read_csv(frases_csv)
    
    results = []
    
    print(f"Iniciando evaluación de {len(df)} frases...")
    
    for index, row in df.iterrows():
        print(f"Procesando {index+1}/{len(df)}...")
        
        # Ejecutar el sistema RAG
        full_response, confidence, _ = checker.verify_claim(row['claim'], row['language'])
        
        # Extraer solo la primera palabra del veredicto para comparar
        veredict = full_response.splitlines()[0].upper()
        
        # Limpieza básica para normalizar la comparación
        pred = "INSUFFICIENT INFORMATION"
        if "TRUE" in veredict or "VERDADERO" in veredict:
            pred = "TRUE"
        elif "FALSE" in veredict or "FALSO" in veredict:
            pred = "FALSE"
            
        results.append({
            "claim": row['claim'],
            "esperado": row['expected_verdict'].upper(),
            "pred": pred,
            "confidence": confidence
        })
        # Pequeño sleep para no saturar el servidor si es externo
        time.sleep(0.5)

    # Crear DataFrame de resultados
    df_res = pd.DataFrame(results)
    
    # Mostrar reporte de clasificación
    print("\n" + "="*30)
    print(" REPORTE DE EVALUACIÓN")
    print("="*30)
    print(classification_report(df_res['esperado'], df_res['pred']))
    
    # Guardar para tu memoria del proyecto
    df_res.to_csv("resultados_finales.csv", index=False)
    print(" Resultados guardados en 'resultados_finales.csv'")

if __name__ == "__main__":
    system_eval("QUESTIONS.csv")