"""
Demo: Digits - classificação multiclasse (dígitos 0-9)
Dataset gerado via sklearn.datasets.load_digits
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

if __name__ == "__main__":

    codigo = """
    /* Demo Digits - classificacao multiclasse */
    dataset "data/digits/digits.csv"
    target target
    problem classification
    split test_size=0.2, random_state=42
    scaler standard
    pipeline {
        PCA { n_components=20 },
        SVM { C=1.0, kernel=rbf, gamma=scale }
    }
    evaluate accuracy
    """

    resultado = compilar(codigo, verbose=True)

    if resultado:
        print(f"\n[OK] RESULTADO FINAL:")
        print(f"  Score (accuracy): {resultado['score']:.4f}")
        print(f"  Modelo: {resultado['modelo_nome']}")
        print(f"  Features após PCA: {resultado['n_features']}")
        print(f"  Treino: {resultado['n_amostras_treino']} | Teste: {resultado['n_amostras_teste']}")
        print(f"  Duração: {resultado['duracao_segundos']}s")
