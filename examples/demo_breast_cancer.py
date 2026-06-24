"""
Demo: Breast Cancer - classificação binária (maligno/benigno)
Dataset gerado via sklearn.datasets.load_breast_cancer
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

if __name__ == "__main__":

    codigo = """
    /* Demo Breast Cancer - classificacao binaria */
    dataset "data/breast_cancer/breast_cancer.csv"
    target target
    problem classification
    split test_size=0.2, random_state=42
    scaler standard
    pipeline {
        PCA { n_components=10 },
        RandomForest {
            n_estimators=100,
            max_depth=5,
            random_state=42
        }
    }
    evaluate f1
    """

    resultado = compilar(codigo, verbose=True)

    if resultado:
        print(f"\n[OK] RESULTADO FINAL:")
        print(f"  Score (f1): {resultado['score']:.4f}")
        print(f"  Modelo: {resultado['modelo_nome']}")
        print(f"  Features: {resultado['n_features']}")
        print(f"  Treino: {resultado['n_amostras_treino']} | Teste: {resultado['n_amostras_teste']}")
        print(f"  Duração: {resultado['duracao_segundos']}s")
