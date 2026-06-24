"""
Demo: Diabetes - regressão (predição de progressão da doença)
Dataset gerado via sklearn.datasets.load_diabetes
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

if __name__ == "__main__":

    codigo = """
    /* Demo Diabetes - regressao */
    dataset "data/diabetes/diabetes.csv"
    target target
    problem regression
    split test_size=0.2, random_state=42
    pipeline {
        GradientBoosting {
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        }
    }
    evaluate r2
    """

    resultado = compilar(codigo, verbose=True)

    if resultado:
        print(f"\n[OK] RESULTADO FINAL:")
        print(f"  Score (r2): {resultado['score']:.4f}")
        print(f"  Modelo: {resultado['modelo_nome']}")
        print(f"  Features: {resultado['n_features']}")
        print(f"  Duração: {resultado['duracao_segundos']}s")
