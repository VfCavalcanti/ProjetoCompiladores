from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

codigo_dsl = """
dataset "data/wine/wine.csv"
target quality
problem classification
split test_size=0.3, random_state=42
scaler standard
pipeline {
    StandardScaler,
    RandomForest {
        n_estimators=50,
        max_depth=6,
        random_state=42
    }
}
evaluate accuracy
"""

if __name__ == "__main__":
    print("--- Executando Experimento: Dataset Wine ---")
    resultado = compilar(codigo_dsl, verbose=True)
    if resultado:
        print(f"\n[Sucesso] Score final (Accuracy): {resultado['score']:.4f}")