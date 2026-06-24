from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

if __name__ == "__main__":

    codigo = """
    dataset "data/iris/Iris.csv"
    target Species
    problem classification
    split test_size=0.3, random_state=42
    scaler standard
    pipeline {
        PCA {
            n_components=2
        },
        RandomForest {
            n_estimators=100,
            max_depth=5,
            random_state=42
        }
    }
    evaluate accuracy
    """

    resultado = compilar(codigo, verbose=True)

    if resultado:
        print(f"\n[OK] RESULTADO FINAL:")
        print(f"  Score: {resultado['score']:.4f}")
        print(f"  Modelo: {type(resultado['modelo']).__name__}")
