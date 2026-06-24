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
    crossvalidation folds=5, scoring=accuracy
    pipeline {
        StandardScaler,
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
        print(f"\n[OK] RESULTADO FINAL (CV):")
        print(f"  Scoring: {resultado['scoring']}")
        print(f"  Media:   {resultado['score']:.4f}")
        print(f"  Desvio:  {resultado['std']:.4f}")
        print(f"  Folds:   {resultado['cv_scores']}")
