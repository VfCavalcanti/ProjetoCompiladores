from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import compilar

if __name__ == "__main__":
    # Exemplo de como usar com outro dataset
    # Para isto, coloque seu CSV em data/seu_dataset/dados.csv

    codigo = """
    dataset "data/seu_dataset/dados.csv"
    target label_coluna
    problem classification
    split test_size=0.2, random_state=42
    pipeline {
        SVM {
            C=1.0,
            kernel=rbf
        }
    }
    evaluate f1
    """

    resultado = compilar(codigo, verbose=True)

    if resultado:
        print(f"\n[OK] RESULTADO FINAL:")
        print(f"  Score: {resultado['score']:.4f}")
        print(f"  Modelo: {type(resultado['modelo']).__name__}")
    else:
        print("\n[ERRO] Falhou! Verifique se data/seu_dataset/dados.csv existe e tem a coluna 'label_coluna'")
