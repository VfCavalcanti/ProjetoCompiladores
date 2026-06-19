from main import compilar

if __name__ == "__main__":

    print("INTERPRETADOR DSL - PIPELINE DE MACHINE LEARNING")
    print("=" * 70)

    codigo = """
dataset "Iris.csv"
target Species
problem classification
split test_size=0.3, random_state=42
scaler standard
pipeline {
    StandardScaler,
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

    resultado = compilar(codigo)

    if resultado:
        print(f"\nRESULTADO FINAL:")
        print(f"  Score: {resultado['score']:.4f}")
        print(f"  Modelo: {type(resultado['modelo']).__name__}")