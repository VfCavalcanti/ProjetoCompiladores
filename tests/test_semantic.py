import pytest
from src.grammar import gramatica
from src.semantic_analyzer import analiseSemantica


def test_semantic_valid_classification():
    """Testa análise semântica de classificação válida"""
    code = """
    dataset "data/iris/Iris.csv"
    target Species
    problem classification
    split test_size=0.3
    pipeline {
        RandomForest {
            n_estimators=10
        }
    }
    evaluate accuracy
    """
    tree = gramatica.parse(code)
    ctx = analiseSemantica(tree)
    assert len(ctx['erros']) == 0
    assert ctx['problema'] == 'classification'
    assert ctx['target'] == 'Species'


def test_semantic_invalid_target():
    """Testa se target inválido é detectado"""
    code = """
    dataset "data/iris/Iris.csv"
    target coluna_inexistente
    problem classification
    split test_size=0.3
    pipeline {
        RandomForest { n_estimators=10 }
    }
    evaluate accuracy
    """
    tree = gramatica.parse(code)
    ctx = analiseSemantica(tree)
    assert len(ctx['erros']) > 0


def test_semantic_invalid_problem_model_combination():
    """Testa incompatibilidade modelo-problema"""
    code = """
    dataset "data/iris/Iris.csv"
    target SepalLengthCm
    problem regression
    split test_size=0.3
    pipeline {
        RandomForest {
            n_estimators=10
        }
    }
    evaluate accuracy
    """
    tree = gramatica.parse(code)
    ctx = analiseSemantica(tree)
    # LinearRegression é regressão, RandomForest é regression, accuracy é classification
    # Deve detectar erro
    assert len(ctx['erros']) > 0


def test_semantic_data_leakage_warning():
    """Testa detecção de possível vazamento de dados"""
    code = """
    dataset "data/iris/Iris.csv"
    target Species
    problem classification
    pipeline {
        StandardScaler,
        RandomForest { n_estimators=10 }
    }
    evaluate accuracy
    """
    tree = gramatica.parse(code)
    ctx = analiseSemantica(tree)
    # Deve ter erro por vazamento (StandardScaler antes do split)
    assert len(ctx['erros']) > 0
