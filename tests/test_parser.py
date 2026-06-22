import pytest
from src.grammar import gramatica


def test_parse_valid_dsl():
    """Testa se DSL válida é parseada corretamente"""
    code = """
    dataset "test.csv"
    target label
    problem classification
    """
    tree = gramatica.parse(code)
    assert tree is not None


def test_parse_complete_pipeline():
    """Testa parsing de pipeline completo"""
    code = """
    dataset "data/iris/Iris.csv"
    target Species
    problem classification
    split test_size=0.3
    pipeline {
        RandomForest {
            n_estimators=100
        }
    }
    evaluate accuracy
    """
    tree = gramatica.parse(code)
    assert tree is not None


def test_parse_fails_on_invalid_syntax():
    """Testa se sintaxe inválida lança erro"""
    with pytest.raises(Exception):
        gramatica.parse("dataset")  # Incompleto


def test_parse_fails_on_incomplete_syntax():
    """Testa se sintaxe incompleta (só 'dataset') falha"""
    with pytest.raises(Exception):
        gramatica.parse("dataset")
