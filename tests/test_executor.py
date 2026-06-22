import pytest
from src.executor import executar


def test_executor_returns_result():
    """Testa se executor retorna resultado com score"""
    ctx = {
        'dataset': 'data/iris/Iris.csv',
        'target': 'Species',
        'problema': 'classification',
        'split': {'test_size': 0.3, 'random_state': 42},
        'split_feito': True,
        'scaler': 'standard',
        'pipeline': [
            {
                'tipo': 'modelo',
                'nome': 'RandomForest',
                'params': {'n_estimators': 10, 'random_state': 42}
            }
        ],
        'metrica': 'accuracy',
    }
    resultado = executar(ctx)
    assert resultado is not None
    assert 'score' in resultado
    assert 'modelo' in resultado
    assert 0 <= resultado['score'] <= 1


def test_executor_with_transformers():
    """Testa execução com transformadores"""
    ctx = {
        'dataset': 'data/iris/Iris.csv',
        'target': 'Species',
        'problema': 'classification',
        'split': {'test_size': 0.3, 'random_state': 42},
        'split_feito': True,
        'scaler': 'standard',
        'pipeline': [
            {'tipo': 'transformador', 'nome': 'PCA', 'params': {'n_components': 2}},
            {
                'tipo': 'modelo',
                'nome': 'RandomForest',
                'params': {'n_estimators': 5, 'random_state': 42}
            }
        ],
        'metrica': 'accuracy',
    }
    resultado = executar(ctx)
    assert resultado is not None
    assert 'score' in resultado
    assert 0 <= resultado['score'] <= 1
