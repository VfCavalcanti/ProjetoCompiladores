import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Mapa de nomes DSL para nomes de classe sklearn (strings)
# Usado na análise semântica
MAPA_MODELOS = {
    'RandomForest': {
        'classification': 'RandomForestClassifier',
        'regression': 'RandomForestRegressor'
    },
    'SVM': {
        'classification': 'SVC',
        'regression': 'SVR'
    },
    'LinearRegression': {
        'regression': 'LinearRegression'
    }
}

# Mapa de nomes DSL para classes sklearn
# Usado na execução
MAPA_MODELOS_SKLEARN = {
    'RandomForest': {
        'classification': RandomForestClassifier,
        'regression': RandomForestRegressor
    },
    'SVM': {
        'classification': SVC,
        'regression': SVR
    },
    'LinearRegression': {
        'regression': LinearRegression
    }
}

MAPA_TRANSFORMADORES_SKLEARN = {
    'StandardScaler': StandardScaler,
    'PCA': PCA,
    'SelectKBest': SelectKBest
}

MAPA_ESCALADORES_SKLEARN = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler
}

MAPA_METRICAS_SKLEARN = {
    'classification': {
        'accuracy': accuracy_score,
        'precision': lambda y, y_pred: precision_score(y, y_pred, average='weighted'),
        'recall': lambda y, y_pred: recall_score(y, y_pred, average='weighted'),
        'f1': lambda y, y_pred: f1_score(y, y_pred, average='weighted')
    },
    'regression': {
        'mse': mean_squared_error,
        'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error,
        'r2': r2_score
    }
}

# Métricas válidas por tipo de problema
METRICAS_VALIDAS = {
    'classification': {'accuracy', 'precision', 'recall', 'f1'},
    'regression': {'mse', 'rmse', 'mae', 'r2'}
}

# Parâmetros válidos para cada modelo/transformador
PARAMS_VALIDOS = {
    'RandomForest': {
        'n_estimators': {'tipo': int, 'min': 1},
        'max_depth': {'tipo': int, 'min': 1},
        'random_state': {'tipo': int}
    },
    'SVM': {
        'C': {'tipo': float, 'min': 0},
        'kernel': {'tipo': str, 'opcoes': ['linear', 'rbf', 'poly']},
        'gamma': {'tipo': str, 'opcoes': ['scale', 'auto']}
    },
    'LinearRegression': {
        'fit_intercept': {'tipo': bool},
    },
    'PCA': {
        'n_components': {'tipo': int, 'min': 1},
        'random_state': {'tipo': int}
    },
    'SelectKBest': {
        'k': {'tipo': int, 'min': 1}
    }
}

# Transformadores que causam vazamento de dados
TRANSFORMADORES_COM_VAZAMENTO = {'StandardScaler', 'PCA', 'SelectKBest'}

