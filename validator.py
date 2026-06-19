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

mapa_modelos = {
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

mapa_transformadores = {
    'StandardScaler': 'StandardScaler',
    'PCA': 'PCA',
    'SelectKBest': 'SelectKBest'
}

mapa_scalers = {
    'standard': 'StandardScaler',
    'minmax': 'MinMaxScaler'
}

metricas_validas = {
    'classification': {'accuracy', 'precision', 'recall', 'f1'},
    'regression': {'mse', 'rmse', 'mae', 'r2'}
}

params_validos = {
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
        'normalize': {'tipo': bool}
    },
    'PCA': {
        'n_components': {'tipo': int, 'min': 1},
        'random_state': {'tipo': int}
    },
    'SelectKBest': {
        'k': {'tipo': int, 'min': 1}
    }
}

transformadores_com_vazamento = {'StandardScaler', 'PCA', 'SelectKBest'}

mapa_modelos_sklearn = {
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

mapa_transformadores_sklearn = {
    'StandardScaler': StandardScaler,
    'PCA': PCA,
    'SelectKBest': SelectKBest
}

mapa_scalers_sklearn = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler
}

mapa_metricas_sklearn = {
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