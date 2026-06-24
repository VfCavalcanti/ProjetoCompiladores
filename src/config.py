import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures
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
    },
    'GradientBoosting': {
        'classification': 'GradientBoostingClassifier',
        'regression': 'GradientBoostingRegressor'
    },
    'LogisticRegression': {
        'classification': 'LogisticRegression'
    },
    'Ridge': {
        'regression': 'Ridge'
    },
    'Lasso': {
        'regression': 'Lasso'
    },
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
    },
    'GradientBoosting': {
        'classification': GradientBoostingClassifier,
        'regression': GradientBoostingRegressor
    },
    'LogisticRegression': {
        'classification': LogisticRegression
    },
    'Ridge': {
        'regression': Ridge
    },
    'Lasso': {
        'regression': Lasso
    },
}

# Mapa de strings de score_func para funções sklearn
MAPA_SCORE_FUNC = {
    'f_classif': f_classif,
    'f_regression': f_regression,
}

MAPA_TRANSFORMADORES_SKLEARN = {
    'StandardScaler': StandardScaler,
    'PCA': PCA,
    'SelectKBest': SelectKBest,
    'RobustScaler': RobustScaler,
    'PolynomialFeatures': PolynomialFeatures,
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
    'GradientBoosting': {
        'n_estimators': {'tipo': int, 'min': 1},
        'max_depth': {'tipo': int, 'min': 1},
        'learning_rate': {'tipo': float, 'min': 0},
        'random_state': {'tipo': int}
    },
    'LogisticRegression': {
        'C': {'tipo': float, 'min': 0},
        'max_iter': {'tipo': int, 'min': 1},
        'random_state': {'tipo': int}
    },
    'Ridge': {
        'alpha': {'tipo': float, 'min': 0},
        'fit_intercept': {'tipo': bool},
    },
    'Lasso': {
        'alpha': {'tipo': float, 'min': 0},
        'fit_intercept': {'tipo': bool},
        'max_iter': {'tipo': int, 'min': 1},
    },
    'PCA': {
        'n_components': {'tipo': int, 'min': 1},
        'random_state': {'tipo': int}
    },
    'SelectKBest': {
        'k': {'tipo': int, 'min': 1},
        'score_func': {'tipo': str, 'opcoes': ['f_classif', 'f_regression']}
    },
    'RobustScaler': {},
    'PolynomialFeatures': {
        'degree': {'tipo': int, 'min': 2},
    },
}

# Transformadores que causam vazamento de dados
TRANSFORMADORES_COM_VAZAMENTO = {'StandardScaler', 'PCA', 'SelectKBest', 'RobustScaler', 'PolynomialFeatures'}
