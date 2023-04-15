import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def evaluate_model(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:

    param_grid = get_grid_parameters(model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    grid = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_parameters = grid.best_params_
    accuracy = grid.best_score_

    return best_model, best_parameters, accuracy

def get_grid_parameters(model: object) -> dict:
    model_name = type(model).__name__.lower()

    if model_name == 'xgbclassifier':
        return {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.5, 0.8],
            'colsample_bytree': [0.5, 0.8],
        }
    elif model_name == 'onevsrestclassifier':
        return {
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        }
    else:
        raise ValueError(f'Unsupported model type: {model_name}')