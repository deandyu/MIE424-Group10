import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def evaluate_model(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Evaluate a given model on the provided training data using a grid search over a range of hyperparameters.

    Args:
        model (object): An instance of the model to be evaluated.
        X_train (pd.DataFrame): A Pandas dataframe containing the feature data for the training set.
        y_train (pd.Series): A Pandas series containing the target labels for the training set.

    Returns:
        tuple: A tuple containing the best model, best hyperparameters, and accuracy score.

    """
    param_grid = get_grid_parameters(model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    grid = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_parameters = grid.best_params_
    accuracy = grid.best_score_

    return best_model, best_parameters, accuracy

def get_grid_parameters(model: object) -> dict:
    """
    Get a grid of hyperparameters for a given model type.

    Parameters:
        model (object): An instance of the model for which to retrieve hyperparameters.

    Returns:
        dict: A dictionary containing the hyperparameters to be used in a grid search.

    Raises:
        ValueError: If the provided model type is not supported.

    """
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