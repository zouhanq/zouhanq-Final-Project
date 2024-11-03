# model_training.py

from sklearn.calibration import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree classifier with hyperparameter tuning."""
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=tscv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters for Decision Tree:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model  # Only the model is returned

def train_svm(X_train, y_train):
    """Train an SVM classifier with a linear kernel and simplified hyperparameter tuning."""
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save feature names
    feature_names = X_train.columns

    # Simplified Parameter Grid
    param_grid = {'C': [0.1, 1, 10]}
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        LinearSVC(max_iter=1000),
        param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)
    print("Best parameters for Linear SVM:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # Return the model, scaler, and feature names
    return best_model, scaler, feature_names

def train_knn(X_train, y_train, n_neighbors=5):
    """Train a K-Nearest Neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


