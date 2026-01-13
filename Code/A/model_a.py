from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


class ModelA:
    """
    Wraps the SVM classifier workflow, including training, grid search, and evaluation.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42
        )
        self.best_params = None

    def train(self, X_train, y_train):
        """
        Fits the SVM model to the training data.

        Args:
            X_train (array-like): Feature matrix.
            y_train (array-like): Target labels.
        """
        print(f"Training SVM (Kernel: {self.kernel}, C: {self.C})...")
        self.model.fit(X_train, y_train)

    def grid_search(self, X_train, y_train):
        """
        Performs Grid Search to find the best complexity parameters.
        Useful for discussing Model Complexity.
        """
        print(">> Running Grid Search to analyze Model Complexity...")
        param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]}

        grid = GridSearchCV(
            SVC(probability=True), param_grid, cv=3, scoring="accuracy", verbose=2
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        print(f"Best Parameters found: {self.best_params}")

        return pd.DataFrame(grid.cv_results_)

    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Predicts on test data and prints classification metrics.

        Args:
            X_test (array-like): Test features.
            y_test (array-like): True test labels.
            dataset_name (str): Label for the print output (e.g. 'Validation').

        Returns:
            tuple: (accuracy, precision, recall, f1)
        """
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"--- {dataset_name} Metrics (Model A) ---")
        print(
            f"Accuracy:  {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}"
        )

        return acc, precision, recall, f1
