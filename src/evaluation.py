# src/evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score
# from ..src.config_models import EvaluateModelConfig # Example for type hinting
# from pydantic import validate_call # For validating inputs

# TODO: Define expected input/output schemas for data (y_true, y_pred_proba, y_pred).
#       Ensure consistency with how predictions are formatted by the modeling module.

def plot_roc_auc(y_true, y_pred_proba, ax=None, model_name: str = "", **kwargs):
    """
    Plots the ROC curve and displays AUC.

    Args:
        y_true: True binary labels. # TODO: Validate type and shape.
        y_pred_proba: Probabilities of the positive class. # TODO: Validate type and shape.
        ax: Matplotlib Axes object to plot on. If None, creates a new one.
        model_name: Name of the model for the plot title.
        **kwargs: Additional arguments passed to RocCurveDisplay.from_predictions.

    Returns:
        Matplotlib Axes object.
    """
    # TODO: Add try-except for robustness, e.g., if y_true/y_pred_proba are not suitable for ROC.
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        # Use model_name for the name parameter in from_predictions for legend if needed elsewhere
        RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax, name=model_name, **kwargs)
        auc = roc_auc_score(y_true, y_pred_proba) # This might fail if inputs are bad
        title = f"ROC Curve (AUC = {auc:.2f})"
        if model_name:
            title = f"{model_name} - {title}"
        ax.set_title(title)
    except ValueError as ve: # More specific error for roc_auc_score or from_predictions issues
        ax.text(0.5, 0.5, f"ROC Plot Error: {ve}", ha='center', va='center', wrap=True)
        print(f"ValueError plotting ROC for {model_name}: {ve}")
    except Exception as e: # Generic catch-all
        ax.text(0.5, 0.5, f"Could not plot ROC: {e}", ha='center', va='center', wrap=True)
        print(f"Error plotting ROC for {model_name}: {e}")
        
    return ax

def plot_confusion_matrix(y_true, y_pred, ax=None, class_names=None, model_name: str = "", **kwargs):
    # TODO: Similar input validation and error handling for plot_confusion_matrix.
    """
    Plots the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        ax: Matplotlib Axes object to plot on. If None, creates a new one.
        class_names: Names of the classes for display. Default: ['Class 0', 'Class 1']
        model_name: Name of the model for the plot title.
        **kwargs: Additional arguments passed to ConfusionMatrixDisplay.from_predictions.

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    
    if class_names is None:
        class_names = ['No Sig. Move', 'Sig. Move'] # Default class names for this project context

    try:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, display_labels=class_names, cmap=plt.cm.Blues, **kwargs)
        title = "Confusion Matrix"
        if model_name:
            title = f"{model_name} - {title}"
        ax.set_title(title)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not plot CM: {e}", ha='center', va='center')
        print(f"Error plotting CM for {model_name}: {e}")
        
    return ax

if __name__ == '__main__':
    # Example Usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Test plot_roc_auc
    fig_roc, ax_roc_test = plt.subplots()
    plot_roc_auc(y_test, y_pred_proba, ax=ax_roc_test, model_name="Logistic Regression")
    plt.show()
    print(f"AUC for test: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Test plot_confusion_matrix
    fig_cm, ax_cm_test = plt.subplots()
    plot_confusion_matrix(y_test, y_pred, ax=ax_cm_test, model_name="Logistic Regression") # Using default class_names
    plt.show()
    from sklearn.metrics import confusion_matrix as sk_cm
    print(f"CM for test:\n{sk_cm(y_test, y_pred)}")
