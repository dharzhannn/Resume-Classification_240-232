
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# File paths
DATA_PATH = "outputs/results/tfidf_data.pkl"
MODEL_PATH = "outputs/results/models"
OUTPUT_PATH = "outputs/results"
CM_PATH = "outputs/confusion_matrices"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CM_PATH, exist_ok=True)

# Load test data
data = joblib.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

# Models to evaluate
model_files = {
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "SVM": "svm.pkl",
    "Naive Bayes": "naive_bayes.pkl"
}

results = []

print("=" * 60)
print("          RESUME CLASSIFICATION MODEL EVALUATION")
print("=" * 60)

for name, filename in model_files.items():

    model_file = os.path.join(MODEL_PATH, filename)

    if not os.path.exists(model_file):
        print(f"\nModel not found: {filename}")
        continue

    model = joblib.load(model_file)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    graph_name = name.lower().replace(" ", "_")
    plt.savefig(
        os.path.join(CM_PATH, f"{graph_name}.png")
    )
    plt.close()

# Save comparison results
if results:
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(OUTPUT_PATH, "evaluation_results.csv"),
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nEvaluation results saved successfully!")
else:
    print("\nNo models were evaluated. Check your model files.")
