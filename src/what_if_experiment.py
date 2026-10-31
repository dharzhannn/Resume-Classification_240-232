
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load TF-IDF data
data = joblib.load("outputs/results/tfidf_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Test different C values
c_values = [0.1, 1, 10]

results = []

print("=" * 55)
print("SVM WHAT-IF EXPERIMENT")
print("=" * 55)

for c in c_values:

    print(f"\nTraining SVM with C = {c}")

    model = LinearSVC(C=c, random_state=42, max_iter=5000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "C Value": c,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

df = pd.DataFrame(results)

df.to_csv(
    "outputs/results/svm_what_if_results.csv",
    index=False
)

print("\nFINAL WHAT-IF COMPARISON")
print(df.to_string(index=False))

print("\nExperiment completed successfully!")
