
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

# Load TF-IDF features
DATA_PATH = "outputs/results/tfidf_data.pkl"

data = joblib.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

# Define four classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    ),

    "SVM": LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=5000
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.5
    )
}

# Create model directory
os.makedirs("outputs/results/models", exist_ok=True)

print("=" * 50)
print("       RESUME CLASSIFICATION MODEL TRAINING")
print("=" * 50)

results = {}

# Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        f"outputs/results/models/{filename}"
    )

    print(f"{name} Accuracy: {accuracy:.4f}")

print("\n" + "=" * 50)
print("       INITIAL MODEL COMPARISON")
print("=" * 50)

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")

print("\nAll four models trained successfully!")
