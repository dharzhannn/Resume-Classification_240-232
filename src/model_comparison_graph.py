
import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
df = pd.read_csv("outputs/results/model_comparison.csv")

# Plot model accuracy
plt.figure(figsize=(10, 6))

plt.bar(
    df["Algorithm"],
    df["Accuracy"] * 100
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.title("Resume Classification - Model Comparison")

plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save graph
plt.savefig("outputs/graphs/model_comparison.png")

plt.show()

print("Model comparison graph saved successfully!")
