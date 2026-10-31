
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
df = pd.read_csv(
    "outputs/results/svm_what_if_results.csv"
)

# Plot accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    df["C Value"],
    df["Accuracy"] * 100,
    marker="o",
    linewidth=2
)

plt.xlabel("SVM C Value")
plt.ylabel("Accuracy (%)")
plt.title("SVM What-If Experiment")
plt.grid(True)

plt.xticks(df["C Value"])

plt.savefig(
    "outputs/graphs/svm_what_if_graph.png"
)

plt.show()

print("Graph saved successfully!")
