
import os

# Main dataset folder
DATASET_PATH = "dataset"

print("=" * 50)
print("       RESUME DATASET ANALYSIS")
print("=" * 50)

total_resumes = 0
categories = {}

# Read all category folders
for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):
        files = os.listdir(category_path)

        # Count resume files
        resume_files = [
            file for file in files
            if file.lower().endswith((".txt", ".pdf", ".docx"))
        ]

        categories[category] = len(resume_files)
        total_resumes += len(resume_files)

# Display results
print("\nTotal Resume Categories:", len(categories))
print("Total Resume Files:", total_resumes)

print("\nResume Count by Category:")
print("-" * 40)

for category, count in sorted(categories.items()):
    print(f"{category}: {count}")

print("\nDataset Analysis Completed!")

# Check resume file formats

print("\nRESUME FILE FORMAT ANALYSIS")
print("-" * 40)

file_formats = {}

for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            extension = os.path.splitext(file)[1].lower()

            file_formats[extension] = (
                file_formats.get(extension, 0) + 1
            )

for extension, count in sorted(file_formats.items()):
    print(f"{extension}: {count}")
