
import os
import re
import fitz
import pandas as pd

DATASET_PATH = "dataset"
OUTPUT_PATH = "outputs/results/resume_extracted.csv"


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF resume."""

    try:
        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text() + " "

        document.close()

        return text.strip()

    except Exception as error:
        print(f"Error reading {pdf_path}: {error}")
        return ""


def clean_text(text):
    """Clean extracted resume text."""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )

    # Remove phone numbers
    text = re.sub(
        r"\b\d{10,}\b",
        " ",
        text
    )

    # Remove unwanted special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_resumes():

    resume_data = []
    failed_files = 0

    for category in os.listdir(DATASET_PATH):

        category_path = os.path.join(
            DATASET_PATH, category
        )

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):

            if filename.lower().endswith(".pdf"):

                pdf_path = os.path.join(
                    category_path, filename
                )

                raw_text = extract_text_from_pdf(pdf_path)

                if not raw_text:
                    failed_files += 1
                    continue

                cleaned_text = clean_text(raw_text)

                if not cleaned_text:
                    failed_files += 1
                    continue

                resume_data.append({
                    "resume_text": cleaned_text,
                    "category": category
                })

    df = pd.DataFrame(resume_data)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nRESUME PREPROCESSING COMPLETED")
    print("-" * 40)

    print("Successfully extracted:", len(df))
    print("Failed or empty files:", failed_files)
    print("Total categories:", df["category"].nunique())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nOutput saved to:", OUTPUT_PATH)

    return df


if __name__ == "__main__":
    process_resumes()
