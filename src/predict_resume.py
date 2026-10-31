
import os
import re
import fitz
import joblib

# File paths
MODEL_PATH = "outputs/results/models/svm.pkl"
TFIDF_PATH = "outputs/results/tfidf_data.pkl"


def clean_text(text):
    """Clean resume text exactly like preprocessing.py."""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\b\d{10,}\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text from uploaded PDF."""

    try:
        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text() + " "

        document.close()

        return clean_text(text)

    except Exception as error:
        print("PDF extraction error:", error)
        return ""


def predict_resume(pdf_path):

    # Check PDF
    if not os.path.exists(pdf_path):
        print("Error: Resume file not found!")
        return

    # Load trained SVM model
    if not os.path.exists(MODEL_PATH):
        print("SVM model not found. Run train_models.py first.")
        return

    model = joblib.load(MODEL_PATH)

    # Load saved TF-IDF vectorizer
    if not os.path.exists(TFIDF_PATH):
        print("TF-IDF data not found. Run feature_extraction.py first.")
        return

    data = joblib.load(TFIDF_PATH)
    vectorizer = data["vectorizer"]

    # Extract and clean resume text
    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text:
        print("No readable text found in this PDF.")
        return

    # Convert text into TF-IDF features
    resume_vector = vectorizer.transform([resume_text])

    # Predict category
    prediction = model.predict(resume_vector)[0]

    print("\n" + "=" * 45)
    print("       RESUME CLASSIFICATION RESULT")
    print("=" * 45)

    print("\nResume:", os.path.basename(pdf_path))
    print("Predicted Job Category:", prediction)

    print("\nPrediction completed successfully!")


if __name__ == "__main__":

    print("=" * 45)
    print("       RESUME CLASSIFICATION SYSTEM")
    print("=" * 45)

    pdf_path = input("\nEnter PDF resume path: ").strip().strip('"')

    predict_resume(pdf_path)
