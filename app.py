
import streamlit as st
import os
import sys
import tempfile
import joblib
import fitz

from src.predict_resume import predict_resume

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Classification System")
st.write("Upload a resume PDF to predict its job category.")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Classify Resume"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            result = predict_resume(temp_path)
            st.success("Resume classified successfully!")
            st.subheader(f"Predicted Category: {result}")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.remove(temp_path)
