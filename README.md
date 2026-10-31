# Resume Classification Using Machine Learning

## 1. Project Overview

Resume Classification is a machine learning-based text classification project that automatically classifies resumes into different job categories based on their textual content.

The system uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyse resume text and predict the most suitable category.

The main objective is to compare different machine learning algorithms and identify the best-performing model for resume classification.

## 2. Dataset Description

- **Dataset:** Resume Dataset – Kaggle
- **Dataset Source:** [Resume Dataset – Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Total Resumes:** 1,714 PDF files
- **Number of Categories:** 18
- **File Format:** PDF
- **Task Type:** Multiclass Text Classification

The dataset contains resumes belonging to different professional categories, including:

ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE and FITNESS.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyMuPDF
- Matplotlib
- Streamlit
- TF-IDF Vectorization

## 4. Project Workflow

1. **Dataset Collection:** Collect PDF resumes organised into different job categories.
2. **Text Extraction:** Extract textual content from PDF files using PyMuPDF.
3. **Text Preprocessing:** Clean and prepare extracted resume text for classification.
4. **Feature Extraction:** Convert text into numerical features using TF-IDF.
5. **Model Training:** Train four machine learning algorithms using the same processed dataset.
6. **Model Evaluation:** Compare the models using Accuracy, Precision, Recall and F1-score.
7. **What-If Experiment:** Change the SVM parameter C to study its effect on classification performance.
8. **Prediction:** Predict the job category of a new resume.

## 5. Machine Learning Algorithms

The following four algorithms are implemented:

- **Decision Tree:** Classifies resumes using decision rules.
- **K-Nearest Neighbors (KNN):** Predicts the category based on similar training resumes.
- **Support Vector Machine (SVM):** Finds decision boundaries between different resume categories.
- **Naive Bayes:** Uses probability to predict the most likely resume category.

## 6. Model Evaluation Results

The models were evaluated using the same train-test split and TF-IDF features.

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Decision Tree | 65.79% | 66.88% | 65.79% | 65.71% |
| KNN | 54.68% | 57.97% | 54.68% | 54.54% |
| SVM | 72.81% | 73.73% | 72.81% | 72.31% |
| Naive Bayes | 58.48% | 61.02% | 58.48% | 56.20% |

**Best Performing Model: SVM**

SVM achieved the highest accuracy of 72.81% among the four evaluated algorithms.

## 7. What-If Experiment

A what-if experiment was performed by changing the SVM regularization parameter C.

| SVM C Value | Accuracy |
|---|---:|
| 0.1 | 68.13% |
| 1 | 72.81% |
| 10 | 72.22% |

The experiment shows that changing the C parameter affects classification performance. Among the tested values, C = 1 achieved the highest accuracy.

## 8. Visualizations and Outputs

The project includes:

- Model comparison graph
- SVM what-if experiment graph
- Confusion matrices
- Classification reports
- Model evaluation results
- Saved trained machine learning models

These outputs help analyse and compare the performance of the algorithms.

## 9. Project Structure

```text
Resume-Classification/
│
├── dataset/
│   ├── ACCOUNTANT/
│   ├── ADVOCATE/
│   ├── AGRICULTURE/
│   └── ... other categories
│
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── model_comparison_graph.py
│   ├── what_if_experiment.py
│   ├── plot_experiment.py
│   └── predict_resume.py
│
├── outputs/
│   ├── confusion_matrices/
│   ├── graphs/
│   └── results/
│
├── app.py
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

**Step 1: Clone the repository**

```bash
git clone https://github.com/dharzhannn/Resume-Classification_240-232.git
```

**Step 2: Open the project folder**

```bash
cd Resume-Classification_240-232
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Streamlit application**

```bash
streamlit run app.py
```

## 11. Conclusion

This project demonstrates how Natural Language Processing and machine learning algorithms can be used to classify resumes into different professional categories.

Four machine learning algorithms were implemented and evaluated using common performance metrics. Based on the experimental results, SVM achieved the highest accuracy of 72.81%.

The project also demonstrates how changing model parameters can influence classification performance through a what-if experiment.

---

**Project:** Resume Classification Using Machine Learning  
**Course:** Information Retrieval (21CSE377T)  
**Task:** Task 2: Text Classification Battle – Compare, Analyze and Defend
