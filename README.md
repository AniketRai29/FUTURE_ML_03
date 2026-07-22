# ATS Resume Screening & Candidate Ranking System

## 📌 Overview

This project is an AI-powered Resume Screening System that automatically analyzes resumes, extracts technical skills, compares them with a job description, and ranks candidates using an ATS (Applicant Tracking System) score.

It helps recruiters shortlist the best candidates quickly and efficiently.

---

## 🚀 Features

- Resume preprocessing using NLP
- Skill extraction
- Job description matching
- TF-IDF vectorization
- Cosine similarity
- ATS score calculation
- Candidate ranking
- HR Analytics Dashboard using Streamlit
- CSV export of ranked candidates

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- spaCy
- NLTK
- Scikit-learn
- Matplotlib
- Streamlit

---
The dataset is not included in this repository because of its size.

Download the resume dataset and place `Resume.csv` inside the `data/` folder before running the project.

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
## 📂 Project Structure

```
Resume-Screening-ML/
│
├── data/
│   └── Resume.csv
│
├── outputs/
│   ├── cleaned_resume_dataset.csv
│   ├── preprocessed_resume_dataset.csv
│   ├── skill_extracted_dataset.csv
│   └── ranked_candidates.csv
│
├── dashboard.py
├── resume_screening.py
├── job_description.txt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙ Installation

```bash
git clone <repository-url>

cd Resume-Screening-ML

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

---

## ▶ Run Resume Processing

```bash
python resume_screening.py
```

---

## ▶ Run Dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Outputs

- Cleaned Resume Dataset
- Preprocessed Resume Dataset
- Skill Extracted Dataset
- Ranked Candidates
- ATS Dashboard

---

## 👨‍💻 Author

Aniket Rai
