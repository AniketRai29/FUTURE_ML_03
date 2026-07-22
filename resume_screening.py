# ============================================================
# Resume Screening & Candidate Ranking System
# Future Interns - Machine Learning Task 2
# Part 1 - Project Setup & Exploratory Data Analysis
# ============================================================

# =========================
# Import Libraries
# =========================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import re
import string

import nltk
import spacy

from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
# =========================
# Download Required NLTK Data
# =========================

nltk.download("stopwords")
# =========================
# Load spaCy Model
# =========================

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner", "textcat"]
)
# =========================
# English Stopwords
# =========================

stop_words = set(stopwords.words("english"))
# ======================================================
# Resume Cleaning Function
# ======================================================

# ==========================================================
# Fast Resume Cleaning Function
# ==========================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"\+?\d[\d\s()-]{8,}", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()
# =========================
# Project Paths
# =========================

DATA_PATH = "data/Resume.csv"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# Project Heading
# =========================

print("=" * 70)
print("Resume Screening & Candidate Ranking System")
print("=" * 70)

print("\nLoading Dataset...")

# =========================
# Load Dataset
# =========================

resume_df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully.")

# =========================
# Basic Dataset Information
# =========================

print("\nDataset Shape")
print(resume_df.shape)

print("\nFirst Five Rows")
print(resume_df.head())

print("\nDataset Information")
print(resume_df.info())

print("\nColumn Names")
print(resume_df.columns.tolist())

# =========================
# Missing Values
# =========================

print("\nMissing Values")
print(resume_df.isnull().sum())

# =========================
# Duplicate Records
# =========================

print("\nDuplicate Rows :", resume_df.duplicated().sum())

resume_df = resume_df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")
print(resume_df.shape)

# =========================
# Data Types
# =========================

print("\nData Types")

print(resume_df.dtypes)

# =========================
# Resume Categories
# =========================

print("\nResume Categories")

print(resume_df["Category"].value_counts())

# =========================
# Resume Length
# =========================

resume_df["Resume_Length"] = resume_df["Resume_str"].astype(str).apply(len)

print("\nResume Length Statistics")

print(resume_df["Resume_Length"].describe())

# =========================
# Average Resume Length
# =========================

print("\nAverage Resume Length")

print(round(resume_df["Resume_Length"].mean(), 2))

# =========================
# Category Distribution Plot
# =========================

plt.figure(figsize=(14,8))

resume_df["Category"].value_counts().sort_values().plot(
    kind="barh"
)

plt.title("Number of Resumes per Category")

plt.xlabel("Count")

plt.ylabel("Category")

plt.tight_layout()

plt.show()

# =========================
# Resume Length Distribution
# =========================

plt.figure(figsize=(10,6))

plt.hist(
    resume_df["Resume_Length"],
    bins=30
)

plt.title("Resume Length Distribution")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

# =========================
# Resume Length Boxplot
# =========================

plt.figure(figsize=(6,5))

plt.boxplot(resume_df["Resume_Length"])

plt.title("Resume Length Boxplot")

plt.ylabel("Characters")

plt.tight_layout()

plt.show()

# =========================
# Top 10 Categories
# =========================

top_categories = resume_df["Category"].value_counts().head(10)

plt.figure(figsize=(12,6))

plt.bar(
    top_categories.index,
    top_categories.values
)

plt.title("Top 10 Resume Categories")

plt.xticks(rotation=45)

plt.ylabel("Count")

plt.tight_layout()

plt.show()

# =========================
# Save Clean Dataset
# =========================

resume_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cleaned_resume_dataset.csv"
    ),
    index=False
)

print("\nCleaned dataset saved successfully.")

print("\nPart 1 Completed Successfully!")

# ==========================================================
# PART 2
# NLP PREPROCESSING
# ==========================================================

print("\n" + "=" * 70)
print("PART 2 - NLP PREPROCESSING")
print("=" * 70)

print("\nPreprocessing Resume Text...")

resume_df["Processed_Text"] = (
    resume_df["Resume_str"]
    .astype(str)
    .apply(preprocess_text)
)

print("Basic Cleaning Completed.")

print("\nRunning spaCy Lemmatization...")

cleaned_resumes = []

for doc in nlp.pipe(
    resume_df["Processed_Text"],
    batch_size=32
):

    words = []

    for token in doc:

        if (
            token.is_stop == False
            and token.is_alpha
            and len(token.text) > 2
        ):

            words.append(token.lemma_.lower())

    cleaned_resumes.append(
        " ".join(words)
    )

resume_df["Clean_Resume"] = cleaned_resumes

print("Lemmatization Completed.")

resume_df["Clean_Length"] = (
    resume_df["Clean_Resume"]
    .str.len()
)

print("\nClean Resume Statistics")

print(
    resume_df["Clean_Length"].describe()
)

plt.figure(figsize=(10,6))

plt.hist(
    resume_df["Clean_Length"],
    bins=30
)

plt.title("Clean Resume Length")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

print("\nGenerating Word Cloud...")

text = " ".join(
    resume_df["Clean_Resume"]
)

wordcloud = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    max_words=250
).generate(text)

plt.figure(figsize=(16,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Most Frequent Resume Words")

plt.show()

from collections import Counter

words = text.split()

counter = Counter(words)

top_words = pd.DataFrame(
    counter.most_common(20),
    columns=["Word","Frequency"]
)

print("\nTop 20 Words\n")

print(top_words)

plt.figure(figsize=(12,6))

plt.bar(
    top_words["Word"],
    top_words["Frequency"]
)

plt.xticks(rotation=45)

plt.title("Top 20 Resume Words")

plt.tight_layout()

plt.show()

resume_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "preprocessed_resume_dataset.csv"
    ),

    index=False

)

print("\nPreprocessed Dataset Saved Successfully.")

print("\n" + "="*70)
print("Part 2 Completed Successfully!")
print("="*70)

# ==========================================================
# PART 3
# ATS SKILL EXTRACTION
# ==========================================================

print("\n" + "="*70)
print("PART 3 - ATS SKILL EXTRACTION")
print("="*70)

with open(
    "job_description.txt",
    "r",
    encoding="utf-8"
) as file:

    JOB_DESCRIPTION = file.read().lower()

SKILL_DICT = {

    "python": ["python"],

    "java": ["java"],

    "c++": ["c++","cpp"],

    "sql": ["sql","mysql","postgresql"],

    "javascript": ["javascript","js"],

    "react": ["react","reactjs"],

    "node.js": ["node","nodejs"],

    "html": ["html"],

    "css": ["css"],

    "git": ["git"],

    "github": ["github"],

    "pandas": ["pandas"],

    "numpy": ["numpy"],

    "scikit-learn": [
        "scikit",
        "scikit learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow",
        "tf"
    ],

    "pytorch": [
        "pytorch",
        "torch"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "dl"
    ],

    "data analysis": [
        "data analysis",
        "data analytics"
    ],

    "problem solving": [
        "problem solving"
    ],

    "communication": [
        "communication"
    ],

    "teamwork": [
        "teamwork",
        "team work"
    ]
}

SKILL_WEIGHT = {

    "python":10,

    "java":8,

    "c++":8,

    "sql":8,

    "javascript":7,

    "react":7,

    "node.js":7,

    "html":5,

    "css":5,

    "git":5,

    "github":5,

    "pandas":8,

    "numpy":8,

    "scikit-learn":10,

    "tensorflow":10,

    "pytorch":10,

    "machine learning":12,

    "deep learning":12,

    "data analysis":8,

    "problem solving":5,

    "communication":5,

    "teamwork":4
}

import re

def extract_skills(text):

    text = text.lower()

    found = []

    score = 0

    for skill, aliases in SKILL_DICT.items():

        for alias in aliases:

            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, text):

                found.append(skill)

                score += SKILL_WEIGHT[skill]

                break

    return found, score

print("\nExtracting Skills...")

resume_df["Matched_Skills"] = ""

resume_df["Skill_Score"] = 0

for index, row in resume_df.iterrows():

    skills, score = extract_skills(

        row["Clean_Resume"]

    )

    resume_df.at[index,"Matched_Skills"] = ", ".join(skills)

    resume_df.at[index,"Skill_Score"] = score

print("Skill Extraction Completed.")

required_skills = list(SKILL_DICT.keys())

missing = []

for skills in resume_df["Matched_Skills"]:

    present = skills.split(", ")

    missing_skills = [

        skill

        for skill in required_skills

        if skill not in present

    ]

    missing.append(", ".join(missing_skills))

resume_df["Missing_Skills"] = missing

resume_df["Total_Skills"] = (

    resume_df["Matched_Skills"]

    .apply(

        lambda x:

        len(

            [i for i in x.split(", ") if i]

        )

    )

)

print("\nAverage Skill Score")

print(

    round(

        resume_df["Skill_Score"].mean(),

        2

    )

)

print("\nHighest Skill Score")

print(

    resume_df["Skill_Score"].max()

)

print("\nLowest Skill Score")

print(

    resume_df["Skill_Score"].min()

)

plt.figure(figsize=(10,6))

plt.hist(

    resume_df["Skill_Score"],

    bins=20

)

plt.title("Skill Score Distribution")

plt.xlabel("Skill Score")

plt.ylabel("Candidates")

plt.tight_layout()

plt.show()

all_skills = []

for skills in resume_df["Matched_Skills"]:

    all_skills.extend(

        [

            i

            for i in skills.split(", ")

            if i

        ]

    )

skill_frequency = (

    pd.Series(all_skills)

    .value_counts()

)

print("\nMost Common Skills\n")

print(skill_frequency)

plt.figure(figsize=(12,6))

skill_frequency.head(15).plot(

    kind="bar"

)

plt.title("Top 15 Skills")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

resume_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "skill_extracted_dataset.csv"

    ),

    index=False

)

print("\nSkill Dataset Saved Successfully.")

print("\n" + "="*70)
print("Part 3 Completed Successfully!")
print("="*70)

# ==========================================================
# PART 4
# ATS CANDIDATE RANKING
# ==========================================================

print("\n" + "="*70)
print("PART 4 - ATS CANDIDATE RANKING")
print("="*70)

print("\nPreparing Documents...")

documents = [JOB_DESCRIPTION]

documents.extend(
    resume_df["Clean_Resume"].tolist()
)

print("Creating TF-IDF Matrix...")

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    documents
)

print("TF-IDF Matrix Created.")

print("\nCalculating Cosine Similarity...")

job_vector = tfidf_matrix[0]

resume_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(
    job_vector,
    resume_vectors
)

resume_df["Similarity"] = similarity_scores.flatten()

print("Similarity Calculation Completed.")

resume_df["Similarity_Percentage"] = (
    resume_df["Similarity"] * 100
).round(2)

max_skill = resume_df["Skill_Score"].max()

resume_df["Normalized_Skill"] = (
    resume_df["Skill_Score"] / max_skill
) * 100

resume_df["ATS_Score"] = (

    0.70 * resume_df["Similarity_Percentage"]

    +

    0.30 * resume_df["Normalized_Skill"]

)

resume_df["ATS_Score"] = (

    resume_df["ATS_Score"]

    .round(2)

)

resume_df = resume_df.sort_values(

    by="ATS_Score",

    ascending=False

).reset_index(drop=True)

resume_df["Rank"] = resume_df.index + 1

q75 = resume_df["ATS_Score"].quantile(0.75)
q50 = resume_df["ATS_Score"].quantile(0.50)
q25 = resume_df["ATS_Score"].quantile(0.25)

def recommendation(score):
    if score >= q75:
        return "Highly Recommended"
    elif score >= q50:
        return "Recommended"
    elif score >= q25:
        return "Consider"
    else:
        return "Not Recommended"

resume_df["Recommendation"] = (

    resume_df["ATS_Score"]

    .apply(recommendation)

)

print("\nTop 10 Candidates\n")

print(

    resume_df[

        [

            "Rank",

            "Category",

            "ATS_Score",

            "Similarity_Percentage",

            "Skill_Score",

            "Recommendation"

        ]

    ].head(10)

)

print("\nATS Score Statistics\n")

print(

    resume_df["ATS_Score"]

    .describe()

)

plt.figure(figsize=(10,6))

plt.hist(

    resume_df["ATS_Score"],

    bins=20

)

plt.title("ATS Score Distribution")

plt.xlabel("ATS Score")

plt.ylabel("Candidates")

plt.tight_layout()

plt.show()

plt.figure(figsize=(12,6))

top15 = resume_df.head(15)

plt.bar(

    top15["Rank"].astype(str),

    top15["ATS_Score"]

)

plt.title("Top 15 Ranked Candidates")

plt.xlabel("Rank")

plt.ylabel("ATS Score")

plt.tight_layout()

plt.show()

recommendation_counts = (

    resume_df["Recommendation"]

    .value_counts()

)

print("\nRecommendation Summary\n")

print(recommendation_counts)

plt.figure(figsize=(7,7))

plt.pie(

    recommendation_counts,

    labels=recommendation_counts.index,

    autopct="%1.1f%%"

)

plt.title("Recommendation Distribution")

plt.show()

resume_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "ranked_candidates.csv"

    ),

    index=False

)

print("\nRanked Candidate Dataset Saved Successfully.")

print("\n" + "="*70)
print("Part 4 Completed Successfully!")
print("="*70)