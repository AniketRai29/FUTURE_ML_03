import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ATS Resume Screening Dashboard",
    page_icon="📄",
    layout="wide"
)

st.title("📄 ATS Resume Screening Dashboard")

st.markdown(
    "### AI-Based Resume Screening & Candidate Ranking System"
)

df = pd.read_csv("outputs/ranked_candidates.csv")

st.sidebar.title("Dashboard")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

if category != "All":
    filtered_df = df[df["Category"] == category]
else:
    filtered_df = df.copy()

total_candidates = len(filtered_df)

average_score = round(
    filtered_df["ATS_Score"].mean(),
    2
)

highest_score = round(
    filtered_df["ATS_Score"].max(),
    2)

recommended = len(
    filtered_df[
        filtered_df["Recommendation"]
        == "Highly Recommended"
    ]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Candidates",
    total_candidates
)

col2.metric(
    "Average ATS",
    average_score
)

col3.metric(
    "Highest Score",
    highest_score
)

col4.metric(
    "Highly Recommended",
    recommended
)

st.subheader("🏆 Top Ranked Candidates")

st.dataframe(

    filtered_df[

        [

            "Rank",

            "Category",

            "ATS_Score",

            "Skill_Score",

            "Similarity_Percentage",

            "Recommendation"

        ]

    ].head(20),

    use_container_width=True

)

st.subheader("ATS Score Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    filtered_df["ATS_Score"],
    bins=20
)

ax.set_xlabel("ATS Score")

ax.set_ylabel("Candidates")

st.pyplot(fig)

st.subheader("Recommendation Distribution")

counts = (
    filtered_df["Recommendation"]
    .value_counts()
)

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%"
)

st.pyplot(fig)

st.subheader("Category Distribution")

fig, ax = plt.subplots(figsize=(10,5))

filtered_df["Category"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

st.subheader("Top ATS Scores")

top10 = filtered_df.head(10)

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(

    top10["Rank"].astype(str),

    top10["ATS_Score"]

)

ax.set_xlabel("Rank")

ax.set_ylabel("ATS Score")

st.pyplot(fig)

st.subheader("Search Candidate")

rank = st.number_input(

    "Enter Candidate Rank",

    min_value=1,

    max_value=len(filtered_df),

    value=1

)

candidate = filtered_df[
    filtered_df["Rank"] == rank
]

if not candidate.empty:

    st.write(candidate.T)

csv = filtered_df.to_csv(index=False)

st.download_button(

    "Download Ranked Candidates",

    csv,

    file_name="ranked_candidates.csv",

    mime="text/csv"

)

st.markdown("---")

st.markdown(
    "Developed using Python, NLP, TF-IDF, Machine Learning & Streamlit"
)