# ============================================================
# 📄 Resume Screening App — Professional Edition (with SQLite)
# ============================================================

import streamlit as st

# Must be the FIRST Streamlit command
st.set_page_config(page_title="Resume Screening App (Pro)", layout="wide", page_icon="📄")

# -----------------------------
# Imports
# -----------------------------
import os, re, io, sqlite3, base64
from datetime import datetime
import pandas as pd
import numpy as np
import PyPDF2, docx
from fpdf import FPDF
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Setup data directory and database
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, "resume_screening.db")

# -----------------------------
# Initialize database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Job descriptions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            job_title TEXT,
            job_description TEXT
        )
    """)

    # Resume results table
    c.execute("""
        CREATE TABLE IF NOT EXISTS resume_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            resume_name TEXT,
            match_percent REAL,
            found_skills TEXT,
            missing_skills TEXT,
            experience_yrs INTEGER,
            education TEXT,
            FOREIGN KEY (job_id) REFERENCES job_history (id)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Utility: Save & Load DB
# -----------------------------
def save_analysis(job_text, results_df):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute(
        "INSERT INTO job_history (timestamp, job_title, job_description) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), job_text[:80], job_text[:4000])
    )
    job_id = c.lastrowid

    for _, row in results_df.iterrows():
        c.execute("""
            INSERT INTO resume_results 
            (job_id, resume_name, match_percent, found_skills, missing_skills, experience_yrs, education)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            row["Resume"],
            row["Match %"],
            row["Found Skills"],
            row["Missing Skills"],
            row["Experience (yrs)"],
            row["Education"]
        ))

    conn.commit()
    conn.close()

def load_past_jobs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM job_history ORDER BY id DESC", conn)
    conn.close()
    return df

def load_results_for_job(job_id):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT * FROM resume_results WHERE job_id={job_id}", conn)
    conn.close()
    return df

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Text extraction functions
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# -----------------------------
# Skill extraction helpers
# -----------------------------
skills_master = [
    "python","sql","machine learning","deep learning","data analysis","pandas",
    "numpy","spark","hadoop","docker","kubernetes","aws","azure","gcp",
    "nlp","tensorflow","pytorch","scikit-learn","excel","power bi","tableau",
    "javascript","react","node","django","flask","communication","leadership"
]

def extract_skills(text):
    found = []
    lower = text.lower()
    for skill in skills_master:
        if re.search(r'\b' + re.escape(skill) + r'\b', lower):
            found.append(skill)
    return sorted(set(found))

def extract_years_experience(text):
    matches = re.findall(r'(\d{1,2})\s*(?:years|yrs|year)', text.lower())
    return int(matches[0]) if matches else 0

def extract_education(text):
    lower = text.lower()
    if "phd" in lower: return "PhD"
    if "master" in lower or "msc" in lower: return "Master"
    if "bachelor" in lower or "bsc" in lower: return "Bachelor"
    return "Not mentioned"

# -----------------------------
# PDF Report Generator
# -----------------------------
def create_pdf(df, job_text):
    def safe_text(t):
        return t.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Resume Screening Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, safe_text(f"Job Description (truncated): {job_text[:800]}"))
    pdf.ln(6)
    pdf.cell(0, 6, txt=f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
    pdf.ln(8)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(70, 6, "Resume", border=1)
    pdf.cell(25, 6, "Match %", border=1)
    pdf.cell(90, 6, "Missing Skills", border=1)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for _, r in df.iterrows():
        pdf.cell(70, 6, safe_text(str(r["Resume"])[:40]), border=1)
        pdf.cell(25, 6, safe_text(str(r["Match %"])), border=1)
        pdf.cell(90, 6, safe_text(str(r["Missing Skills"])[:60]), border=1)
        pdf.ln()
    return pdf.output(dest="S").encode("latin-1", "replace")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Resume Screening App — Professional Edition")
st.markdown("Automate resume shortlisting with AI-based similarity scoring and skill gap visualization.")

input_col, preview_col = st.columns([2, 1])

with input_col:
    st.subheader("Job Input")
    mode = st.radio("Provide Job Description via:", ["Paste Text", "Job Posting URL"])
    job_text = ""

    if mode == "Paste Text":
        job_text = st.text_area("Paste Job Description", height=200)
    else:
        job_url = st.text_input("Paste Job Posting URL:")
        if st.button("Fetch Description"):
            import requests
            from bs4 import BeautifulSoup
            if job_url.strip():
                with st.spinner("Fetching job details..."):
                    try:
                        r = requests.get(job_url, timeout=8)
                        soup = BeautifulSoup(r.text, "html.parser")
                        paras = soup.find_all(["p", "li", "div"])
                        job_text = " ".join([p.get_text(" ", strip=True) for p in paras])[:5000]
                        st.success("Job description fetched successfully!")
                        st.text_area("Fetched Preview", job_text[:2000], height=150)
                    except Exception:
                        st.error("Failed to fetch job posting.")
            else:
                st.warning("Please enter a valid URL.")

    uploaded = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

with preview_col:
    st.subheader("Skill Reference")
    st.write(", ".join(skills_master[:15]) + ", ...")

# -----------------------------
# Analysis Button
# -----------------------------
if st.button("🔍 Analyze Resumes"):
    if not job_text.strip():
        st.warning("Please provide a Job Description first.")
    elif not uploaded:
        st.warning("Please upload at least one resume file.")
    else:
        with st.spinner("Analyzing resumes..."):
            jd_clean = clean_text(job_text)
            jd_embed = model.encode(jd_clean, convert_to_numpy=True)
            jd_skills = extract_skills(jd_clean)
            rows = []

            for f in uploaded:
                fname = f.name
                ext = fname.split(".")[-1].lower()
                fb = io.BytesIO(f.read())
                text = extract_text_from_pdf(fb) if ext == "pdf" else extract_text_from_docx(fb)
                text = clean_text(text)

                emb = model.encode(text, convert_to_numpy=True)
                similarity = float(util.cos_sim(jd_embed, emb))
                found = extract_skills(text)
                missing = [s for s in jd_skills if s not in found]
                exp = extract_years_experience(text)
                edu = extract_education(text)

                rows.append({
                    "Resume": fname,
                    "Match %": round(similarity * 100, 2),
                    "Found Skills": ", ".join(found) or "None",
                    "Missing Skills": ", ".join(missing) or "None",
                    "Experience (yrs)": exp,
                    "Education": edu
                })

            df = pd.DataFrame(rows).sort_values("Match %", ascending=False).reset_index(drop=True)
            save_analysis(job_text, df)

        st.success("✅ Analysis complete and saved to database!")

        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Candidate Details", "Skill Gaps", "History"])

        with tab1:
            st.subheader("📊 Overview")
            st.metric("Total Candidates", len(df))
            st.metric("Average Match %", f"{df['Match %'].mean():.2f}")
            st.metric("Top Match", f"{df['Match %'].max():.2f}")
            st.plotly_chart(px.bar(df, x="Resume", y="Match %", title="Match Percentage per Resume"), use_container_width=True)

        with tab2:
            st.subheader("📋 Candidate Details")
            st.dataframe(df, use_container_width=True)

        with tab3:
            st.subheader("🧠 Skill Insights")
            found_all = [x.strip() for sub in df["Found Skills"] for x in sub.split(",")]
            miss_all = [x.strip() for sub in df["Missing Skills"] for x in sub.split(",")]

            found_freq = pd.Series(found_all).value_counts().reset_index()
            found_freq.columns = ["Skill", "Count"]
            st.plotly_chart(px.bar(found_freq.head(15), x="Skill", y="Count", title="Top Found Skills"), use_container_width=True)

            miss_freq = pd.Series(miss_all).value_counts().reset_index()
            miss_freq.columns = ["Skill", "Count"]
            st.plotly_chart(px.treemap(miss_freq.head(15), path=["Skill"], values="Count", title="Top Missing Skills"), use_container_width=True)

        with tab4:
            st.subheader("📁 Previous Analyses")
            jobs = load_past_jobs()
            if jobs.empty:
                st.info("No past analyses yet.")
            else:
                job_choice = st.selectbox("Select past job:", jobs["job_title"].tolist())
                job_id = jobs[jobs["job_title"] == job_choice]["id"].iloc[0]
                results = load_results_for_job(job_id)
                st.dataframe(results.drop(columns=["id", "job_id"]), use_container_width=True)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "resume_results.csv", "text/csv")
        st.download_button("📄 Download PDF", create_pdf(df, job_text), "resume_report.pdf", "application/pdf")
