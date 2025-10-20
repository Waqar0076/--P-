# ============================================================
# Resume Screening App - Industry Version
# ============================================================

import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="Resume Screening App (Pro)", layout="wide", page_icon="📄")

# -------------------------
# Imports (after set_page_config)
# -------------------------
import os
import re
import io
import time
import sqlite3
import base64
from datetime import datetime

import pandas as pd
import numpy as np
import PyPDF2
import docx
import plotly.express as px
from fpdf import FPDF

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Constants / Files
# -------------------------
DATA_DIR = "data"
SKILLS_FILE = os.path.join(DATA_DIR, "skills_master.csv")
DB_FILE = os.path.join(DATA_DIR, "analysis_history.db")

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def ensure_skills_file():
    if not os.path.exists(SKILLS_FILE):
        sample_skills = [
            "python","sql","machine learning","deep learning","data analysis","pandas",
            "numpy","spark","hadoop","docker","kubernetes","aws","azure","gcp",
            "nlp","tensorflow","pytorch","scikit-learn","excel","power bi","tableau",
            "javascript","react","node","django","flask","communication","leadership"
        ]
        pd.DataFrame({"skill": sample_skills}).to_csv(SKILLS_FILE, index=False)

ensure_skills_file()
skills_master = pd.read_csv(SKILLS_FILE)["skill"].astype(str).str.lower().tolist()

# -------------------------
# Load embedding model (cached)
# -------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# -------------------------
# DB helpers (simple SQLite)
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            job_text TEXT,
            results_csv BLOB
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_analysis_to_db(job_text, csv_bytes):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO analyses (timestamp, job_text, results_csv) VALUES (?, ?, ?)",
              (datetime.utcnow().isoformat(), job_text, sqlite3.Binary(csv_bytes)))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id, timestamp, job_text FROM analyses ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df

# -------------------------
# Text extraction utilities
# -------------------------
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    # keep punctuation for experience detection, but remove problematic unicode
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# -------------------------
# Skill extraction using master list
# -------------------------
def extract_skills_from_text(text, skills_list):
    found = []
    lower = text.lower()
    for s in skills_list:
        # word boundary to avoid partial matches
        if re.search(r'\b' + re.escape(s) + r'\b', lower):
            found.append(s)
    return sorted(list(set(found)))

# -------------------------
# Experience & Education extraction (heuristic)
# -------------------------
def extract_years_experience(text):
    # look for patterns like "5 years", "5+ years", "5 yrs", "experience: 5"
    matches = re.findall(r'(\d{1,2})\s*\+?\s*(?:years|yrs|year)', text.lower())
    if matches:
        try:
            return int(matches[0])
        except:
            return 0
    # fallback: check for 'experience' with numbers nearby
    matches2 = re.findall(r'experience[^0-9]{0,20}(\d{1,2})', text.lower())
    if matches2:
        return int(matches2[0])
    return 0

def extract_education_level(text):
    lower = text.lower()
    levels = []
    if "phd" in lower or "doctorate" in lower: levels.append("PhD")
    if "master" in lower or "msc" in lower or "m.s." in lower or "m.s" in lower: levels.append("Master")
    if "bachelor" in lower or "bsc" in lower or "b.s." in lower or "b.s" in lower: levels.append("Bachelor")
    if not levels:
        return "Not mentioned"
    return ", ".join(sorted(set(levels), reverse=True))

# -------------------------
# Highlight matched / missing skills in resume text
# -------------------------
def highlight_text_skills(text, present_skills, jd_skills):
    # Show matches in green and missing JD skills as red (we show missing separately)
    html = text
    # limit size to prevent huge output
    display_text = html[:5000]
    # escape HTML special chars
    display_text = display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for skill in sorted(set(present_skills), key=lambda s: -len(s)):
        display_text = re.sub(rf'(?i)\b({re.escape(skill)})\b', r'<mark style="background: #c7f9d9">\1</mark>', display_text)
    return display_text

# -------------------------
# PDF Report generator (FPDF)
# -------------------------
def create_pdf_report(df, job_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt="Resume Screening Report", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt="Job Description (truncated):")
    pdf.multi_cell(0, 6, txt=job_text[:800])
    pdf.ln(4)
    pdf.cell(0, 6, txt=f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
    pdf.ln(6)
    # Table header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(80, 6, "Resume", border=1)
    pdf.cell(30, 6, "Match %", border=1)
    pdf.cell(60, 6, "Missing Skills (truncated)", border=1)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for _, r in df.iterrows():
        name = str(r["Resume"])[:40]
        score = str(r["Match %"])
        missing = str(r["Missing Skills"])[:60]
        pdf.cell(80, 6, name, border=1)
        pdf.cell(30, 6, score, border=1)
        pdf.cell(60, 6, missing, border=1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# -------------------------
# Main App UI
# -------------------------
# Styling
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(90deg, #f3f7ff, #ffffff); }
    .big-font { font-size:18px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Resume Screening App — Professional Edition")
st.markdown("Use semantic matching, skill extraction and candidate analytics to screen resumes like an HR pro.")

# Input tabs
input_col, preview_col = st.columns([2, 1])

with input_col:
    st.header("Job Input")
    job_mode = st.radio("Job input method:", ("Paste JD", "Job posting URL"))
    job_text = ""
    if job_mode == "Paste JD":
        job_text = st.text_area("Paste the job description here:", height=220)
    else:
        job_url = st.text_input("Paste the job posting URL:")
        if st.button("Fetch Job Description"):
            if job_url.strip():
                with st.spinner("Fetching..."):
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        r = requests.get(job_url, timeout=8)
                        soup = BeautifulSoup(r.content, "html.parser")
                        # heuristics: grab main text blocks
                        paras = soup.find_all(["p","li","div"])
                        fetched = " ".join([p.get_text(" ", strip=True) for p in paras])
                        job_text = fetched[:5000]
                        st.success("Fetched job text (preview below).")
                        st.text_area("Fetched Preview:", job_text[:2000], height=200)
                    except Exception as e:
                        st.error("Could not fetch job URL. Paste JD manually.")
            else:
                st.warning("Enter a valid URL.")

    st.header("Upload Resumes")
    uploaded = st.file_uploader("Upload resumes (PDF or DOCX). You can upload multiple files.", type=["pdf", "docx"], accept_multiple_files=True)

    st.header("Advanced Options")
    use_bert = st.checkbox("Use Sentence-BERT embeddings (recommended)", value=True)
    top_k = st.slider("Top K candidates to highlight", 1, min(1, max(1, 5)), 5)

with preview_col:
    st.header("Skills Master")
    st.markdown("You can edit `data/skills_master.csv` to improve detection.")
    sample_skills = ", ".join(skills_master[:12])
    st.write(sample_skills + (", ..." if len(skills_master) > 12 else ""))

# ANALYZE
if st.button("🔍 Analyze"):
    if not job_text or job_text.strip() == "":
        st.warning("Please provide the job description (paste or fetch).")
    elif not uploaded:
        st.warning("Please upload at least one resume file.")
    else:
        with st.spinner("Processing resumes..."):
            jd_clean = clean_text(job_text)
            # JD embedding
            if use_bert:
                jd_embed = model.encode(jd_clean, convert_to_numpy=True)
            else:
                jd_embed = None

            jd_skills = extract_skills_from_text(jd_clean, skills_master)

            rows = []
            # iterate resumes
            for f in uploaded:
                fname = f.name
                ext = fname.split(".")[-1].lower()
                # read bytes into buffer for repeated consumption
                fb = io.BytesIO(f.read())
                fb.seek(0)
                if ext == "pdf":
                    text = extract_text_from_pdf(fb)
                elif ext == "docx":
                    fb.seek(0)
                    text = extract_text_from_docx(fb)
                else:
                    text = ""
                text = clean_text(text)
                # embedding and similarity
                if use_bert:
                    emb = model.encode(text, convert_to_numpy=True)
                    # cosine similarity
                    sim = float(util.cos_sim(jd_embed, emb))
                else:
                    # fallback: basic bag-of-words similarity using simple ratio
                    sim = 0.0
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        vect = TfidfVectorizer(stop_words="english")
                        vs = vect.fit_transform([jd_clean, text])
                        sim = float(cosine_similarity(vs[0:1], vs[1:2])[0][0])
                    except:
                        sim = 0.0

                found_skills = extract_skills_from_text(text, skills_master)
                missing_skills = [s for s in jd_skills if s not in found_skills]
                yrs = extract_years_experience(text)
                edu = extract_education_level(text)

                rows.append({
                    "Resume": fname,
                    "Match %": round(sim * 100, 2),
                    "Found Skills": ", ".join(found_skills) if found_skills else "None",
                    "Missing Skills": ", ".join(missing_skills[:10]) if missing_skills else "None",
                    "Experience (yrs)": yrs,
                    "Education": edu,
                    "Raw Text": text
                })

            results_df = pd.DataFrame(rows).sort_values(by="Match %", ascending=False).reset_index(drop=True)

            # save to DB
            csv_bytes = results_df.drop(columns=["Raw Text"]).to_csv(index=False).encode("utf-8")
            save_analysis_to_db(job_text[:2000], csv_bytes)

        st.success("Analysis complete!")

        # -------------------------
        # Dashboard Tabs
        # -------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Candidates", "Skill Gaps", "History"])

        with tab1:
            st.subheader("Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates", f"{len(results_df)}")
            col2.metric("Average Match %", f"{results_df['Match %'].mean():.2f}")
            col3.metric("Top Match", f"{results_df['Match %'].max():.2f}")

            fig = px.bar(results_df, x="Resume", y="Match %", color="Match %", color_continuous_scale="teal", title="Match Percentage by Resume")
            st.plotly_chart(fig, use_container_width=True)

            # Top candidates pie
            top_df = results_df.head(top_k)
            if not top_df.empty:
                fig2 = px.pie(top_df, names="Resume", values="Match %", title=f"Top {len(top_df)} Candidate Share")
                st.plotly_chart(fig2, use_container_width=True)

            # Histogram
            fig3 = px.histogram(results_df, x="Match %", nbins=10, title="Distribution of Match Scores")
            st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            st.subheader("Candidates (detailed)")
            st.dataframe(results_df.drop(columns=["Raw Text"]), use_container_width=True)

            # Select a candidate to preview
            selected = st.selectbox("Select a candidate to preview highlighted resume text", results_df["Resume"].tolist())
            if selected:
                row = results_df[results_df["Resume"] == selected].iloc[0]
                st.markdown(f"**Match %:** {row['Match %']}  |  **Experience (yrs):** {row['Experience (yrs)']}  |  **Education:** {row['Education']}")
                st.markdown("**Found Skills:** " + row["Found Skills"])
                st.markdown("**Missing Skills:** " + row["Missing Skills"])
                st.markdown("---")
                highlighted = highlight_text_skills(row["Raw Text"], row["Found Skills"].split(", ") if row["Found Skills"]!="None" else [], jd_skills)
                st.markdown(highlighted, unsafe_allow_html=True)

        with tab3:
            st.subheader("Skill Coverage & Gaps")
            # Skill frequency
            all_found = []
            for s in results_df["Found Skills"]:
                if s and s != "None":
                    all_found.extend([x.strip() for x in s.split(",") if x.strip()!=""])
            if all_found:
                freq = pd.Series(all_found).value_counts().reset_index()
                freq.columns = ["Skill", "Count"]
                fig4 = px.bar(freq.head(20), x="Skill", y="Count", title="Top Skills Found Across Candidates")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No skills found in resumes based on skills_master list.")

            # Show aggregated missing skills
            all_missing = []
            for s in results_df["Missing Skills"]:
                if s and s != "None":
                    all_missing.extend([x.strip() for x in s.split(",") if x.strip()!=""])
            if all_missing:
                miss_freq = pd.Series(all_missing).value_counts().reset_index()
                miss_freq.columns = ["Skill", "Count"]
                fig5 = px.bar(miss_freq.head(20), x="Skill", y="Count", title="Top Missing JD Skills")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No missing skills (or JD skills not identified).")

        with tab4:
            st.subheader("Analysis History")
            hist = get_history()
            if hist.empty:
                st.write("No history yet.")
            else:
                st.dataframe(hist, use_container_width=True)

        # Downloads
        st.download_button("⬇️ Download CSV (results)", data=csv_bytes, file_name="resume_results.csv", mime="text/csv")

        pdf_bytes = create_pdf_report(results_df.drop(columns=["Raw Text"]), job_text)
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="resume_report.pdf", mime="application/pdf")
