import sqlite3
import pandas as pd

conn = sqlite3.connect("data/resume_screening.db")

# View all tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)

# View job history
job_history = pd.read_sql_query("SELECT * FROM job_history", conn)
print(job_history)

# View all resume results
resume_results = pd.read_sql_query("SELECT * FROM resume_results", conn)
print(resume_results.head())

conn.close()
