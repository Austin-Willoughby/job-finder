import sqlite3
import os
from datetime import datetime
import pandas as pd

class JobDatabase:
    def __init__(self, db_path="data/jobs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                source TEXT,
                titles TEXT,
                companies TEXT,
                location TEXT,
                level TEXT,
                desc TEXT,
                scraped_at TIMESTAMP,
                similarity_score REAL,
                html TEXT
            )
        ''')
        self.conn.commit()

    def job_exists(self, job_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,))
        return cursor.fetchone() is not None

    def insert_job(self, job_data: dict):
        cursor = self.conn.cursor()
        scraped_at = datetime.now().isoformat()
        cursor.execute('''
            INSERT OR REPLACE INTO jobs 
            (job_id, source, titles, companies, location, level, desc, scraped_at, similarity_score, html)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_data.get('job_id'),
            job_data.get('source'),
            job_data.get('titles'),
            job_data.get('companies'),
            job_data.get('location'),
            job_data.get('level'),
            job_data.get('desc'),
            scraped_at,
            job_data.get('similarity_score'),
            job_data.get('html')
        ))
        self.conn.commit()

    def get_all_jobs(self):
        return pd.read_sql_query("SELECT * FROM jobs", self.conn)

    def close(self):
        self.conn.close()
