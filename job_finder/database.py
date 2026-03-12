import sqlite3
import os
from datetime import datetime
import pandas as pd
from job_finder.logging_config import get_logger

logger = get_logger(__name__)

class JobDatabase:
    def __init__(self, db_path="data/jobs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        logger.debug(f"Connected to database at {self.db_path}")
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
                posted_at TIMESTAMP,
                similarity_score REAL,
                score_geospatial REAL,
                score_energy REAL,
                score_cv_robotics REAL,
                score_llm_science REAL,
                html TEXT
            )
        ''')
        
        # Migration: Add new columns if they don't exist
        new_columns = [
            ("posted_at", "TIMESTAMP"),
            ("score_geospatial", "REAL"),
            ("score_energy", "REAL"),
            ("score_cv_robotics", "REAL"),
            ("score_llm_science", "REAL"),
            ("html", "TEXT")
        ]
        
        cursor.execute("PRAGMA table_info(jobs)")
        existing_columns = [info[1] for info in cursor.fetchall()]
        
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                logger.info(f"Adding column '{col_name}' to 'jobs' table.")
                cursor.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")
                
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
            (job_id, source, titles, companies, location, level, desc, scraped_at, posted_at,
             similarity_score, score_geospatial, score_energy, score_cv_robotics, score_llm_science, html)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_data.get('job_id'),
            job_data.get('source'),
            job_data.get('titles'),
            job_data.get('companies'),
            job_data.get('location'),
            job_data.get('level'),
            job_data.get('desc'),
            scraped_at,
            job_data.get('posted_at'),
            job_data.get('similarity_score'),
            job_data.get('score_geospatial'),
            job_data.get('score_energy'),
            job_data.get('score_cv_robotics'),
            job_data.get('score_llm_science'),
            job_data.get('html')
        ))
        self.conn.commit()
        logger.debug(f"Inserted/Updated job {job_data.get('job_id')} into database.")

    def update_job_scores(self, job_id: str, scores: dict):
        """
        Updates the similarity scores for an existing job.
        scores: dict with keys like 'score_geospatial', 'similarity_score', etc.
        """
        cursor = self.conn.cursor()
        set_clause = ", ".join([f"{k} = ?" for k in scores.keys()])
        params = list(scores.values()) + [job_id]
        cursor.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", params)
        self.conn.commit()

    def get_all_jobs(self):
        return pd.read_sql_query("SELECT * FROM jobs", self.conn)

    def close(self):
        self.conn.close()
