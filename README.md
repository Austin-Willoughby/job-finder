# Job Finder

Job Finder is a modular Python library designed to scrape job postings from LinkedIn and Google, classify them using Machine Learning models, and evaluate their relevance using semantic search and OpenAI's GPT models.

## Features

- **Rapid API Discovery**: Fetch dozens of jobs in seconds using LinkedIn's internal guest APIs (no Selenium needed).
- **Authenticated Scraping**: Support for authenticated LinkedIn sessions to bypass guest limits and handle persistent logins.
- **Zero-Shot Semantic Search**: Filter and rank jobs instantly based on your profile and criteria using vector embeddings.
- **ML Classification**: Processes job data with TF-IDF and classifies them using optimized SVM or Logistic Regression models.
- **OpenAI Integration**: Automatically evaluates job desirability based on personalized criteria.
- **Persistent Storage**: All scraped jobs are stored in a local SQLite database (`data/jobs.db`) to prevent duplicates.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd job-finder
   ```

2. **Set up the Environment**:
   It is recommended to use the provided Conda environment:
   ```bash
   conda activate j_scraper
   pip install -r requirements.txt
   ```

3. **Download Embedding Models**:
   Run the helper script to download the local embedding model for semantic search:
   ```bash
   python download_model.py
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   CHROME_USER_DATA_DIR=C:\Users\YourUser\path\to\chrome-profile
   ```

## Usage

The library is controlled via the `main.py` entry point.

### 1. Rapid API Discovery (Fastest)
Scrape jobs using internal APIs. This is significantly faster than browser automation.
```bash
# Scrape jobs into the database without processing
python main.py --semantic --api --max-pages 5 --scrape-only

# With custom filters
python main.py --semantic --api --distance 50 --f-tpr r604800 --scrape-only
```
- `--distance`: Radius in miles.
- `--f-tpr`: Time posted (`r86400`=day, `r604800`=week, `r2592000`=month).

### 2. Authenticated LinkedIn Scraper
Use your actual LinkedIn session via Selenium (requires `CHROME_USER_DATA_DIR` in `.env`).
```bash
python main.py --semantic --auth --max-pages 5 --scrape-only
```

### 3. Run Semantic Ranking
Once jobs are in the database, rank them based on your `PROFILE_TEXT` and `CRITERIA` in `config.py`:
```bash
python main.py --semantic
```

### 4. ML Classification & OpenAI Evaluation
To use the traditional ML pipeline and OpenAI rating:
```bash
python main.py --predict --evaluate
```

---

## Testing

Test scripts are located in the `tests/` directory:
- **API Test**: `python tests/api_scraper_test.py`
- **Integration Test**: `python tests/integration_test.py`
- **Semantic Test**: `python tests/semantic_integration_test.py`

## Project Structure

- `job_finder/`: Core package (scrapers, features, models).
- `main.py`: Main CLI entry point.
- `tests/`: Integration and unit tests.
- `data/`: SQLite database (`jobs.db`) and CSV outputs.
- `models/`: Persistent storage for trained ML models.
- `.env`: API keys and configuration.
