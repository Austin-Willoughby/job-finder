"""
Configuration settings and constants for the Job Finder project.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Set your chromedriver path here:
CHROMEDRIVER_PATH = r"C:\Users\awill\OneDrive\Documents\Job Search\job_scraper\chromedriver-win64\chromedriver.exe"

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAISECRETKEY")

# Chrome User Data Directory (to persist login)
# Example: r"C:\Users\awill\AppData\Local\Google\Chrome\User Data"
CHROME_USER_DATA_DIR = os.getenv("CHROME_USER_DATA_DIR")

# Fix python-dotenv mangling if backslashes were interpreted as escapes (e.g. \a -> \x07)
if CHROME_USER_DATA_DIR:
    # Common mangled escapes in paths
    CHROME_USER_DATA_DIR = CHROME_USER_DATA_DIR.replace('\x07', r'\a')
    CHROME_USER_DATA_DIR = CHROME_USER_DATA_DIR.replace('\t', r'\t')
    CHROME_USER_DATA_DIR = CHROME_USER_DATA_DIR.replace('\n', r'\n')
    CHROME_USER_DATA_DIR = CHROME_USER_DATA_DIR.replace('\r', r'\r')
    # Strip quotes if present
    CHROME_USER_DATA_DIR = CHROME_USER_DATA_DIR.strip('"\'')

# Define keyword bins for feature engineering
KEYWORDS_BINS = {
    'environmental': [
        'environmental', 'climate change', 'earth science', 'earth sciences',
        'sustainable energy', 'renewable energy',
        'ecology', 'biodiversity', 'natural resources', 'green technology',
        'green tech', 'clean energy', 'carbon', 'methane', 'GHG', 'green house', 'co2',
        'air quality', 'water quality', 'soil quality', 'soil', 'sustainable development', 'carbon footprint',
        'sequestration', 'climate modeling', 'climate simulation', 'climate', 'EPA',
        'environmental protection', 'environmental management', 'environmental policy',
        'natural disaster', 'disaster management', 'disaster', 'risk', 'conservation',
        'environmental impact', 'sustainability', 'environmental monitoring', 'sustainable',
    ],
    'CV_autonomous_robotics': [
        'computer vision', 'robotics', 'autonomous vehicles', 'self-driving cars',
        'segmentation', 'CNN', 'OpenCV', 'autonomous'
    ],
    'LLM_related': [
        'LLM', 'LLMs', 'large language model', 'large language models',
        'generative AI', 'generative artificial intelligence', 'gen AI', 'genAI'
    ],
    'geospatial_r_sensing': [
        'remote sensing', 'satellite imagery', 'satellite', 'spatial', 'GIS',
        'geographic information system', 'earth observation', 'geospatial analysis',
        'spatial analysis', 'spatial modeling', 'remote monitoring', 'geoinformatics',
        'UAV', 'UAVs', 'drones', 'geopandas', 'arcgis', 'aerial vehicles',
    ],
    'energy': [
        'energy systems', 'electric', 'natural gas', 'utility', 'utiltities',
        'solar energy', 'wind energy', 'hydropower', 'thermal energy', 'energy system', 'energy network',
        'bioenergy', 'hydrogen energy', 'power grid', 'battery'
    ],
    'coding': [
        'python', 'deep learning', 'DL',
        'neural net', 'tensorflow',
    ]
}

PROFILE_TEXT = """
M.S. Data Science graduate and M.S. Environmental Engineering graduate with 4 years of experience applying machine learning, geospatial analytics, and algorithm development to real-world problems 
in climate tech, energy, and environmental sectors. Strong background in Python and SQL for data science, modeling, and production-grade analytics, with expertise in TensorFlow, scikit-learn, 
GeoPandas, and remote sensing. Skilled in building custom models, physical system algorithms, and applied statistical solutions. Passionate about working on meaningful, mission-driven problems that 
have tangible impact on critical industries such as energy, climate, infrastructure, and environmental resilience. Additional strengths include geospatial data science, remote sensing-based machine learning, methane leak detection, and physical system modeling. Skilled in building and scaling machine 
learning algorithms for production systems, including customer-facing applications in climate tech, gas utilities, and infrastructure risk modeling. Experienced in ETL pipeline development, dashboard 
creation, and applied statistical and deep learning techniques.
"""

CRITERIA = """
- Roles should primarily involve Python-based data science, machine learning, or algorithm development.
- Prefer mid-level to early-senior roles; entry-level roles are acceptable if they align closely with my background.
- Strong interest in geospatial analytics, GIS, environmental science, energy, earth science, resource extraction, electric grid, EVs, civil infrastructure, and climate-related industries.
- Preference for mission-driven companies focused on meaningful impact (e.g., energy, environment, infrastructure).
- Less interested in roles in industries which aren't important to humanity (e.g. luxury goods, consumer packaged goods, marketing, or aesthetics industries).
- Open to any company size (startup to enterprise).
- Location preferences: based in the San Francisco Bay Area or fully remote. Open to hybrid if location is within reasonable commute to Santa Cruz.
- Favor job titles like Data Scientist, Geospatial Data Scientist, or Energy Data Analyst.
- Bonus for opportunities involving statistical model development, algorithm R&D, or physical modeling.
- Interest in applied machine learning, real-world problem solving, and production-grade analytics.
"""

MODEL_NAME = "gpt-4o-mini"
COST_PER_1K_TOKENS = 0.0006
