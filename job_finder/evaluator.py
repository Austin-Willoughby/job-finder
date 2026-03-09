"""
OpenAI evaluation logic to determine job desirability
"""
from openai import OpenAI
from job_finder.config import OPENAI_API_KEY, PROFILE_TEXT, CRITERIA, MODEL_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

# Global usage counter
total_tokens_used = 0

def get_desirability(company: str, title: str, location: str, desc: str) -> int:
    global total_tokens_used

    prompt = f"""You are a career advisor. Given the user's profile and criteria, and the following job details, rate how likely they would want to apply on a scale from 1 to 10.

User Profile:
{PROFILE_TEXT}

Application Criteria:
{CRITERIA}

Job Details:
- Company: {company}
- Title: {title}
- Location: {location}
- Description: {desc}

Return only the integer rating (1–10), with no additional text.
"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )

    # accumulate usage
    total_tokens_used += resp.usage.total_tokens

    try:
        return int(resp.choices[0].message.content.strip())
    except ValueError:
        return 0
