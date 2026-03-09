"""
Scraping logic for LinkedIn and Google Jobs
"""
import os
import time
import json
import random
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import requests

from job_finder.config import CHROMEDRIVER_PATH, CHROME_USER_DATA_DIR

def get_chrome_driver(options: Options = None, user_data_dir: str = None) -> webdriver.Chrome:
    """
    Initialize and return a Chrome WebDriver.
    """
    if options is None:
        options = Options()
    
    if user_data_dir:
        # Avoid escape sequence issues with backslashes on Windows
        safe_path = user_data_dir.replace("\\", "/")
        options.add_argument(f"user-data-dir={safe_path}")
        print(f"Using Chrome user data directory: {safe_path}")
        
    return webdriver.Chrome(options=options)

def scrape_linkedin_jobs(db=None, max_jobs: int = 5, num_scrolls: int = 40) -> pd.DataFrame:
    """
    Scrape LinkedIn jobs from a target URL.
    Checks against JobDatabase to skip existing jobs.
    Returns a DataFrame of newly scraped jobs.
    """
    jobs_list = []
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/107.0.0.0 Safari/537.36")
    
    # Set the target URL here (adjust as needed)
    target_url = ("https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location=San%20Jose&geoId=106233382&distance=50&f_TPR=r2592000&position=1&pageNum=0")
    
    driver = get_chrome_driver(options)
    driver.get(target_url)
    time.sleep(random.uniform(18, 22))
    
    # Scroll and click "See more jobs" button if present
    for i in range(num_scrolls):
        if i == 1:
            time.sleep(300.284)
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(random.uniform(1.5, 5))
        try:
            see_more_button = driver.find_element(
                By.XPATH,
                '//button[@aria-label="See more jobs" and contains(@class, "infinite-scroller__show-more-button")]'
            )
            if see_more_button.is_displayed():
                see_more_button.click()
                time.sleep(random.uniform(1.5, 5))
        except Exception:
            pass

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    job_cards = soup.find_all('div', class_='base-card')
    print(f"Total LinkedIn job cards found: {len(job_cards)}")
    
    job_ids = []
    for card in job_cards:
        if len(job_ids) >= max_jobs:
            break
        try:
            job_id = card.get('data-entity-urn').split(":")[3]
            # Skip if we already have it in the database
            if db and db.job_exists(job_id):
                print(f"Skipping already scraped job {job_id}")
                continue
            job_ids.append(job_id)
        except AttributeError:
            continue


    job_details_url = 'https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}'
    
    try:
        for job_id in job_ids:
            retries = 0
            max_retries = 3
            success = False
            while not success and retries < max_retries:
                try:
                    driver.get(job_details_url.format(job_id))
                    time.sleep(random.uniform(1.5, 3))

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    job_info = {}

                    try:
                        job_info["companies"] = (soup.find("div", {"class": "top-card-layout__card"})
                                                    .find("a")
                                                    .find("img")
                                                    .get('alt'))
                    except AttributeError:
                        job_info["companies"] = None

                    try:
                        job_info["titles"] = (soup.find("div", {"class": "top-card-layout__entity-info"})
                                                    .find("a")
                                                    .text.strip())
                    except AttributeError:
                        job_info["titles"] = None

                    try:
                        job_info["level"] = (soup.find("ul", {"class": "description__job-criteria-list"})
                                                    .find("li")
                                                    .text.replace("Seniority level", "")
                                                    .strip())
                    except AttributeError:
                        job_info["level"] = None

                    try:
                        job_info["location"] = (soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"})
                                                    .text.strip())
                    except AttributeError:
                        job_info["location"] = None

                    try:
                        job_info["desc"] = (soup.find("div", {"class": "show-more-less-html__markup"})
                                                .get_text(separator=' ', strip=True))
                    except AttributeError:
                        job_info["desc"] = None

                    job_info["job_id"] = job_id
                    job_info["source"] = "LinkedIn"
                    
                    jobs_list.append(job_info)
                    
                    if db:
                        db.insert_job(job_info)

                    time.sleep(random.uniform(5, 15))
                    success = True
                except Exception as inner_e:
                    print(f"Error scraping job_id {job_id} (attempt {retries+1}): {inner_e}")
                    retries += 1
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = get_chrome_driver(options)
            if not success:
                print(f"Failed to scrape job_id {job_id} after {max_retries} attempts, skipping.")
    except Exception as e:
        print(f"General scraping error: {e}. Returning partial results.")
    finally:
        driver.quit()

    df = pd.DataFrame(jobs_list)
    return df

def scrape_linkedin_jobs_api(db=None, keywords="Data Scientist", location="San Jose", max_jobs=25, distance=50, f_tpr=None):
    """
    Scrape LinkedIn jobs using the internal guest search API (faster than Selenium).
    Supports spatial (distance) and temporal (f_tpr) filtering.
    f_tpr values: 'r86400' (day), 'r604800' (week), 'r2592000' (month)
    """
    jobs_list = []
    base_search_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    detail_api_url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    start = 0
    while len(jobs_list) < max_jobs:
        params = {
            "keywords": keywords,
            "location": location,
            "start": start,
            "distance": distance
        }
        if f_tpr:
            params["f_TPR"] = f_tpr
        
        try:
            response = requests.get(base_search_url, params=params, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching job list: {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            job_cards = soup.find_all('div', class_='base-card')
            if not job_cards:
                print("No job cards found in API response.")
                break
            
            print(f"Total jobs found on API page: {len(job_cards)}")
            
            job_ids = []
            for card in job_cards:
                try:
                    job_id = card.get('data-entity-urn').split(":")[3]
                    if db and db.job_exists(job_id):
                        continue
                    job_ids.append(job_id)
                except (AttributeError, IndexError):
                    continue
            
            print(f"Found {len(job_ids)} new job IDs on API page (start={start})")
            
            for job_id in job_ids:
                if len(jobs_list) >= max_jobs:
                    break
                
                # Fetch details via API
                try:
                    detail_resp = requests.get(detail_api_url.format(job_id), headers=headers)
                    if detail_resp.status_code == 200:
                        detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')
                        job_info = {"job_id": job_id, "source": "LinkedIn"}
                        
                        # Use existing parsing logic (DRY principle would be better but keeping it simple for now)
                        try:
                            job_info["companies"] = (detail_soup.find("div", {"class": "top-card-layout__card"})
                                                        .find("a").find("img").get('alt'))
                        except: job_info["companies"] = None
                        
                        try:
                            job_info["titles"] = (detail_soup.find("div", {"class": "top-card-layout__entity-info"})
                                                       .find("a").text.strip())
                        except: job_info["titles"] = None
                        
                        try:
                            job_info["desc"] = (detail_soup.find("div", {"class": "show-more-less-html__markup"})
                                                    .get_text(separator=' ', strip=True))
                        except: job_info["desc"] = None
                        
                        jobs_list.append(job_info)
                        if db:
                            db.insert_job(job_info)
                        
                        print(f"Fetched {job_id}: {job_info['titles']}")
                        time.sleep(random.uniform(2, 5))
                except Exception as e:
                    print(f"Error fetching details for {job_id}: {e}")
            
            start += 25
            if not job_ids: # No more new jobs found
                break
                
        except Exception as e:
            print(f"API scraping error: {e}")
            break
            
    return pd.DataFrame(jobs_list)

def scrape_google_jobs(job_board_url: str, num_scrolls: int = 2) -> pd.DataFrame:
    """
    Scrape job postings from a Google jobs board URL and return a DataFrame.
    """
    driver = get_chrome_driver()
    driver.get(job_board_url)
    time.sleep(3)
    
    # Activate the job area by clicking the first job card
    try:
        first_job_card = driver.find_element(
            By.CSS_SELECTOR, 'div.GoEOPd, li.iFjolb.gws-plugins-horizon-jobs__li-ed'
        )
        first_job_card.click()
    except Exception as e:
        print("Could not click first job card:", e)
    time.sleep(1)
    
    # Scroll down to load more jobs
    for _ in range(num_scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    job_cards = driver.find_elements(By.XPATH, "//c-wiz[@data-encoded-docid]")
    print("Job cards found:", len(job_cards))
    if len(job_cards) == 0:
        job_cards = soup.find_all('li', class_='iFjolb gws-plugins-horizon-jobs__li-ed')
    
    all_descriptions = soup.find_all('span', class_='HBvzbc')
    jobs = []
    
    # Loop over all but the first job card
    for index, card in enumerate(job_cards[1:]):
        try:
            title = card.find_element(
                By.CSS_SELECTOR, 'h1.LZAQDf.cS4Vcb-pGL6qe-IRtXtf'
            ).text.strip()
        except Exception:
            title = 'N/A'
        try:
            company = card.find_element(By.CSS_SELECTOR, 'div.vNEEBe').text.strip()
        except Exception:
            company = 'N/A'
        try:
            location = card.find_element(By.CSS_SELECTOR, 'div.Qk80Jf').text.strip()
        except Exception:
            location = 'N/A'
        if index < len(all_descriptions):
            description = all_descriptions[index].get_text(separator=' ', strip=True)
        else:
            description = 'N/A'
        jobs.append({'titles': title, 'companies': company, 'location': location, 'desc': description})
    
    driver.quit()
    
    google_jobs_df = pd.DataFrame(jobs)
    google_jobs_df['source'] = 'Google'
    return google_jobs_df
