import os
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from job_finder.config import CHROMEDRIVER_PATH, CHROME_USER_DATA_DIR

class AuthenticatedLinkedInScraper:
    def __init__(self, driver, db=None):
        self.driver = driver
        self.db = db

    def scrape_jobs(self, target_url, max_pages=5):
        """
        Scrape multiple pages of LinkedIn jobs while logged in.
        """
        all_new_jobs = []
        self.driver.get(target_url)
        
        # Ensure user is logged in
        if not self._wait_for_login():
            print("Proceeding anyway, but scraper may fail if not logged in.")
        
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")
            
            # Scroll the jobs rail to load all cards
            self._scroll_jobs_rail()
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            # Try multiple selectors for job cards
            job_cards = soup.select('li.jobs-search-results-list__item') or \
                        soup.select('.job-card-container') or \
                        soup.select('.jobs-search-results__list-item')
            
            print(f"Found {len(job_cards)} job cards on page {page}")

            new_job_ids = []
            for card in job_cards:
                job_id = card.get('data-job-id')
                if not job_id:
                    # Try alternative way to get job ID from link
                    link = card.select_one('a.job-card-list__title--link')
                    if link and 'view/' in link.get('href', ''):
                        job_id = link.get('href').split('view/')[1].split('/')[0]
                
                if job_id:
                    if self.db and self.db.job_exists(job_id):
                        # Ensure we have details, otherwise re-fetch
                        if self._job_has_details(job_id):
                            continue
                    new_job_ids.append(job_id)

            print(f"Found {len(new_job_ids)} new jobs to fetch on this page")
            
            for job_id in new_job_ids:
                job_info = self._fetch_job_details(job_id)
                if job_info:
                    all_new_jobs.append(job_info)
                    if self.db:
                        self.db.insert_job(job_info)
                time.sleep(random.uniform(2, 5))

            if page < max_pages:
                if not self._go_to_next_page(page + 1):
                    print("No more pages found or could not navigate.")
                    break
                time.sleep(random.uniform(4, 7))

        return pd.DataFrame(all_new_jobs)

    def _scroll_jobs_rail(self):
        """
        Scroll the left-hand jobs rail to ensure all cards are loaded.
        """
        try:
            # Try multiple possible rail selectors
            rail = None
            for selector in ['.jobs-search-results-list', '.scaffold-layout__list', '.jobs-search-results-display__list']:
                try:
                    rail = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if rail:
                        break
                except:
                    continue
            
            if not rail:
                print("Could not find jobs rail to scroll.")
                return

            for _ in range(10):
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", rail)
                time.sleep(0.5)
        except Exception as e:
            print(f"Error scrolling rail: {e}")

    def _fetch_job_details(self, job_id):
        """
        Fetch details for a single job ID in a new tab.
        """
        view_url = f'https://www.linkedin.com/jobs/view/{job_id}/'
        
        # Open a new tab and switch to it
        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        
        self.driver.get(view_url)
        time.sleep(random.uniform(2, 4))
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        job_info = {"job_id": job_id, "source": "LinkedIn"}
        
        try:
            # Selectors might be different in the authenticated view
            # Using more robust/semantic selectors found via investigation
            title_elem = soup.select_one('.job-details-jobs-unified-top-card__job-title') or \
                         soup.select_one('h1') or \
                         soup.select_one('.top-card-layout__title') or \
                         soup.select_one('h2.t-24')
            job_info["titles"] = title_elem.get_text(strip=True) if title_elem else None

            company_elem = soup.select_one('a[href*="/company/"]') or \
                           soup.select_one('.job-details-jobs-unified-top-card__company-name') or \
                           soup.select_one('.topcard__flavor--bullet') or \
                           soup.select_one('.job-details-jobs-unified-top-card__primary-description a')
            job_info["companies"] = company_elem.get_text(strip=True) if company_elem else None

            # Location is often in the primary description as a span
            location_elem = soup.select_one('.job-details-jobs-unified-top-card__bullet') or \
                            soup.select_one('.topcard__flavor--bullet') or \
                            soup.select_one('.job-details-jobs-unified-top-card__primary-description span:nth-child(2)')
            
            if not location_elem:
                # Try finding text with location-like format
                primary_desc = soup.select_one('.job-details-jobs-unified-top-card__primary-description')
                if primary_desc:
                    # Often format is "Company Name · Location · Date"
                    parts = primary_desc.get_text(separator='|').split('|')
                    if len(parts) > 1:
                        class SimpleElem:
                            def __init__(self, text): self.text = text
                            def get_text(self, strip=True): return self.text.strip() if strip else self.text
                        location_elem = SimpleElem(parts[1])

            job_info["location"] = location_elem.get_text(strip=True) if location_elem else None

            # Posting date in authenticated view
            date_elem = soup.select_one('.job-details-jobs-unified-top-card__primary-description span:nth-child(3)') or \
                        soup.select_one('span.tvm__text--low-emphasis:nth-child(3)') or \
                        soup.select_one('.posted-time-ago__text')
            job_info["posted_at"] = date_elem.get_text(strip=True) if date_elem else None

            desc_elem = soup.select_one('#workspace') or \
                        soup.select_one('.jobs-description__content') or \
                        soup.select_one('.show-more-less-html__markup')
            job_info["desc"] = desc_elem.get_text(separator=' ', strip=True) if desc_elem else None
            
            # Additional fields if available
            # Fix: BeautifulSoup does not support :contains in select_one
            level_elem = None
            insights = soup.select('.job-details-jobs-unified-top-card__job-insight')
            for insight in insights:
                if "Experience level" in insight.get_text():
                    level_elem = insight
                    break
            
            
            job_info["level"] = level_elem.get_text(strip=True).replace("Experience level", "").strip() if (level_elem and hasattr(level_elem, 'get_text')) else None
            
        except Exception as e:
            print(f"Error parsing job {job_id}: {e}")
            self._close_tab_safely()
            return None
            
        self._close_tab_safely()
        return job_info

    def _close_tab_safely(self):
        """
        Close the current tab and switch back to the main window safely.
        """
        try:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
        except Exception as e:
            print(f"Warning: Could not close tab: {e}")
        
        try:
            self.driver.switch_to.window(self.driver.window_handles[0])
        except Exception as e:
            print(f"Error switching back to main window: {e}")

    def _wait_for_login(self, timeout=300):
        """
        Wait for a logged-in element to appear.
        """
        print(f"Waiting up to {timeout} seconds for login/session...")
        try:
            # Elements that only appear when logged in
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".global-nav__me, .jobs-search-results-list"))
            )
            print("Login/Session verified.")
            return True
        except Exception:
            print("Logged-in element not found within timeout.")
            return False

    def _go_to_next_page(self, next_page_num):
        """
        Navigate to the next page of results.
        """
        try:
            # Scroll to the bottom of the rail to find pagination
            self._scroll_jobs_rail()
            time.sleep(1)

            # Try multiple selectors for the 'Next' button
            next_selectors = [
                'button.jobs-search-pagination__button--next',
                'button[aria-label="View next page"]',
                'button[aria-label="Next"]'
            ]
            
            for selector in next_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and elements[0].is_enabled():
                    try:
                        self.driver.execute_script("arguments[0].click();", elements[0])
                        return True
                    except:
                        continue

            # Alternative: Click the page number
            page_selectors = [
                f'button[aria-label="Page {next_page_num}"]',
                f'button.jobs-search-pagination__indicator-button[aria-label*="{next_page_num}"]'
            ]
            for selector in page_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    try:
                        self.driver.execute_script("arguments[0].click();", elements[0])
                        return True
                    except:
                        continue
                
        except Exception as e:
            print(f"Error navigating to page {next_page_num}: {e}")
            
        return False

    def _job_has_details(self, job_id):
        """
        Check if a job in the database already has sufficient details.
        """
        if not self.db:
            return False
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT titles, desc FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row and row[0] and row[1]:
                return True
        except Exception as e:
            print(f"Error checking job details: {e}")
        return False
