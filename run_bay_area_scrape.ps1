# run_bay_area_scrape.ps1
# Scrapes 10 Tech Hub cities along the Peninsula from San Jose to San Francisco.

$Cities = @(
    "Los Gatos, CA",
    "Santa Cruz, CA",
    "Saratoga, CA",
    "Campbell, CA",
    "Cupertino, CA",
    "San Jose, CA",
    "Santa Clara, CA",
    "Sunnyvale, CA",
    "Mountain View, CA",
    "Los Altos, CA",
    "Palo Alto, CA",
    "Menlo Park, CA",
    "Redwood City, CA",
    "San Mateo, CA",
    "South San Francisco, CA",
    "San Francisco, CA"
)

$PythonPath = "C:\Users\awill\anaconda3\envs\j_scraper\python.exe"
$RepoPath = "C:\Users\awill\OneDrive\Documents\git_repos\job-finder"
$env:PYTHONPATH = $RepoPath

Write-Host "Starting Bay Area Multi-City Scrape (Peninsula Crawl)..." -ForegroundColor Cyan
Write-Host "Parameters: 1-mile radius, 3-week window (month filter), Max 50 pages/city." -ForegroundColor Gray

foreach ($City in $Cities) {
    Write-Host "`n>>> Processing City: $City" -ForegroundColor Green
    & $PythonPath "$RepoPath\main.py" --semantic --api --keywords "Data Scientist" --location "$City" --distance 1 --max-pages 50 --f-tpr r2592000 --scrape-only
}

Write-Host "`nAll 10 cities processed! Check your database for the results." -ForegroundColor Cyan
