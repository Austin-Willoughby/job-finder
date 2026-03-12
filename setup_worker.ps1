# setup_worker.ps1
# Sets up a Windows Task Scheduler task to run the Job Finder daily worker.

$JobName = "JobFinder_DailyWorker"
$PythonPath = "C:\Users\awill\anaconda3\envs\j_scraper\python.exe"
$RepoPath = "C:\Users\awill\OneDrive\Documents\git_repos\job-finder"
$ScriptPath = "$RepoPath\job_finder\worker.py"

# Create the action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory $RepoPath

# Create the trigger (Daily at 8:00 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At 8am

# Create the principal (Run as the current user)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

# Register the task
Register-ScheduledTask -TaskName $JobName -Action $Action -Trigger $Trigger -Principal $Principal -Force

Write-Host "Daily Worker task '$JobName' has been set up to run at 8:00 AM daily." -ForegroundColor Green
Write-Host "You can manage it in Windows Task Scheduler."
