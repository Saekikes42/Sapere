$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'logs') -Force | Out-Null
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Start-Process -FilePath $python -ArgumentList '-m streamlit run sapere_hub.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'logs\hub.stdout.log' -RedirectStandardError 'logs\hub.stderr.log'
Start-Process -FilePath $python -ArgumentList '-m uvicorn backend_api:app --host 127.0.0.1 --port 8000' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'logs\api.stdout.log' -RedirectStandardError 'logs\api.stderr.log'
Write-Output 'Panel: http://127.0.0.1:8501'
Write-Output 'API: http://127.0.0.1:8000/docs'
