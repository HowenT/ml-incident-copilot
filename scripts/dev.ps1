# Start API (:8000) and UI (:8501) in two windows. Run from the repo root after `pip install -r requirements.txt`.
$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; python -m uvicorn backend.app.main:app --port 8000"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; python -m streamlit run frontend/app.py --server.port 8501"
Write-Host "API  http://localhost:8000/docs"
Write-Host "UI   http://localhost:8501   (first start seeds the demo data, ~20 s)"
