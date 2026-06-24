@echo off
REM Start the FastAPI backend (runs from this repo root; mode: backend.main).
cd /d %~dp0
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
