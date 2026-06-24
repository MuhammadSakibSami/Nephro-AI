@echo off
REM Start the React + Vite frontend dev server (http://localhost:5173).
cd /d %~dp0\frontend
if not exist node_modules (
  echo Installing frontend dependencies...
  call npm install
)
if not exist .env if exist .env.example copy .env.example .env >nul
call npm run dev
