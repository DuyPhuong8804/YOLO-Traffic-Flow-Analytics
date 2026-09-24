@echo off
REM Khoi chay giao dien demo (Streamlit) va API thong ke (FastAPI)
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Chua co moi truong venv. Xem huong dan cai dat trong README.md
    pause
    exit /b 1
)
start "Traffic API" /min venv\Scripts\uvicorn.exe traffic.api:app --app-dir src --port 8000
venv\Scripts\streamlit.exe run app\streamlit_app.py
