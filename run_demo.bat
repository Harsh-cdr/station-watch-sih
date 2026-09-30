@echo off
setlocal

if not exist "tf_env\Scripts\python.exe" (
    echo tf_env was not found.
    echo Create it with: py -3.11 -m venv tf_env
    exit /b 1
)

echo Starting Station Watch...
call "tf_env\Scripts\activate.bat"
python -m uvicorn backend_api:app --reload
