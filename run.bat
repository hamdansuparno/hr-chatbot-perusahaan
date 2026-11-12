@echo off
title HR Virtual Assistant & Aturan Perusahaan
echo ================================================================
echo      MEMULAI APLIKASI HR CHATBOT PERUSAHAAN (WEB & HP)
echo ================================================================
echo.
echo [1] Buka di Komputer / Laptop ini : http://127.0.0.1:5000
echo [2] Buka di HP / Smartphone       : Klik tombol 'Buka di HP' pada web
echo.
echo ================================================================

start http://127.0.0.1:5000

IF EXIST ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe app.py
) ELSE (
    python app.py
)
pause
