@echo off
cd /d "%~dp0"
python run.py
start "" "%~dp0output\dashboard.html"
