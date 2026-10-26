@echo off
rem Run the whole study end to end from Command Prompt or PowerShell (Windows).
rem Same as run_all.sh; options are passed through to src\run_all.py (--no_wait, --skip_runs, --notice).
cd /d "%~dp0"
uv run python src\run_all.py %*
