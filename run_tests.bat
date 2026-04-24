@echo off
REM Run senior audit test suite with coverage report
REM Usage: run_tests.bat [--open-html]

python -m pytest tests/test_senior_audit.py ^
    --cov=main ^
    --cov=filters ^
    --cov=cache_manager ^
    --cov=queue_manager ^
    --cov=app ^
    --cov-report=term-missing ^
    --cov-report=html:htmlcov ^
    -v

if "%1"=="--open-html" (
    start htmlcov\index.html
)
