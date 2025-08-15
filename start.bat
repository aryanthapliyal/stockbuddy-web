@echo off
echo Starting StockBuddy Application...
echo.

echo Starting Model Backend (Python Flask)...
start "Model Backend" cmd /k "cd Model_Backend && python app.py"

echo Waiting 5 seconds for Model Backend to start...
timeout /t 5 /nobreak > nul

echo Starting Backend API (Node.js)...
start "Backend API" cmd /k "cd Backend && npm run dev"

echo Waiting 5 seconds for Backend API to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend (React)...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo All services are starting...
echo.
echo Frontend will be available at: http://localhost:5173
echo Backend API will be available at: http://localhost:5000
echo Model Backend will be available at: http://localhost:5001
echo.
echo Press any key to exit this window...
pause > nul
