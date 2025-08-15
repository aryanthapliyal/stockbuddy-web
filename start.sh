#!/bin/bash

echo "Starting StockBuddy Application..."
echo

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use. Please stop the service using that port."
        return 1
    fi
    return 0
}

# Check ports
echo "Checking if ports are available..."
check_port 5001 || exit 1
check_port 5000 || exit 1
check_port 5173 || exit 1

echo "All ports are available."
echo

# Start Model Backend
echo "Starting Model Backend (Python Flask)..."
cd Model_Backend
python app.py &
MODEL_PID=$!
cd ..

echo "Waiting 5 seconds for Model Backend to start..."
sleep 5

# Start Backend API
echo "Starting Backend API (Node.js)..."
cd Backend
npm run dev &
BACKEND_PID=$!
cd ..

echo "Waiting 5 seconds for Backend API to start..."
sleep 5

# Start Frontend
echo "Starting Frontend (React)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "All services are starting..."
echo
echo "Frontend will be available at: http://localhost:5173"
echo "Backend API will be available at: http://localhost:5000"
echo "Model Backend will be available at: http://localhost:5001"
echo
echo "Press Ctrl+C to stop all services..."

# Function to cleanup on exit
cleanup() {
    echo
    echo "Stopping all services..."
    kill $MODEL_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "All services stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
