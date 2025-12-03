#!/bin/bash

# Start services script for Codespaces
echo "🚀 Starting RAG Chatbot services..."

# Function to check if port is in use
check_port() {
    nc -z localhost $1 2>/dev/null
}

# Start backend in background
echo "🔧 Starting FastAPI backend on port 8000..."
cd /workspace
python fastapi_app.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
for i in {1..30}; do
    if check_port 8000; then
        echo "✅ Backend is ready!"
        break
    fi
    sleep 2
done

# Start frontend
echo "🎨 Starting React frontend on port 5173..."
cd frontend
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to start..."
for i in {1..30}; do
    if check_port 5173; then
        echo "✅ Frontend is ready!"
        break
    fi
    sleep 2
done

echo ""
echo "🎉 RAG Chatbot is ready!"
echo "📊 Backend API: http://localhost:8000"
echo "🌐 Frontend UI: http://localhost:5173"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Keep script running
wait