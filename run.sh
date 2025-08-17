#!/bin/bash

# Medical Diagnosis Bot - Quick Start Script
echo "🏥 Medical Diagnosis Bot with Federated Learning"
echo "=============================================="
echo ""

# Function to display menu
show_menu() {
    echo "Please select an option:"
    echo "1) Start Web Application (Streamlit)"
    echo "2) Start Federated Learning Server"
    echo "3) Start Federated Learning Client"
    echo "4) Start Complete System (Docker)"
    echo "5) Train Model (Standard)"
    echo "6) Run Inference/Evaluation"
    echo "7) Install Dependencies"
    echo "8) Exit"
    echo ""
}

# Function to install dependencies
install_deps() {
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed!"
}

# Function to start web app
start_webapp() {
    echo "🌐 Starting Web Application..."
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
}

# Function to start FL server
start_fl_server() {
    echo "🖥️ Starting Federated Learning Server..."
    ./scripts/start_fl_server.sh
}

# Function to start FL client
start_fl_client() {
    echo "💻 Starting Federated Learning Client..."
    read -p "Enter client ID (e.g., hospital_1): " client_id
    ./scripts/start_fl_client.sh --client-id "$client_id"
}

# Function to start docker system
start_docker() {
    echo "🐳 Starting Complete System with Docker..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
        echo "✅ System started! Access the web app at http://localhost:8501"
    else
        echo "❌ Docker Compose not found. Please install Docker and Docker Compose."
    fi
}

# Function to train model
train_model() {
    echo "🧠 Starting Model Training..."
    ./scripts/train.sh
}

# Function to run inference
run_inference() {
    echo "🔍 Running Model Inference..."
    ./scripts/predict.sh
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-8): " choice
    
    case $choice in
        1)
            start_webapp
            ;;
        2)
            start_fl_server
            ;;
        3)
            start_fl_client
            ;;
        4)
            start_docker
            ;;
        5)
            train_model
            ;;
        6)
            run_inference
            ;;
        7)
            install_deps
            ;;
        8)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done