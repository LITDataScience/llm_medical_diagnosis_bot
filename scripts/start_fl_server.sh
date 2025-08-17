#!/bin/bash

# Start Federated Learning Server
echo "🚀 Starting Federated Learning Server for Medical Diagnosis Bot"
echo "=================================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Server configuration
SERVER_ADDRESS="0.0.0.0:8080"
NUM_ROUNDS=100
MIN_CLIENTS=2
FRACTION_FIT=0.5
PRIVACY_BUDGET=10.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --address)
      SERVER_ADDRESS="$2"
      shift 2
      ;;
    --rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --min-clients)
      MIN_CLIENTS="$2"
      shift 2
      ;;
    --no-dp)
      NO_DP="--no-dp"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p checkpoints/federated
mkdir -p logs

# Start server
echo "Starting server at $SERVER_ADDRESS"
echo "Number of rounds: $NUM_ROUNDS"
echo "Minimum clients: $MIN_CLIENTS"
echo "Differential Privacy: ${NO_DP:-Enabled}"

python federated_learning/fl_server.py \
    --address "$SERVER_ADDRESS" \
    --rounds "$NUM_ROUNDS" \
    --min-clients "$MIN_CLIENTS" \
    --fraction-fit "$FRACTION_FIT" \
    --privacy-budget "$PRIVACY_BUDGET" \
    $NO_DP 2>&1 | tee logs/fl_server_$(date +%Y%m%d_%H%M%S).log