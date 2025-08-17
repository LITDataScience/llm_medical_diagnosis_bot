#!/bin/bash

# Start Federated Learning Client
echo "🏥 Starting Federated Learning Client for Medical Diagnosis Bot"
echo "============================================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Client configuration
SERVER_ADDRESS="localhost:8080"
CLIENT_ID=""
DATA_FOLDER="dataset"
PRIVACY_MULTIPLIER=1.0
CLIP_NORM=1.0
LOCAL_EPOCHS=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --server)
      SERVER_ADDRESS="$2"
      shift 2
      ;;
    --client-id)
      CLIENT_ID="$2"
      shift 2
      ;;
    --data-folder)
      DATA_FOLDER="$2"
      shift 2
      ;;
    --privacy-multiplier)
      PRIVACY_MULTIPLIER="$2"
      shift 2
      ;;
    --local-epochs)
      LOCAL_EPOCHS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Generate client ID if not provided
if [ -z "$CLIENT_ID" ]; then
    CLIENT_ID="client_$(hostname)_$(date +%s)"
fi

# Create necessary directories
mkdir -p checkpoints/federated/$CLIENT_ID
mkdir -p logs

# Start client
echo "Connecting to server at: $SERVER_ADDRESS"
echo "Client ID: $CLIENT_ID"
echo "Data folder: $DATA_FOLDER"
echo "Local epochs: $LOCAL_EPOCHS"

python federated_learning/fl_client.py \
    --server "$SERVER_ADDRESS" \
    --client-id "$CLIENT_ID" \
    --data-folder "$DATA_FOLDER" \
    --privacy-multiplier "$PRIVACY_MULTIPLIER" \
    --clip-norm "$CLIP_NORM" \
    --local-epochs "$LOCAL_EPOCHS" 2>&1 | tee logs/fl_client_${CLIENT_ID}_$(date +%Y%m%d_%H%M%S).log