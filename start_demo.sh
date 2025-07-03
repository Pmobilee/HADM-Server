#!/bin/bash

# HADM Visualization Demo Launcher
# This script starts the web-based visualization demo on port 8188

echo "Starting HADM Visualization Demo..."
echo "================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "demo" ]; then
    echo "Error: demo directory not found. Please run this script from the root of the HADM project."
    exit 1
fi

# Check if required files exist
if [ ! -f "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py" ]; then
    echo "Warning: demo_local.py config not found"
fi

if [ ! -f "projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py" ]; then
    echo "Warning: demo_global.py config not found"
fi

# Check for model files
if [ ! -f "pretrained_models/HADM-L_0249999.pth" ]; then
    echo "Warning: HADM-L model not found at pretrained_models/HADM-L_0249999.pth"
    echo "You can set HADM_L_MODEL_PATH environment variable to specify a different path"
fi

if [ ! -f "pretrained_models/HADM-G_0249999.pth" ]; then
    echo "Warning: HADM-G model not found at pretrained_models/HADM-G_0249999.pth"
    echo "You can set HADM_G_MODEL_PATH environment variable to specify a different path"
fi

# Set default environment variables if not set
export HADM_L_MODEL_PATH=${HADM_L_MODEL_PATH:-"pretrained_models/HADM-L_0249999.pth"}
export HADM_G_MODEL_PATH=${HADM_G_MODEL_PATH:-"pretrained_models/HADM-G_0249999.pth"}
export EVA02_BACKBONE_PATH=${EVA02_BACKBONE_PATH:-"pretrained_models/eva02_L_coco_det_sys_o365.pth"}

echo "Environment variables:"
echo "  HADM_L_MODEL_PATH: $HADM_L_MODEL_PATH"
echo "  HADM_G_MODEL_PATH: $HADM_G_MODEL_PATH"
echo "  EVA02_BACKBONE_PATH: $EVA02_BACKBONE_PATH"
echo "================================================================"

# Parse command line arguments
HOST="0.0.0.0"
PORT="8188"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST     Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT     Port to bind to (default: 8188)"
            echo "  --reload        Enable auto-reload for development"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  HADM_L_MODEL_PATH    Path to HADM-L model file"
            echo "  HADM_G_MODEL_PATH    Path to HADM-G model file"
            echo "  EVA02_BACKBONE_PATH  Path to EVA02 backbone model file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting web server on $HOST:$PORT"
echo "Open your browser to http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"
echo "================================================================"

# Run the demo
python3 start_demo.py --host "$HOST" --port "$PORT" $RELOAD 