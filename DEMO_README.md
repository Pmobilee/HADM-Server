# HADM Visualization Demo

This directory contains scripts to run the HADM (Human Artifact Detection Models) visualization demo.

## Overview

The HADM project includes two main models:
- **HADM-L (Local)**: Detects local human artifacts like faces, torsos, arms, legs, hands, and feet
- **HADM-G (Global)**: Detects global human artifacts like missing or extra body parts

## Configuration Files

The models use these configuration files:
- `projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py` - Configuration for HADM-L
- `projects/ViTDet/configs/eva2_o365_to_coco/demo_global.py` - Configuration for HADM-G

These are detectron2 LazyConfig files that define the model architecture, training parameters, and inference settings.

## Demo Scripts

### 1. Web-based Visualization Demo (Recommended)

**File**: `start_demo.py` and `start_demo.sh`

This launches a web-based interface on port 8188 where you can upload images and see detection results.

```bash
# Using the shell script (recommended)
./start_demo.sh

# Or using Python directly
python3 start_demo.py

# Custom host/port
./start_demo.sh --host 0.0.0.0 --port 8188
```

**Features**:
- Web-based interface accessible via browser
- Drag-and-drop image upload
- Real-time visualization of detection results
- Support for both HADM-L and HADM-G models
- Runs on port 8188 by default

**Access**: Open your browser to `http://localhost:8188`

### 2. Command-line Demo

**File**: `run_demo.py`

This uses the existing `demo/demo.py` script for command-line inference.

```bash
# Run HADM-L on a single image
python3 run_demo.py --input demo/images/test6.jpeg --model local --output results/

# Run HADM-G on a directory of images
python3 run_demo.py --input test_images/ --model global --output results/

# With custom confidence threshold
python3 run_demo.py --input image.jpg --model local --confidence 0.7
```

**Options**:
- `--input`: Input image file or directory
- `--output`: Output directory for results (optional)
- `--model`: Model to use (`local` for HADM-L, `global` for HADM-G)
- `--confidence`: Confidence threshold (default: 0.5)

## Model Files

The scripts expect model files in the `pretrained_models/` directory:
- `pretrained_models/HADM-L_0249999.pth` - HADM-L model weights
- `pretrained_models/HADM-G_0249999.pth` - HADM-G model weights
- `pretrained_models/eva02_L_coco_det_sys_o365.pth` - EVA02 backbone weights

## Environment Variables

You can customize model paths using environment variables:

```bash
export HADM_L_MODEL_PATH="/path/to/your/HADM-L_model.pth"
export HADM_G_MODEL_PATH="/path/to/your/HADM-G_model.pth"
export EVA02_BACKBONE_PATH="/path/to/your/eva02_backbone.pth"
```

## Requirements

- Python 3.7+
- PyTorch
- detectron2
- FastAPI and uvicorn (for web demo)
- PIL/Pillow
- OpenCV
- NumPy

## Detection Classes

### HADM-L (Local Artifacts)
- Class 0: face
- Class 1: torso
- Class 2: arm
- Class 3: leg
- Class 4: hand
- Class 5: feet

### HADM-G (Global Artifacts)
- Class 0: human missing arm
- Class 1: human missing face
- Class 2: human missing feet
- Class 3: human missing hand
- Class 4: human missing leg
- Class 5: human missing torso
- Class 6: human with extra arm
- Class 7: human with extra face
- Class 8: human with extra feet
- Class 9: human with extra hand
- Class 10: human with extra leg
- Class 11: human with extra torso

## Troubleshooting

1. **Models not found**: Make sure model files are in the correct location or set environment variables
2. **Config files not found**: Ensure you're running from the root of the HADM project
3. **CUDA issues**: The demo will automatically use CPU if CUDA is not available
4. **Port already in use**: Use `--port` option to specify a different port for the web demo

## Examples

### Web Demo
```bash
# Start web demo on default port 8188
./start_demo.sh

# Start on custom port
./start_demo.sh --port 9000
```

### Command Line
```bash
# Process a single image with HADM-L
python3 run_demo.py --input demo/images/test6.jpeg --model local

# Process all images in a directory with HADM-G
python3 run_demo.py --input test_images/ --model global --output results/
```

The web demo provides the best user experience with real-time visualization, while the command-line demo is useful for batch processing and automation. 